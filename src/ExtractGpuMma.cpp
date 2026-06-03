#include "ExtractGpuMma.h"

#include "IRMutator.h"
#include "IROperator.h"
#include "IRVisitor.h"
#include "Simplify.h"
#include "Substitute.h"
#include "Target.h"
#include "Util.h"

#include <initializer_list>
#include <set>
#include <utility>

namespace Halide {
namespace Internal {

namespace {

using std::string;
using std::vector;

// --- M0: fp16 inputs, fp32 accumulator, one mma.sync.m16n8k16 per warp. ---
// The warp owns a 16x8 output tile (no k-loop accumulation yet: K == 16, one mma).
// Operand fragments are gathered by *direct* per-lane shared loads using the
// documented mma.m16n8k16 fragment layout (ldmatrix is a later perf opt). The one
// thing this pins on hardware is the fragment + accumulator coordinate maps below.
constexpr int MMA_M = 16, MMA_N = 8, MMA_K = 16;

std::set<string> vars_in(const Expr &e) {
    struct C : public IRVisitor {
        std::set<string> vars;
        using IRVisitor::visit;
        void visit(const Variable *v) override {
            vars.insert(v->name);
        }
    } c;
    e.accept(&c);
    return c.vars;
}

// Evaluate an affine index Expr with some integer vars pinned, then simplify.
Expr eval_at(Expr e, const vector<std::pair<string, int>> &subs) {
    for (const auto &s : subs) {
        e = substitute(s.first, make_const(Int(32), s.second), e);
    }
    return simplify(e);
}

// Peel an optional Cast-to-f32 off an f16 Load. Returns the inner Load or null.
const Load *f16_load(const Expr &e) {
    const Expr *p = &e;
    if (const Cast *c = e.as<Cast>()) {
        p = &c->value;
    }
    const Load *l = p->as<Load>();
    if (l && l->type.is_float() && l->type.bits() == 16 && l->type.is_scalar()) {
        return l;
    }
    return nullptr;
}

// Is `v` the MAC value `Load(acc) + f16load_A * f16load_B`? If so, bind A,B.
bool is_mac(const Expr &v, const string &acc, const Load **A, const Load **B) {
    const Add *add = v.as<Add>();
    if (!add) {
        return false;
    }
    Expr prod = add->b;
    const Load *self = add->a.as<Load>();
    if (!(self && self->name == acc)) {
        // try the other order: Mul + Load(acc)
        self = add->b.as<Load>();
        if (!(self && self->name == acc)) {
            return false;
        }
        prod = add->a;
    }
    const Mul *mul = prod.as<Mul>();
    if (!mul) {
        return false;
    }
    const Load *la = f16_load(mul->a), *lb = f16_load(mul->b);
    if (!la || !lb) {
        return false;
    }
    *A = la;
    *B = lb;
    return true;
}

// Collect the reduce Store (acc += A*B) and the epilogue Store (out = acc) for a
// candidate accumulator allocation.
struct MmaFinder : public IRVisitor {
    string acc;
    const Store *reduce = nullptr;
    const Store *epilogue = nullptr;
    const Load *la = nullptr, *lb = nullptr;
    // The ProducerConsumer name wrapping the init/reduce and epilogue. This is the
    // Func name ("prod"), which differs from the storage name `acc` ("prod.0").
    string cur_pc, pc_name;
    using IRVisitor::visit;
    void visit(const ProducerConsumer *op) override {
        string old = cur_pc;
        cur_pc = op->name;
        IRVisitor::visit(op);
        cur_pc = old;
    }
    void visit(const Store *s) override {
        if (s->name == acc) {
            const Load *A, *B;
            if (is_mac(s->value, acc, &A, &B)) {
                reduce = s;
                la = A;
                lb = B;
                pc_name = cur_pc;
            }
        } else if (const Load *l = s->value.as<Load>(); l && l->name == acc) {
            epilogue = s;
        }
        IRVisitor::visit(s);
    }
};

struct GpuMma : public IRMutator {
    using IRMutator::visit;

    // Name of the warp lane var (a gpu_thread loop of extent 32), if we're in one.
    string lane;

    // Geometry of the current match (set when we rewrite an accumulator alloc).
    bool matched = false;
    string acc_name;  // storage name, e.g. "prod.0"
    string pc_name;   // ProducerConsumer (Func) name, e.g. "prod"
    // A: out[m,n] += A[m,k]*B[k,n]; indices As[base_a + m*da_m + k*da_k] etc.
    string a_name, b_name, c_name;
    Buffer<> a_image, b_image;
    Parameter a_param, b_param, c_param;
    Expr base_a, da_m, da_k;
    Expr base_b, db_k, db_n;
    Expr base_c, cm, cn;

    Stmt visit(const For *op) override {
        auto ext = as_const_int(simplify(op->extent()));
        if (is_gpu(op->for_type) && is_const_zero(op->min) && ext && *ext == 32) {
            ScopedValue<string> s(lane, op->name);
            return IRMutator::visit(op);
        }
        return IRMutator::visit(op);
    }

    // Build a scalar f16 load of A at (m,k) / B at (k,n).
    Expr load_a(const Expr &m, const Expr &k) {
        return Load::make(Float(16), a_name, base_a + m * da_m + k * da_k,
                          a_image, a_param, const_true(), ModulusRemainder());
    }
    Expr load_b(const Expr &k, const Expr &n) {
        return Load::make(Float(16), b_name, base_b + k * db_k + n * db_n,
                          b_image, b_param, const_true(), ModulusRemainder());
    }

    Stmt visit(const Allocate *op) override {
        // Only fire on a register fp32 accumulator inside a 32-lane warp, on sm_80+.
        if (lane.empty() || op->memory_type != MemoryType::Register ||
            !(op->type.is_float() && op->type.bits() == 32)) {
            return IRMutator::visit(op);
        }
        Stmt flat = substitute_in_all_lets(op->body);
        MmaFinder f;
        f.acc = op->name;
        flat.accept(&f);
        if (!f.reduce || !f.epilogue) {
            return IRMutator::visit(op);
        }

        // Identify the loop vars from the operand indices. k is shared by A and B;
        // m is A-only (and not the lane). n is encoded in the lane (B's column).
        Expr ia = f.la->index, ib = f.lb->index, ec = f.epilogue->index;
        std::set<string> va = vars_in(ia), vb = vars_in(ib);
        string k_var, m_var;
        for (const string &v : va) {
            if (v == lane) {
                continue;
            }
            if (vb.count(v)) {
                k_var = v;
            } else {
                m_var = v;
            }
        }
        if (k_var.empty() || m_var.empty() || f.pc_name.empty()) {
            return IRMutator::visit(op);
        }
        // Epilogue loop var (the r in `for r in 0..3`): the accumulator-load index
        // of the epilogue is `acc[(lane*4+r)%16]`, so its only non-lane var is r.
        string r_var;
        if (const Load *al = f.epilogue->value.as<Load>()) {
            for (const string &v : vars_in(al->index)) {
                if (v != lane) {
                    r_var = v;
                }
            }
        }
        if (r_var.empty()) {
            return IRMutator::visit(op);
        }

        // Extract affine coefficients (constants for shared tiles; symbolic for C).
        auto AT = [](const Expr &e, std::initializer_list<std::pair<string, int>> s) {
            return eval_at(e, vector<std::pair<string, int>>(s));
        };
        Expr base_a_ = AT(ia, {{m_var, 0}, {k_var, 0}, {lane, 0}});
        Expr da_m_ = simplify(AT(ia, {{m_var, 1}, {k_var, 0}, {lane, 0}}) - base_a_);
        Expr da_k_ = simplify(AT(ia, {{m_var, 0}, {k_var, 1}, {lane, 0}}) - base_a_);
        Expr base_b_ = AT(ib, {{k_var, 0}, {lane, 0}});
        Expr db_k_ = simplify(AT(ib, {{k_var, 1}, {lane, 0}}) - base_b_);
        // n == lane/4 in the B fragment: bumping lane by 4 advances n by 1.
        Expr db_n_ = simplify(AT(ib, {{k_var, 0}, {lane, 4}}) - base_b_);

        // C address as a function of (m,n): base + m*cm + n*cn. The schedule maps
        // (lane,r) linearly to (m,n) = ((lane*4+r)%16, (lane*4+r)/16); pick points.
        Expr base_c_ = AT(ec, {{lane, 0}, {r_var, 0}});               // (m,n)=(0,0)
        Expr cm_ = simplify(AT(ec, {{lane, 0}, {r_var, 1}}) - base_c_);  // (1,0)
        Expr cn_ = simplify(AT(ec, {{lane, 4}, {r_var, 0}}) - base_c_);  // (0,1)

        // Commit the match; recurse so the inner ProducerConsumer nodes rewrite.
        ScopedValue<bool> m1(matched, true);
        ScopedValue<string> m2(acc_name, op->name);
        ScopedValue<string> m2b(pc_name, f.pc_name);
        ScopedValue<string> m3(a_name, f.la->name), m4(b_name, f.lb->name),
            m5(c_name, f.epilogue->name);
        ScopedValue<Buffer<>> m6(a_image, f.la->image), m7(b_image, f.lb->image);
        ScopedValue<Parameter> m8(a_param, f.la->param), m9(b_param, f.lb->param),
            m10(c_param, f.epilogue->param);
        ScopedValue<Expr> e1(base_a, base_a_), e2(da_m, da_m_), e3(da_k, da_k_);
        ScopedValue<Expr> e4(base_b, base_b_), e5(db_k, db_k_), e6(db_n, db_n_);
        ScopedValue<Expr> e7(base_c, base_c_), e8(cm, cm_), e9(cn, cn_);

        Stmt body = mutate(op->body);
        // Shrink the accumulator to the 4 fp32 D-fragment registers per lane.
        return Allocate::make(op->name, op->type, op->memory_type,
                              {make_const(Int(32), 4)}, op->condition, body);
    }

    Stmt visit(const ProducerConsumer *op) override {
        if (!matched || op->name != pc_name) {
            return IRMutator::visit(op);
        }
        Expr ln = Variable::make(Int(32), lane);
        Expr gid = ln / 4, tid2 = (ln % 4) * 2;

        if (op->is_producer) {
            // Gather the A (4 regs = 8 halfs) and B (2 regs = 4 halfs) fragments per
            // the mma.m16n8k16 .f16 layout, then one mma into the D registers.
            vector<Expr> a = {
                load_a(gid, tid2), load_a(gid, tid2 + 1),
                load_a(gid + 8, tid2), load_a(gid + 8, tid2 + 1),
                load_a(gid, tid2 + 8), load_a(gid, tid2 + 9),
                load_a(gid + 8, tid2 + 8), load_a(gid + 8, tid2 + 9)};
            vector<Expr> b = {
                load_b(tid2, gid), load_b(tid2 + 1, gid),
                load_b(tid2 + 8, gid), load_b(tid2 + 9, gid)};
            Expr afrag = Shuffle::make_concat(a);
            Expr bfrag = Shuffle::make_concat(b);
            Expr cinit = Broadcast::make(make_zero(Float(32)), 4);
            Expr d = Call::make(Float(32, 4), "gpu_mma_f16_f32",
                                {afrag, bfrag, cinit}, Call::Intrinsic);
            return Store::make(acc_name, d, Ramp::make(make_const(Int(32), 0), make_const(Int(32), 1), 4),
                               Parameter(), const_true(4), ModulusRemainder());
        }

        // Consumer: scatter the 4 D registers to C at the accumulator-fragment coords
        //   D0->(gid, 2*tid)  D1->(gid, 2*tid+1)  D2->(gid+8, 2*tid)  D3->(gid+8, 2*tid+1)
        vector<Stmt> stores;
        for (int i = 0; i < 4; i++) {
            Expr mm = (i < 2) ? gid : gid + 8;
            Expr nn = (i % 2 == 0) ? tid2 : tid2 + 1;
            Expr dval = Load::make(Float(32), acc_name, make_const(Int(32), i),
                                   Buffer<>(), Parameter(), const_true(), ModulusRemainder());
            Expr addr = base_c + mm * cm + nn * cn;
            stores.push_back(Store::make(c_name, dval, addr, c_param,
                                         const_true(), ModulusRemainder()));
        }
        return Block::make(stores);
    }
};

}  // namespace

Stmt extract_gpu_mma(const Stmt &s, const Target &t) {
    if (!t.has_feature(Target::CUDA) ||
        t.get_cuda_capability_lower_bound() < 80) {
        return s;  // P-G: older GPUs keep the ordinary fma reduction.
    }
    return GpuMma()(s);
}

}  // namespace Internal
}  // namespace Halide
