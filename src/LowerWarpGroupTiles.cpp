#include "LowerWarpGroupTiles.h"

#include "CodeGen_GPU_Dev.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "IRVisitor.h"
#include "Simplify.h"
#include "Substitute.h"
#include "Target.h"

namespace Halide {
namespace Internal {

namespace {

// Find the (first) VectorReduce in an expr. Pre-fusion the reduce is not yet
// tagged (TagGPUVectorScope stamps it inside fuse_gpu_thread_loops, after this
// pass), so we instead key on the enclosing gpu_warps For's realization +
// warps_per_group -- the same pre-fusion signal TagGPUVectorScope reads.
class FindVectorReduce : public IRVisitor {
    using IRVisitor::visit;
    void visit(const VectorReduce *op) override {
        if (!found) {
            found = op;
        }
        IRVisitor::visit(op);
    }

public:
    const VectorReduce *found = nullptr;
};

const VectorReduce *find_vector_reduce(const Expr &e) {
    FindVectorReduce f;
    e.accept(&f);
    return f.found;
}

// The two shared operands of a recognized tile reduce, parsed from the reduce
// value reduce_add(widening_mul(LoadA, LoadB)).
struct Operand {
    std::string buffer;
    Type elem_type;
};

// Match reduce_add(widening_mul(LoadA, LoadB)) (peeling the f16->f32 casts the
// f32 accumulator inserts) and pull out the two shared operand buffers.
bool parse_operands(const VectorReduce *vr, Operand &a, Operand &b) {
    if (vr->op != VectorReduce::Add) {
        return false;
    }
    const Mul *mul = vr->value.as<Mul>();
    if (!mul) {
        return false;
    }
    auto as_load = [](Expr e, Operand &o) -> bool {
        while (const Cast *c = e.as<Cast>()) {
            e = c->value;
        }
        const Load *l = e.as<Load>();
        if (!l) {
            return false;
        }
        const Ramp *r = l->index.as<Ramp>();
        if (!r || !is_const_one(r->stride)) {
            return false;
        }
        o.buffer = l->name;
        o.elem_type = l->type.element_of();
        return true;
    };
    return as_load(mul->a, a) && as_load(mul->b, b);
}

// The Hopper wgmma.m64n16k16 .f32 accumulator fragment map: which (row m, col
// n) of the 64x16 output tile the per-thread register `i` (0..7) holds, as a
// function of the warp-group thread id (lane). 4 warps x 32 lanes x 8 regs =
// 1024 = 64x16. FIRST-GUESS formula -- the exact bit layout is hardware-pinned
// and gets validated/corrected against the f64 oracle on H100 (see
// research/gpu_recognizer_design.md S5b step 3). The recognizer OWNS this map
// (it is the one non-affine, schedule-independent detail).
Expr frag_row_m(const Expr &lane, int i) {
    Expr warp = lane / 32;
    Expr l = lane % 32;
    return warp * 16 + (l / 4) + 8 * ((i / 2) % 2);
}
Expr frag_col_n(const Expr &lane, int i) {
    Expr l = lane % 32;
    return (l % 4) * 2 + (i % 2) + 8 * (i / 4);
}

// Rewrite recognized WarpGroup tile reduces into the wgmma collective: each of
// the (unrolled) per-thread reduce stores becomes a store of one wgmma fragment
// register to the global slot the hardware map assigns, at a NON-AFFINE index
// keyed on the lane. Codegen emits the wgmma.mma_async once per group and
// extracts register i; the per-thread iteration is preserved (this is NOT a
// fragment-loop rebuild) -- only WHERE each result lands is redirected.
class RewriteWarpGroupTiles : public IRMutator {
    using IRMutator::visit;

    std::string group_var, thread_var;
    int warps_per_group = -1;
    int frag_index = 0;

    Stmt visit(const For *op) override {
        std::string saved_group = group_var, saved_thread = thread_var;
        int saved_wpg = warps_per_group, saved_frag = frag_index;
        bool is_group = false;
        if (op->for_type == ForType::GPUThread) {
            if (op->warps_per_group >= 0 || op->realization == GPUVectorScope::WarpGroup) {
                group_var = op->name;
                warps_per_group = op->warps_per_group;
                is_group = true;
                frag_index = 0;  // reset the fragment register counter per group
            } else {
                thread_var = op->name;
            }
        }
        Stmt s = IRMutator::visit(op);
        group_var = saved_group;
        thread_var = saved_thread;
        warps_per_group = saved_wpg;
        if (!is_group) {
            frag_index = saved_frag;
        }
        return s;
    }

    Stmt visit(const Store *op) override {
        // wgmma scope = an EXPLICIT gpu_warps group (warps_per_group > 0).
        if (group_var.empty() || warps_per_group <= 0 || thread_var.empty()) {
            return IRMutator::visit(op);
        }
        const VectorReduce *vr = find_vector_reduce(op->value);
        if (!vr) {
            return IRMutator::visit(op);
        }
        Operand a, b;
        if (!parse_operands(vr, a, b)) {
            return IRMutator::visit(op);
        }

        if (getenv("HL_DEBUG_WGMMA")) {
            debug(0) << "[wgtile] rewriting tile store #" << frag_index
                     << " -> wgmma frag, buffer=" << op->name << "\n";
        }
        int i = frag_index++;
        if (i >= 8) {
            // More reduce stores than the m64n16k16 fragment has registers --
            // not the M0 shape; leave it for the generic fallback.
            return IRMutator::visit(op);
        }

        Expr lane = Variable::make(Int(32), thread_var);
        // The value: register i of the wgmma fragment. Carry i + the two shared
        // operand buffers (as base Loads codegen introspects for the matrix
        // descriptors). scaleD=0 (overwrite) -- M0 is a single k=16 step, so we
        // drop the `+ Load(C)` accumulation of the original store value.
        Expr load_a = Load::make(a.elem_type, a.buffer, 0, Buffer<>(), Parameter(),
                                 const_true(), ModulusRemainder());
        Expr load_b = Load::make(b.elem_type, b.buffer, 0, Buffer<>(), Parameter(),
                                 const_true(), ModulusRemainder());
        Expr frag = Call::make(op->value.type(), "wgmma_m64n16k16_f32",
                               {i, load_a, load_b}, Call::Intrinsic);

        // The index: the global slot the hardware map assigns to (lane, reg i).
        // M0 ASSUMPTION (single block at origin, dense 64x16 output, m
        // contiguous): C offset = m + n*64. Generalize later by reading the
        // output strides + block/min from the original store index.
        Expr index = simplify(frag_row_m(lane, i) + frag_col_n(lane, i) * 64);

        return Store::make(op->name, frag, index, op->param, const_true(),
                           ModulusRemainder());
    }

public:
    Stmt run(const Stmt &s) {
        return mutate(s);
    }
};

}  // namespace

Stmt lower_warp_group_tiles(Stmt s, const Target &t) {
    // wgmma is sm_90+ only. On any other target the WarpGroup reduce keeps its
    // tag and falls back to the generic thread-level reduce in codegen, so the
    // same schedule stays correct everywhere.
    if (!t.has_feature(Target::CUDACapability90)) {
        return s;
    }
    return RewriteWarpGroupTiles().run(s);
}

}  // namespace Internal
}  // namespace Halide
