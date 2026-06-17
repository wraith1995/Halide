#include "LowerWarpGroupTiles.h"

#include "CodeGen_GPU_Dev.h"
#include "Deinterleave.h"
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
    Expr base_index;  // the operand Load's index (scalarized to the tile origin later)
};

// Match reduce_add(widening_mul(LoadA, LoadB)) (peeling the f16->f32 casts the
// f32 accumulator inserts) and pull out the two shared operand buffers. We do NOT
// require a contiguous-k ramp: the core-matrix operand layout reads the k tile as a
// gather, and wgmma replaces the whole reduce wholesale (the IR's k-access pattern
// is discarded), so only the buffer + tile origin matter. base_index is whatever
// the Load index is (vector or scalar); tile_origin() scalarizes it.
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
        o.buffer = l->name;
        o.elem_type = l->type.element_of();
        o.base_index = l->index;
        return true;
    };
    return as_load(mul->a, a) && as_load(mul->b, b);
}

// Replace every Variable -- and every runtime buffer-accessor query (buffer min/
// max/extent/stride: translation-invariant for an intra-allocation offset) -- with
// 0, leaving the constant (thread/loop/buffer-independent) part of an index: the
// tile-origin offset within the operand's shared allocation. e.g. As (k/8)*64 +
// (k%8) + (m%8)*8 + (m/8)*128 -> 0. Buffer mins are zeroed because lets-substitution
// can fold a coordinate var (C.s1.m) into a min-bearing expr that Variable-zeroing
// alone would not reduce (e.g. (C.min.0 % 8)*8).
class ZeroVars : public IRMutator {
    using IRMutator::visit;
    Expr visit(const Variable *op) override {
        return make_zero(op->type);
    }
    Expr visit(const Call *op) override {
        if (op->name.find("_halide_buffer_get_") == 0) {
            return make_zero(op->type);
        }
        return IRMutator::visit(op);
    }

public:
    Expr run(const Expr &e) {
        return mutate(e);
    }
};

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
    std::map<std::string, Expr> lets;

    Stmt visit(const LetStmt *op) override {
        lets[op->name] = op->value;
        Stmt s = IRMutator::visit(op);
        lets.erase(op->name);
        return s;
    }

    // The shared element offset of the operand tile at vector lane `lane` (k=lane),
    // with all thread/loop vars + buffer mins zeroed: resolve CSE'd lets, zero vars,
    // then extract the given lane. lane 0 = the tile origin (k=0). lane 16 = the start
    // of the next wgmma K-chunk (used to derive the per-chunk descriptor stride for
    // K>16 accumulation -- the vectorized k tile is a core-matrix gather, so the chunk
    // stride is the affine address delta over 16 k-steps).
    Expr tile_origin_at_lane(Expr base, int lane) {
        for (int iter = 0; iter < 32; iter++) {
            Expr prev = base;
            base = substitute(lets, base);
            if (base.same_as(prev)) {
                break;
            }
        }
        Expr z = simplify(ZeroVars().run(simplify(base)));
        if (z.type().lanes() > 1) {
            z = extract_lane(z, lane);
        }
        return simplify(z);
    }
    Expr tile_origin(const Expr &base) {
        return tile_origin_at_lane(base, 0);
    }

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

        // The contraction is one wide reduce. m64n16k16 contracts K=16 per
        // wgmma.mma_async, so a width-K reduce = K/16 accumulating mma's (the async
        // chunks). The reduce input is the float32xK product, so its lanes == K.
        const int K_TILE = 16;
        int reduce_k = vr->value.type().lanes();
        if (reduce_k % K_TILE != 0 || reduce_k < K_TILE) {
            return IRMutator::visit(op);  // not a wgmma-shaped contraction
        }
        int n_chunks = reduce_k / K_TILE;

        if (getenv("HL_DEBUG_WGMMA")) {
            debug(0) << "[wgtile] rewriting tile store #" << frag_index
                     << " -> wgmma frag, buffer=" << op->name
                     << ", reduce_k=" << reduce_k << " n_chunks=" << n_chunks << "\n"
                     << "    A=" << a.buffer << " idx=" << a.base_index << "\n"
                     << "    B=" << b.buffer << " idx=" << b.base_index << "\n";
        }
        int i = frag_index++;
        if (i >= 8) {
            // More reduce stores than the m64n16k16 fragment has registers --
            // not the M0 shape; leave it for the generic fallback.
            return IRMutator::visit(op);
        }

        Expr lane = Variable::make(Int(32), thread_var);
        // The value: register i of the wgmma fragment. Carry i + n_chunks + each
        // operand's tile origin (lane-0 shared offset) and per-chunk descriptor stride
        // (the affine address delta over one K_TILE step, derived from lane K_TILE vs
        // lane 0 of the core-matrix gather). codegen advances the descriptor start by
        // chunk*stride and accumulates the chunks into the D fragment (scaleD carry).
        // The original `+ Load(C)` is dropped -- the fragment IS the accumulator.
        Expr base_a = tile_origin_at_lane(a.base_index, 0);
        Expr base_b = tile_origin_at_lane(b.base_index, 0);
        Expr stride_a = (n_chunks > 1)
                            ? simplify(tile_origin_at_lane(a.base_index, K_TILE) - base_a)
                            : Expr(0);
        Expr stride_b = (n_chunks > 1)
                            ? simplify(tile_origin_at_lane(b.base_index, K_TILE) - base_b)
                            : Expr(0);
        Expr load_a = Load::make(a.elem_type, a.buffer, base_a,
                                 Buffer<>(), Parameter(), const_true(), ModulusRemainder());
        Expr load_b = Load::make(b.elem_type, b.buffer, base_b,
                                 Buffer<>(), Parameter(), const_true(), ModulusRemainder());
        Expr frag = Call::make(op->value.type(), "wgmma_m64n16k16_f32",
                               {i, n_chunks, load_a, stride_a, load_b, stride_b},
                               Call::Intrinsic);

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
