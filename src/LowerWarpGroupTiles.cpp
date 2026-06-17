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

// True if e references a host-side buffer-query call (_halide_buffer_get_*). These are
// invalid inside a device kernel -- the device receives min/stride/extent as kernel-param
// Variables (e.g. C.stride.1), so we must NOT expand those param lets when rebuilding a
// device-side store index.
bool has_buffer_query(const Expr &e) {
    class V : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Call *op) override {
            if (op->name.find("_halide_buffer_get_") != std::string::npos) {
                found = true;
            }
            IRVisitor::visit(op);
        }

    public:
        bool found = false;
    } v;
    e.accept(&v);
    return v.found;
}

// Collect the output buffer's per-dimension stride from a device store index: the strides
// ride as kernel-param Variables named "<buf>.stride.<d>". The epilogue rewrite multiplies
// the fragment's tile-local (m,n) by these to flatten to the global output offset (so M/N
// tiling + arbitrary output strides Just Work; dim-0 defaults to 1 when contiguous/folded).
class StrideCollector : public IRVisitor {
    using IRVisitor::visit;
    void visit(const Variable *op) override {
        size_t p = op->name.rfind(".stride.");
        if (p != std::string::npos) {
            int d = atoi(op->name.c_str() + p + 8);
            strides[d] = Expr(op);
        }
    }

public:
    std::map<int, Expr> strides;
    Expr dim(int d) const {
        auto it = strides.find(d);
        return it == strides.end() ? Expr(1) : it->second;
    }
};

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

    // The output-tile epilogue base + strides, captured once per group from the first
    // (frag 0) store: out_base = the block tile corner (store index with the thread/warp
    // vars zeroed -> keeps the gpu_blocks offset + mins), out_stride_{m,n} = the output
    // buffer strides. The fragment store index is then out_base + frag_row_m*stride_m +
    // frag_col_n*stride_n -- so M/N output tiling over gpu_blocks Just Works.
    bool out_captured = false;
    Expr out_base, out_stride_m, out_stride_n;

    Stmt visit(const LetStmt *op) override {
        lets[op->name] = op->value;
        Stmt s = IRMutator::visit(op);
        lets.erase(op->name);
        return s;
    }

    // Expand CSE'd lets in an expr. With device_only, skip lets whose value is a host
    // buffer-query (min/stride/extent) -- those stay as kernel-param Variables so a rebuilt
    // device-side index does not pull host-only calls into the kernel (CUDA_ERROR_INVALID_PTX).
    Expr resolve_lets(Expr e, bool device_only = false) {
        std::map<std::string, Expr> use = lets;
        if (device_only) {
            for (auto it = use.begin(); it != use.end();) {
                if (has_buffer_query(it->second)) {
                    it = use.erase(it);
                } else {
                    ++it;
                }
            }
        }
        for (int it = 0; it < 64; it++) {
            Expr prev = e;
            e = substitute(use, e);
            if (e.same_as(prev)) {
                break;
            }
        }
        return e;
    }

    // The shared element offset of the operand tile at vector lane `lane` (k=lane),
    // with all thread/loop vars + buffer mins zeroed: resolve CSE'd lets, zero vars,
    // then extract the given lane. lane 0 = the tile origin (k=0). lane 16 = the start
    // of the next wgmma K-chunk (used to derive the per-chunk descriptor stride for
    // K>16 accumulation -- the vectorized k tile is a core-matrix gather, so the chunk
    // stride is the affine address delta over 16 k-steps).
    Expr tile_origin_at_lane(Expr base, int lane) {
        base = resolve_lets(base);
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
                frag_index = 0;       // reset the fragment register counter per group
                out_captured = false;  // re-capture the epilogue base/strides per group
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
                     << "    B=" << b.buffer << " idx=" << b.base_index << "\n"
                     << "    out store idx=" << simplify(resolve_lets(op->index)) << "\n";
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
        // Capture the output-tile epilogue base + strides once per group (frag 0 is the
        // store whose natural in-tile offset is 0 at the zeroed thread var, so zeroing the
        // thread/warp vars leaves exactly the block tile corner + buffer mins). The strides
        // come straight off the output buffer, so M/N tiling over gpu_blocks + arbitrary
        // output strides are handled by construction.
        if (!out_captured) {
            Expr ri = resolve_lets(op->index, /*device_only*/ true);
            StrideCollector sc;
            ri.accept(&sc);
            out_stride_m = sc.dim(0);
            out_stride_n = sc.dim(1);
            out_base = simplify(substitute(thread_var, make_zero(Int(32)),
                                           substitute(group_var, make_zero(Int(32)), ri)));
            out_captured = true;
        }

        // The global slot the hardware fragment map assigns to (lane, reg f): the tile-local
        // hardware (m,n), flattened through the output strides + offset to the block tile corner.
        auto frag_slot = [&](int f) {
            return simplify(out_base + frag_row_m(lane, f) * out_stride_m +
                            frag_col_n(lane, f) * out_stride_n);
        };

        // PROTOTYPE (1.x, HL_WGMMA_VECFRAG): the vector-native form -- emit the whole fragment
        // as ONE width-8 vector store with a non-affine per-lane scatter index, so we lower a
        // single vector group instead of reconstructing the tile from 8 unrolled scalar stores.
        // Codegen scalarizes the scatter to 8 st.global at the hardware fragment positions.
        if (getenv("HL_WGMMA_VECFRAG")) {
            if (i > 0) {
                return Evaluate::make(0);  // the other 7 frags are folded into the frag-0 vector store
            }
            Expr frag_vec = Call::make(op->value.type().with_lanes(8), "wgmma_m64n16k16_f32_frag8",
                                       {n_chunks, load_a, stride_a, load_b, stride_b}, Call::Intrinsic);
            std::vector<Expr> idx_lanes;
            for (int f = 0; f < 8; f++) {
                idx_lanes.push_back(frag_slot(f));
            }
            Expr index_vec = Shuffle::make_concat(idx_lanes);
            return Store::make(op->name, frag_vec, index_vec, op->param, const_true(8),
                               ModulusRemainder());
        }

        // Default (scalar) path: register i of the fragment, one scalar store per unrolled frag.
        Expr frag = Call::make(op->value.type(), "wgmma_m64n16k16_f32",
                               {i, n_chunks, load_a, stride_a, load_b, stride_b},
                               Call::Intrinsic);
        return Store::make(op->name, frag, frag_slot(i), op->param, const_true(),
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
