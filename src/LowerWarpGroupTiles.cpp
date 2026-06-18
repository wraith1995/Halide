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

#include <set>

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
    const std::set<std::string> *only;  // if non-null, zero ONLY these var names (else all)
    Expr visit(const Variable *op) override {
        if (only && !only->count(op->name)) {
            return op;  // keep -- e.g. the serial-loop (ring_buffer ko%n) rotation
        }
        return make_zero(op->type);
    }
    Expr visit(const Call *op) override {
        if (op->name.find("_halide_buffer_get_") == 0) {
            return make_zero(op->type);
        }
        return IRMutator::visit(op);
    }

public:
    explicit ZeroVars(const std::set<std::string> *o = nullptr) : only(o) {}
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

// True if e reads (a Load from) one of the named buffers. Used by mainloop 1b to spot the
// `C = prod` epilogue copy: a store whose value loads a recorded wgmma register accumulator.
bool loads_one_of(const Expr &e, const std::set<std::string> &names) {
    class V : public IRVisitor {
        using IRVisitor::visit;
        void visit(const Load *op) override {
            if (names.count(op->name)) {
                found = true;
            }
            IRVisitor::visit(op);
        }

    public:
        const std::set<std::string> &names;
        bool found = false;
        explicit V(const std::set<std::string> &n) : names(n) {}
    } v(names);
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

// 1.1 auto-layout: re-encode a NATURAL (dense) operand shared-store index into the
// core-matrix tiling the wgmma descriptor needs. The producer store index is a clean
// 0-based allocation-relative offset; decode it to (role, k) using the operand's K-stride
// (role = the M/N dim), then re-encode core-matrix: ki(stride 1) + role%8 (stride 8) +
// ko(stride 64) + role/8 (stride 8*K = 128*n_chunks). k_stride tells the orientation:
// k_stride==1 -> k is the fast dim (k = idx % K); else role is fast (role = idx % k_stride).
Expr core_matrix_reencode(const Expr &idx, int k_stride, int reduce_k) {
    int n_chunks = reduce_k / 16;
    Expr in = idx;
    // Step 2 (ring_buffer): the dense index may carry a (ko%n)*slot ring/slot offset (storage
    // hoisted above the mainloop, rotated by the serial loop). The core-matrix re-encode is a
    // permutation WITHIN one slot, so the ring offset must be split off and re-added in core-
    // matrix slot units -- NOT fed through the (role,k) decode, which would scramble it (folding
    // (ko%n)*slot into k/8*64 yields the wrong slot stride for k_stride!=1: 128 vs slot=k*K). The
    // core-matrix slot has the same element count as the dense slot (it is a permutation), so the
    // slot offset is identical on both sides; the consumer descriptor keeps the same (ko%n)*slot.
    Expr ring = make_zero(Int(32));
    if (k_stride != 1) {
        int slot = k_stride * reduce_k;  // dense (== core-matrix) slot element count
        ring = (in / slot) * slot;       // 0 in the non-ring case (in < slot)
        in = in % slot;                  // intra-slot offset to re-encode
    }
    // (k_stride==1 needs no split: role/8*128 already advances by exactly one slot per ring step
    // for the m64n16k16 N=16 tile, so the ring offset survives the decode unchanged.)
    Expr role, k;
    if (k_stride == 1) {
        k = in % reduce_k;
        role = in / reduce_k;
    } else {
        role = in % k_stride;
        k = in / k_stride;
    }
    return simplify(ring + (k % 8) + (role % 8) * 8 + (k / 8) * 64 + (role / 8) * (128 * n_chunks));
}

// Pre-scan: find the natural (dense) shared operands of the wgmma tile reduce and record
// their K-stride, so the producer's shared stores can be re-encoded to core-matrix. Only
// operands whose reduce-gather is a plain Ramp are "natural" -- a hand-matched core-matrix
// operand has a non-Ramp gather and is left alone (the 1.2 "already matches" case).
class CollectNaturalOperands : public IRVisitor {
    using IRVisitor::visit;
    int warps_per_group = -1;

    // Track the enclosing explicit gpu_warps (WarpGroup) group, like the mutator, so we only
    // re-layout the operands of a genuine wgmma tile reduce -- NOT every vectorized reduce in
    // the kernel (critical now that auto-layout is always on, not gated).
    void visit(const For *op) override {
        int saved = warps_per_group;
        if (op->for_type == ForType::GPUThread &&
            (op->warps_per_group > 0 || op->realization == GPUVectorScope::WarpGroup)) {
            warps_per_group = op->warps_per_group;
        }
        IRVisitor::visit(op);
        warps_per_group = saved;
    }

    void visit(const Store *op) override {
        if (warps_per_group > 0) {
            const VectorReduce *vr = find_vector_reduce(op->value);
            Operand a, b;
            if (vr && parse_operands(vr, a, b)) {
                int reduce_k = vr->value.type().lanes();
                for (const Operand *o : {&a, &b}) {
                    // Only a plain-Ramp gather is a NATURAL dense operand to re-encode; a
                    // hand-matched core-matrix operand has a non-Ramp gather -> left alone.
                    const Ramp *r = o->base_index.as<Ramp>();
                    if (r) {
                        if (auto s = as_const_int(r->stride)) {
                            layouts[o->buffer] = {(int)*s, reduce_k};
                        }
                    }
                }
            }
        }
        IRVisitor::visit(op);
    }

public:
    std::map<std::string, std::pair<int, int>> layouts;  // buffer -> (k_stride, reduce_k)
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

    // 1.1 auto-layout: natural (dense) shared operands and their K-stride, populated by the
    // pre-scan. A producer store to one of these buffers gets its index re-encoded to the
    // core-matrix tiling. Empty unless HL_WGMMA_AUTOLAYOUT is set (prototype gate).
    std::map<std::string, std::pair<int, int>> natural_operands;

    // Mainloop 1b (HL_WGMMA_KCARRY, Option A): buffers that are wgmma REGISTER accumulators
    // (the `prod` Func). Their wgmma reduce store accumulates into the carried D fragment
    // (accum8, scaleD=1) instead of overwriting; their copy-out (`C = prod`) is the epilogue
    // that gets the non-affine frag map applied to the OUTPUT index. Populated as we rewrite
    // the accumulator's update (visited before the epilogue copy in the same block).
    std::set<std::string> wgmma_accumulators;
    // The accumulator's per-thread base index (frag 0's store index), captured per group so the
    // 8 scalar frag stores all read/write prod[base + i].
    Expr acc_base;

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

    // The shared element offset of the operand tile at vector lane `lane` (k=lane): resolve CSE'd
    // lets, zero ONLY the thread/warp-varying parts (per-lane variation within the warp group) +
    // any folded buffer-query, then extract the given lane. lane 0 = the tile origin (k=0). lane
    // 16 = the start of the next wgmma K-chunk (chunk-stride derivation for K>16).
    // KEY (step 2): serial-loop dependence is KEPT -- with ring_buffer(n) the operand tile lives
    // in slot (ko % n), so the origin carries a `(ko%n)*slot_size` term the descriptor must follow
    // across the mainloop. Zeroing only thread_var/group_var preserves it (NFC for the non-ring
    // case, where the operand address has no serial-loop term -- only thread vars + constants).
    Expr tile_origin_at_lane(Expr base, int lane) {
        base = resolve_lets(base);
        std::set<std::string> thread_vars;
        if (!thread_var.empty()) {
            thread_vars.insert(thread_var);
        }
        if (!group_var.empty()) {
            thread_vars.insert(group_var);
        }
        Expr z = simplify(ZeroVars(&thread_vars).run(simplify(base)));
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

    Stmt visit(const Atomic *op) override {
        // Mainloop 1b (HL_WGMMA_KCARRY): `.atomic()` is needed only to make `vectorize(ki)` on
        // the reduction legal at schedule time; the recognizer then REPLACES the atomic reduction
        // with the wgmma collective, whose result is a plain per-thread REGISTER store (no RMW
        // contention). Strip the now-vestigial Atomic so that store does not lower to an
        // (unsupported) atomic op on register/local memory. Only the wgmma accumulator update is
        // wrapped in Atomic under this schedule, so stripping is safe.
        if (getenv("HL_WGMMA_KCARRY")) {
            return mutate(op->body);
        }
        return IRMutator::visit(op);
    }

    Stmt visit(const Store *op) override {
        // 1.1 auto-layout: a store to a natural shared operand is the PRODUCER fill; re-encode
        // its (clean 0-based allocation-relative) index to the core-matrix tiling so the wgmma
        // descriptor reads it correctly -- no extra staging stage (the producer writes
        // core-matrix directly). Only the index changes; the value (the global load) is kept.
        auto it = natural_operands.find(op->name);
        if (it != natural_operands.end()) {
            Expr new_idx = core_matrix_reencode(op->index, it->second.first, it->second.second);
            if (getenv("HL_DEBUG_WGMMA")) {
                debug(0) << "[wgtile] auto-layout producer " << op->name
                         << " (k_stride=" << it->second.first << ") -> core-matrix\n";
            }
            Expr value = mutate(op->value);
            return Store::make(op->name, value, new_idx, op->param, op->predicate, op->alignment);
        }

        // wgmma scope = an EXPLICIT gpu_warps group (warps_per_group > 0).
        if (group_var.empty() || warps_per_group <= 0 || thread_var.empty()) {
            return IRMutator::visit(op);
        }

        // 1b epilogue (HL_WGMMA_KCARRY): a store whose value LOADS a recorded register
        // accumulator (`prod`) AND has no vector reduce is the `C = prod` copy-out (the
        // accumulator's OWN update `prod += ...` also loads prod, but carries the reduce -- that
        // is the accumulate path below, not the epilogue). prod holds D in frag-register order
        // (8 per thread); apply the non-affine frag map to the OUTPUT index so register f lands
        // at hardware (m,n). Only the index changes; the value (Load prod[frag f]) is kept.
        if (getenv("HL_WGMMA_KCARRY") && !wgmma_accumulators.empty() &&
            loads_one_of(op->value, wgmma_accumulators) && !find_vector_reduce(op->value)) {
            int i = frag_index++;
            if (i >= 8) {
                return IRMutator::visit(op);
            }
            Expr lane = Variable::make(Int(32), thread_var);
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
            Expr slot = simplify(out_base + frag_row_m(lane, i) * out_stride_m +
                                 frag_col_n(lane, i) * out_stride_n);
            if (getenv("HL_DEBUG_WGMMA")) {
                debug(0) << "[wgtile] 1b epilogue store #" << i << " " << op->name
                         << " <- accumulator -> frag slot\n";
            }
            return Store::make(op->name, mutate(op->value), slot, op->param, op->predicate,
                               op->alignment);
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
        // Per-chunk descriptor stride (advancing one K_TILE=16 step). For an AUTO-LAYOUT
        // operand the producer was re-encoded to core-matrix, so the chunk stride is the
        // core-matrix k-advance = K_TILE/8 = 2 ko, each ko = 64 elems = 128 elems -- NOT the
        // natural gather's dense k-stride (which would be K_TILE*role_extent). A hand-matched
        // operand already gathers core-matrix, so the gather delta gives the same 128.
        const int cm_chunk_stride = (K_TILE / 8) * 64;
        auto chunk_stride = [&](const Operand &o, const Expr &base) -> Expr {
            if (n_chunks <= 1) {
                return Expr(0);
            }
            if (natural_operands.count(o.buffer)) {
                return Expr(cm_chunk_stride);
            }
            return simplify(tile_origin_at_lane(o.base_index, K_TILE) - base);
        };
        Expr stride_a = chunk_stride(a, base_a);
        Expr stride_b = chunk_stride(b, base_b);
        Expr load_a = Load::make(a.elem_type, a.buffer, base_a,
                                 Buffer<>(), Parameter(), const_true(), ModulusRemainder());
        Expr load_b = Load::make(b.elem_type, b.buffer, base_b,
                                 Buffer<>(), Parameter(), const_true(), ModulusRemainder());

        // 1b accumulate (HL_WGMMA_KCARRY, Option A): the wgmma reduce store targets a register
        // accumulator (`prod`), NOT the global output. Emit ONE vector store of
        // accum8(Load(prod)) into prod's NATURAL per-thread slots (frag-register order, ramp
        // stride 1, NO frag map -- the frag map is applied later at the C=prod epilogue). D_in =
        // the current prod = the loop-carried accumulator; compute_at gives the carry across ko
        // and hoists the epilogue out of the loop. scaleD=1 (accumulate) lives in accum8.
        if (getenv("HL_WGMMA_KCARRY")) {
            wgmma_accumulators.insert(op->name);
            // The accumulator's per-thread base = frag 0's store index (frags 0..7 land at
            // base+0..7). Emit one SCALAR store per frag prod[base+i] = accum_reg(i, ...), like
            // the M1 scalar+cache path: the wgmma is cached (one per ko body) and seeds D from
            // prod's current value (D_in = Load(prod, base)); scaleD=1 accumulates. No vector
            // store to register memory -> no per-lane scalarization of the collective.
            if (i == 0) {
                acc_base = op->index;
            }
            Expr d_in = Load::make(op->value.type(), op->name, acc_base,
                                   Buffer<>(), op->param, const_true(), ModulusRemainder());
            Expr call = Call::make(op->value.type(), "wgmma_m64n16k16_f32_accum_reg",
                                   {i, d_in, n_chunks, load_a, stride_a, load_b, stride_b},
                                   Call::Intrinsic);
            if (getenv("HL_DEBUG_WGMMA")) {
                debug(0) << "[wgtile] 1b accumulate frag #" << i << " -> accum_reg into "
                         << op->name << " (n_chunks=" << n_chunks << ")\n";
            }
            return Store::make(op->name, call, op->index, op->param, op->predicate, op->alignment);
        }

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
            // Mainloop milestone 1a (HL_WGMMA_KACCUM): keep the `+C` read so D accumulates
            // across a ko serial reduction loop in global (running RMW per hardware slot). NFC
            // for the single-tile case -- the init stage zeroes C, so 0 + D = D. Each thread
            // owns the same frag slots across ko (frag map keys on lane, not ko) -> no race.
            Expr stored = frag_vec;
            if (getenv("HL_WGMMA_KACCUM")) {
                Expr c_old = Load::make(op->value.type().with_lanes(8), op->name, index_vec,
                                        Buffer<>(), op->param, const_true(8), ModulusRemainder());
                stored = c_old + frag_vec;
            }
            return Store::make(op->name, stored, index_vec, op->param, const_true(8),
                               ModulusRemainder());
        }

        // Default (scalar) path: register i of the fragment, one scalar store per unrolled frag.
        Expr frag = Call::make(op->value.type(), "wgmma_m64n16k16_f32",
                               {i, n_chunks, load_a, stride_a, load_b, stride_b},
                               Call::Intrinsic);
        // Mainloop 1a (HL_WGMMA_KACCUM): keep `+C` so D accumulates across ko (see vecfrag above).
        Expr stored = frag;
        if (getenv("HL_WGMMA_KACCUM")) {
            Expr c_old = Load::make(op->value.type(), op->name, frag_slot(i),
                                    Buffer<>(), op->param, const_true(), ModulusRemainder());
            stored = c_old + frag;
        }
        return Store::make(op->name, stored, frag_slot(i), op->param, const_true(),
                           ModulusRemainder());
    }

public:
    Stmt run(const Stmt &s) {
        // Auto-layout is always on: the pre-scan only records natural (plain-Ramp) operands of
        // an explicit gpu_warps tile reduce, so hand-matched (non-Ramp) operands are untouched.
        CollectNaturalOperands c;
        s.accept(&c);
        natural_operands = c.layouts;
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
