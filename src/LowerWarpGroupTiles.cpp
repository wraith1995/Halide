#include "LowerWarpGroupTiles.h"

#include "CodeGen_GPU_Dev.h"
#include "IRMutator.h"
#include "IROperator.h"
#include "IRPrinter.h"
#include "IRVisitor.h"
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

// Step 1a: detector. Walks the pre-fusion nest, tracking the enclosing GPU
// loops, and reports every store whose value is a WarpGroup tile reduce -- the
// structure the wgmma rewrite (1b) will consume. Transformation is identity.
class DetectWarpGroupTiles : public IRMutator {
    using IRMutator::visit;

    std::string block_var, group_var, thread_var;
    int warps_per_group = -1;

    Stmt visit(const For *op) override {
        std::string saved_block = block_var, saved_group = group_var, saved_thread = thread_var;
        int saved_wpg = warps_per_group;
        if (op->for_type == ForType::GPUBlock) {
            block_var = op->name;
        } else if (op->for_type == ForType::GPUThread) {
            if (op->warps_per_group >= 0 || op->realization == GPUVectorScope::WarpGroup) {
                group_var = op->name;
                warps_per_group = op->warps_per_group;
            } else {
                thread_var = op->name;
            }
        }
        Stmt s = IRMutator::visit(op);
        block_var = saved_block;
        group_var = saved_group;
        thread_var = saved_thread;
        warps_per_group = saved_wpg;
        return s;
    }

    // The two shared operands of a recognized tile reduce, parsed from the
    // reduce value `reduce_add(widening_mul(LoadA, LoadB))`. base_index is the
    // ramp base of each Load (= row * leading_dim); the recognizer turns these
    // + the shared allocation into wgmma matrix descriptors.
    struct Operand {
        std::string buffer;
        Expr base_index;  // ramp base of the k-contiguous row Load
        int k_lanes = 0;  // ramp lanes (= K of this wgmma step)
    };

    // Match reduce_add(widening_mul(LoadA, LoadB)) (with the f16->f32 casts the
    // accumulator imposes) and pull out the two shared operand Loads.
    static bool parse_operands(const VectorReduce *vr, Operand &a, Operand &b) {
        if (vr->op != VectorReduce::Add) {
            return false;
        }
        Expr prod = vr->value;
        const Mul *mul = prod.as<Mul>();
        if (!mul) {
            return false;
        }
        auto as_load = [](Expr e, Operand &o) -> bool {
            // Peel the widening f16->f32 cast(s) the accumulator inserts.
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
            o.base_index = r->base;
            o.k_lanes = r->lanes;
            return true;
        };
        return as_load(mul->a, a) && as_load(mul->b, b);
    }

    Stmt visit(const Store *op) override {
        // wgmma scope = an EXPLICIT gpu_warps group (warps_per_group > 0). A
        // derived group (== 0) is the thread-sub-dim decode model, not wgmma.
        if (!group_var.empty() && warps_per_group > 0) {
            if (const VectorReduce *vr = find_vector_reduce(op->value)) {
                Operand a, b;
                bool ok = parse_operands(vr, a, b);
                debug(0) << "[wgtile] WarpGroup tile-reduce store:\n"
                         << "  out buffer = " << op->name << "  index = " << op->index << "\n"
                         << "  parsed = " << (ok ? "YES" : "NO") << "\n";
                if (ok) {
                    debug(0) << "  A: buf=" << a.buffer << " base=" << a.base_index
                             << " k=" << a.k_lanes << "\n"
                             << "  B: buf=" << b.buffer << " base=" << b.base_index
                             << " k=" << b.k_lanes << "\n"
                             << "  lane(thread)var=" << thread_var << "\n";
                }
            }
        }
        return IRMutator::visit(op);
    }

public:
    Stmt run(const Stmt &s) {
        return mutate(s);
    }
};

}  // namespace

Stmt lower_warp_group_tiles(Stmt s, const Target &t) {
    // wgmma is sm_90+ only. On any other target the WarpGroup reduce keeps its
    // tag and falls back to the generic thread-level reduce in codegen.
    if (!t.has_feature(Target::CUDACapability90)) {
        return s;
    }
    if (getenv("HL_DEBUG_WGMMA")) {
        s = DetectWarpGroupTiles().run(s);
    }
    return s;
}

}  // namespace Internal
}  // namespace Halide
