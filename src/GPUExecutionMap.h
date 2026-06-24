#ifndef HALIDE_GPU_EXECUTION_MAP_H
#define HALIDE_GPU_EXECUTION_MAP_H

/** \file
 * The execution-mapping model (M1 of the FuseGPUThreadLoops re-architecture,
 * research/exec_mapping_model.md / fusegpu_rearch_plan.md C.0).
 *
 * One source of truth for "which lanes of a GPU block execute a statement / must
 * rendezvous." Today every sync pass (InjectThreadBarriers, LowerGPUWarpAsyncFork,
 * the partitioner) re-derives this from loop+guard structure independently and they
 * disagree at the hard edges (the deadlock family). This models the block's thread
 * space once and turns every sync decision into a query:
 *   - barrier scope  = scope(join(active(producer), active(consumer)))
 *   - participant ct = count(...)
 *   - completion wait scope = scope(active(consumer))
 *
 * The model is pure analysis (NFC): nothing consumes it yet. M2 migrates
 * InjectThreadBarriers onto it (byte-identical PTX); M3 the ring; M4 TMA falls out.
 */

#include <map>
#include <string>
#include <vector>

#include "Expr.h"
#include "Interval.h"

namespace Halide {
namespace Internal {

struct For;

/** The execution-resource lattice: Thread < Warp < WarpGroup < Block < Cluster < Grid.
 * This is SyncScope (Expr.h) widened with the two ends the deadlock analysis needs:
 * Thread ("no barrier — same lane(s), program order") and Grid. The selector
 * (LowerSyncRequirements) already dispatches Warp/WarpGroup/Block/Cluster; the model
 * only adds the classification *into* this lattice. */
enum class ExecScope {
    Thread,
    Warp,
    WarpGroup,
    Block,
    Cluster,
    Grid,
};

const char *exec_scope_name(ExecScope s);

/** One axis of the block's thread-index space (a GPUThread x/y/z loop, a GPULane loop,
 * or an injected warp-group dimension). Keyed by the loop variable name so guards
 * (which reference that name) narrow it directly. */
struct ExecAxis {
    std::string var;       // the loop variable name (== how IfThenElse guards reference it)
    Expr min;              // loop min
    Expr extent;           // loop extent (number of units along this axis)
    int lane_weight = 1;   // hardware lanes spanned by ONE unit: 1 for a thread/lane axis,
                           // warps_per_group*warp_size for a warp-group axis.
    bool is_warp_group = false;
};

/** A region of the block's thread space: a per-axis box. An axis absent from
 * `narrowed` means "the full extent of that axis" (every unit participates).
 * `empty` marks a peeled / impossible branch (no lanes). */
struct ActiveSet {
    std::map<std::string, Interval> narrowed;  // axis var name -> narrowed interval
    bool empty = false;

    static ActiveSet whole_block() {
        return ActiveSet{};
    }
};

/** The execution-mapping model for one GPU kernel (one launch / outermost gpu_block
 * nest). Pre-scans the kernel for all thread/lane/warp-group axes (the geometry), then
 * tracks the current ActiveSet as a consuming pass descends the IR (enter_for /
 * enter_if / pop, or RAII via ScopedNarrow). Read current() at any statement; ask
 * scope()/count()/join()/same_single_lane() to drive sync decisions. */
class ExecMap {
public:
    /** Build the model: discover the thread/lane/warp-group axes of `kernel`. */
    explicit ExecMap(const Stmt &kernel, int warp_size = 32);

    /** The discovered axes (the block geometry). */
    const std::vector<ExecAxis> &axes() const {
        return axes_;
    }
    /** Total hardware lanes in the block = product over axes of extent*lane_weight. */
    Expr block_lanes() const;

    // --- descent tracking (a consuming pass calls these as it walks) ---
    void enter_for(const For *op);          // narrow by a GPU loop axis (no-op for non-GPU/unknown)
    void enter_if(const Expr &condition);   // narrow by a recognized guard (no-op if unrecognized)
    void pop();                             // undo the last enter_* (LIFO)
    const ActiveSet &current() const {
        return stack_.back();
    }

    // --- queries (pure; usable on any ActiveSet from this model) ---
    /** Number of hardware lanes in the region. */
    Expr count(const ActiveSet &a) const;
    /** Classify the region into the lattice (coarsen UP — the mechanism must cover it). */
    ExecScope scope(const ActiveSet &a) const;
    /** Smallest box containing both (per-axis interval hull). Conservative (>= union). */
    static ActiveSet join(const ActiveSet &a, const ActiveSet &b);
    /** Both regions are the SAME single lane => Thread scope, NO barrier (program order). */
    bool same_single_lane(const ActiveSet &a, const ActiveSet &b) const;

    /** Whether `condition` is a guard form the model recognizes (else enter_if is a no-op
     * and the active set is conservatively NOT narrowed — more sync, never less). */
    bool recognizes_guard(const Expr &condition) const;

private:
    std::vector<ExecAxis> axes_;            // geometry, in discovery (outer->inner) order
    std::vector<ActiveSet> stack_;          // descent stack; back() == current
    int warp_size_;

    const ExecAxis *axis_for_var(const std::string &var) const;
    // Narrow `a` by `var in iv`; returns the narrowed set (empty if disjoint).
    ActiveSet narrow(const ActiveSet &a, const std::string &var, const Interval &iv) const;
    // Effective interval of an axis in `a` ([min,min+extent-1] if not narrowed).
    Interval axis_interval(const ActiveSet &a, const ExecAxis &ax) const;
};

}  // namespace Internal
}  // namespace Halide

#endif
