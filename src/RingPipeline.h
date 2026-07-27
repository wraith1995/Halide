#ifndef HALIDE_RING_PIPELINE_H
#define HALIDE_RING_PIPELINE_H

/** \file
 * The hazard arithmetic of ONE ring buffer, resolved from the schedule.
 *
 * See research/async_storage_model.md §12. A ring buffer is a *storage node* with two incident
 * async edges -- a FILL edge (cp.async / TMA writing the slot) and a DRAIN edge (wgmma reading it) --
 * and each async edge carries not only the `async?` bit but a **lag** \c lambda: the number of
 * iterations by which that edge's retirement trails its issue (its completion group's in-flight
 * depth). The two §5 hazard edges then BRACKET the producer's lead D from opposite sides:
 *
 *     full / RAW  (consumer must not read before the fill retired)   :  D >= lambda_in
 *     empty / WAR (producer must not overwrite a slot still read)    :  D <= Q - 1 - lambda_out
 *     => feasible iff  Q >= lambda_in + lambda_out + 1               (the §8.3 inequality)
 *
 * So `Q >= N_c + N_w + 1` is not a separate pipeline rule -- it is the FEASIBILITY of that
 * interval, and the software-pipeline skew `D = Q-1-N_w` is simply its upper endpoint.
 *
 * Lags only exist for **ordinal** completion carriers (a group counter: `cp.async.wait_group`,
 * `wgmma.wait_group`), where completion is FIFO by issue order and a hazard can only be resolved
 * by loop position. An **addressed** carrier (a per-slot mbarrier: TMA, cluster multicast) names
 * the exact slot and needs no lag.
 *
 * Everything below the `derived` comments is a QUERY over schedule facts, never a knob. This
 * struct exists so the uniform (SoftwarePipeline) and warp-specialized (FuseGPUThreadLoops)
 * lowerings read ONE source of truth: the same lambda_out that the uniform path spends on
 * shrinking the lead, the warp-spec path spends on retiming the consumer's slot release.
 */

#include <map>
#include <string>

#include "Function.h"
#include "IROperator.h"
#include "Schedule.h"
#include "Simplify.h"
#include "Util.h"

namespace Halide {
namespace Internal {

struct RingPipeline {
    int Q = 1;         ///< ring_buffer(Q) -- the storage node's fold factor.
    int lam_in = 0;    ///< fill-edge lag (cp.async/TMA in flight); the model's N_c.
    int lam_out = 0;   ///< drain-edge lag (wgmma groups in flight); the model's N_w.

    // --- derived: the two hazard edges bracket the producer lead D (§12.3) ---

    /** full / RAW lower bound: the lead must cover the fill's own retirement lag. */
    int lead_lo() const {
        return lam_in;
    }
    /** empty / WAR upper bound: the lead must leave the drained slot's reader time to retire. */
    int lead_hi() const {
        return Q - 1 - lam_out;
    }
    /** Is there any legal lead? Equivalent to Q >= lam_in + lam_out + 1 (§8.3). */
    bool feasible() const {
        return lead_lo() <= lead_hi();
    }
    /** The lead to use: maximal legal prefetch (the upper endpoint). */
    int lead() const {
        return lead_hi();
    }

    // --- derived: the RETIMING (§12.4) -- which side carries it follows the PLACEMENT ---

    /** Iterations by which a warp-specialized consumer's empty-edge *arrive* must be retimed:
     * at iteration k it releases slot (k - retiming) % Q, whose read has provably retired under
     * `wgmma.wait_group lam_out`. The uniform path needs no such offset -- there lead() already
     * carries the same lam_out. Zero => byte-identical to the drain (lam_out == 0) lowering. */
    int release_retiming() const {
        return lam_out;
    }
};

/** The ring depth Q of producer `name`, or 0 if it is not a positive compile-time constant
 * (we only reason about statically-sized rings). */
inline int ring_pipeline_depth(const std::map<std::string, Function> &env,
                               const std::string &name) {
    auto it = env.find(name);
    if (it == env.end()) {
        return 0;
    }
    Expr q = it->second.schedule().ring_buffer();
    if (!q.defined()) {
        return 0;
    }
    auto qi = as_const_int(simplify(q));
    return (qi && *qi > 0) ? (int)*qi : 0;
}

/** Resolve the pipeline facts of producer `name`'s ring.
 *
 * The two lags are read from the diagnostic probes HL_CPASYNC_INFLIGHT / HL_WGMMA_INFLIGHT for
 * now. Per the project rule (env flags are probes, never the shipped mechanism) these become
 * `async(level, lag)` schedule fields -- lam_in beside FuncSchedule::async/async_place, lam_out a
 * StageSchedule field on the consumer stage whose own async production is the wgmma. Callers see
 * only the resolved record, so that swap does not touch any lowering. */
inline RingPipeline resolve_ring_pipeline(const std::map<std::string, Function> &env,
                                         const std::string &name) {
    RingPipeline rp;
    rp.Q = ring_pipeline_depth(env, name);
    auto probe = [](const char *var) {
        std::string s = get_env_variable(var);
        if (s.empty()) {
            return 0;
        }
        int v = atoi(s.c_str());
        return v > 0 ? v : 0;
    };
    rp.lam_in = probe("HL_CPASYNC_INFLIGHT");
    rp.lam_out = probe("HL_WGMMA_INFLIGHT");
    return rp;
}

}  // namespace Internal
}  // namespace Halide

#endif
