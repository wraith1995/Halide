#ifndef HALIDE_SOFTWARE_PIPELINE_H
#define HALIDE_SOFTWARE_PIPELINE_H

/** \file
 * Defines the lowering pass for Func::software_pipeline(): rotate a producer's
 * lead within its consumer's serial loop (peel a prologue, run a skewed steady
 * state, drain an epilogue) so the producer's latency overlaps the consumer's
 * compute on the SAME execution unit. This is an intra-execution-unit temporal
 * skew, realized as a late reschedule over already-frozen placement -- it does
 * NOT touch the iteration domain, bounds inference, or the thread/storage
 * mapping (those ran earlier). See Func::software_pipeline and
 * research/software_pipeline_roadmap.md.
 */

#include <map>
#include <string>

#include "Expr.h"

namespace Halide {
namespace Internal {

class Function;

/** Rewrite every serial loop that directly contains a producer scheduled with
 * software_pipeline() into prologue + skewed steady-state + epilogue. Runs after
 * storage flattening, so the ring-buffer slot index is already an explicit
 * function of the loop variable and substituting the variable in a hoisted unit
 * makes the slot follow automatically. Loops without such a producer are
 * untouched (NFC). */
Stmt software_pipeline(Stmt s, const std::map<std::string, Function> &env);

}  // namespace Internal
}  // namespace Halide

#endif
