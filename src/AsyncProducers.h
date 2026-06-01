#ifndef HALIDE_ASYNC_PRODUCERS_H
#define HALIDE_ASYNC_PRODUCERS_H

/** \file
 * Defines the lowering pass that injects task parallelism for producers that are scheduled as async.
 */
#include <map>
#include <string>

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

class Function;

Stmt fork_async_producers(Stmt s, const std::map<std::string, Function> &env);

/** Is this Func scheduled as an async producer stored in GPU shared memory?
 * Such Funcs lower to GPU warp specialization rather than host-thread async. */
bool is_gpu_warp_specialized(const Function &f);

/** A Func scheduled as an async producer stored in GPU shared memory lowers to
 * GPU warp specialization. Validate such Funcs against what is currently
 * supported (CUDA-only, plus not-yet-implemented combinations), emitting clear
 * user errors. Runs early in lowering, before fork_async_producers. */
void validate_gpu_async_producers(const std::map<std::string, Function> &env,
                                  const Target &t);

}  // namespace Internal
}  // namespace Halide

#endif
