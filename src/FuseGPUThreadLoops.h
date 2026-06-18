#ifndef HALIDE_FUSE_GPU_THREAD_LOOPS_H
#define HALIDE_FUSE_GPU_THREAD_LOOPS_H

/** \file
 * Defines the lowering pass that fuses and normalizes loops over gpu
 * threads to target CUDA, OpenCL, and Metal.
 */

#include <map>
#include <string>

#include "Expr.h"

namespace Halide {
namespace Internal {

class Function;

class Function;

/** Rewrite all GPU loops to have a min of zero. */
Stmt zero_gpu_loop_mins(const Stmt &s);

/** Prepare GPU warp-specialized async producers (async() +
 * store_in(GPUShared)) for thread-loop fusion. For each such producer, the
 * producer and consumer work is placed on disjoint warp groups by injecting an
 * extra outer GPU-thread dimension: the producer runs on group 0, the consumer
 * on group 1. The subsequent fuse_gpu_thread_loops pass then turns this into a
 * single launch where some warps produce (e.g. stage into shared memory) while
 * others consume. Runs after canonicalize_gpu_vars and select_gpu_api, before
 * fuse_gpu_thread_loops. Producers whose structure isn't yet supported are left
 * unchanged (they fall back to synchronous shared-memory staging). */
Stmt inject_gpu_warp_specialization(Stmt s, const std::map<std::string, Function> &env);

/** Lower device async Forks from warp-specialized ring producers (async() +
 * store_in(GPUShared) + ring_buffer()) into a warp-group split coordinated by
 * per-slot named barriers: producer warps stage into a shared ring while consumer
 * warps compute, running ahead by the ring depth. Runs after canonicalize_gpu_vars,
 * before inject_gpu_warp_specialization (which handles the non-ring case). */
Stmt lower_gpu_warp_async(Stmt s, const std::map<std::string, Function> &env, const Target &t);

/** Converts Halide's GPGPU IR to the OpenCL/CUDA/Metal model. Within
 * every loop over gpu block indices, fuse the inner loops over thread
 * indices into a single loop (with predication to turn off
 * threads). Push if conditions between GPU blocks to the innermost GPU threads.
 * Also injects synchronization points as needed, and hoists
 * shared allocations at the block level out into a single shared
 * memory array, and heap allocations into a slice of a global pool
 * allocated outside the kernel. */
Stmt fuse_gpu_thread_loops(Stmt s, const Target &t,
                           const std::map<std::string, Function> &env);

}  // namespace Internal
}  // namespace Halide

#endif
