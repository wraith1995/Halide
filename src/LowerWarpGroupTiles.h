#ifndef HALIDE_LOWER_WARP_GROUP_TILES_H
#define HALIDE_LOWER_WARP_GROUP_TILES_H

/** \file
 * Defines the recognizer that lowers a WarpGroup-realized vectorized tile
 * reduce (a gpu_warps tile GEMM: acc[m,n] += A_sh[m,k]*B_sh[k,n]) into a
 * Hopper wgmma collective. Runs pre-fusion, where the gpu_warps thread axis
 * and the VectorReduce's WarpGroup realization tag are both intact.
 *
 * The reduce's value is replaced by a wgmma fragment intrinsic (codegen emits
 * the wgmma.mma_async collective once and extracts the per-thread fragment
 * register), and the output store's index is rewritten to the hardware's
 * non-affine lane->element fragment map -- so each thread writes its own
 * fragment registers to the slots wgmma assigns, with no cross-lane register
 * read. See research/gpu_recognizer_design.md S5b.
 */

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

/** Recognize WarpGroup-realized tile reduces and lower them to wgmma. On
 * targets without wgmma (anything below sm_90) this is a no-op: the reduce
 * keeps its WarpGroup tag and falls back to the generic thread-level reduce in
 * codegen, so the same schedule stays correct everywhere. */
Stmt lower_warp_group_tiles(Stmt s, const Target &t);

}  // namespace Internal
}  // namespace Halide

#endif
