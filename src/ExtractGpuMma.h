#ifndef HALIDE_EXTRACT_GPU_MMA_H
#define HALIDE_EXTRACT_GPU_MMA_H

/** \file
 * Recognise a warp-level fp16 tile multiply-accumulate scheduled into a 32-lane
 * GPU warp and rewrite it to the tensor-core primitives (gpu_mma_f16_f32, and
 * later gpu_ldmatrix) lowered in CodeGen_PTX_Dev. This is the GPU analogue of
 * ExtractTileOperations (AMX): instruction selection from the schedule's tile
 * shape, not a directive. Target-gated to CUDA capability 80+ — on older GPUs the
 * same schedule is left as the ordinary fma reduction (P-G).
 */

#include "Expr.h"

namespace Halide {

struct Target;

namespace Internal {

Stmt extract_gpu_mma(const Stmt &s, const Target &t);

}  // namespace Internal
}  // namespace Halide

#endif
