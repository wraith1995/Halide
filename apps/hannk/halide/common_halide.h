// A collection of utility functions shared by the halide generators.

#ifndef COMMON_HALIDE_H_
#define COMMON_HALIDE_H_

#include "Halide.h"

namespace hannk {

using Halide::Internal::saturating_add;
using Halide::Internal::saturating_sub;
using Halide::Internal::widening_add;
using Halide::Internal::widening_sub;
using Halide::Internal::widening_mul;
using Halide::Internal::rounding_shift_right;

// A tensor has the same requirements as a buffer in Halide by default, except
// the min of the innermost dimension must also be 0.
void interpret_as_tensor(Halide::OutputImageParam p);

// Require dimension dim have the same min and extent.
void require_same_min_extent(int dim, Halide::OutputImageParam first, Halide::OutputImageParam second);
void require_same_min_extent(int first_dim, Halide::OutputImageParam first, int second_dim, Halide::OutputImageParam second);

// Require that the first two dimensions of two buffers have the same bounds.
void require_same_extent_cx(Halide::OutputImageParam first, Halide::OutputImageParam second);

// Check if the first two dimensions of a buffer can be fused cleanly.
Halide::Expr can_fuse_cx(Halide::OutputImageParam p);

// A boundary condition, without likelies that cause loop partitioning.
Halide::Func constant_exterior_tensor(
    Halide::Func t, Halide::Expr exterior,
    Halide::Expr min_c, Halide::Expr extent_c,
    Halide::Expr min_x, Halide::Expr extent_x,
    Halide::Expr min_y, Halide::Expr extent_y,
    Halide::Expr min_b, Halide::Expr extent_b);
Halide::Func constant_exterior_tensor(Halide::ImageParam p, Halide::Expr exterior);

// This function implements the same computation as the ARMv7 NEON VQRDMULH
// instruction.
Halide::Expr multiply_2x_high(const Halide::Expr &a, const Halide::Expr &b);

// Performs right shift and multiply by a multiplier. Aims to be very close to
// tflite's reference implementation. However, tflite is standardizing on left
// (exponent-like) shifts.
Halide::Expr multiply_quantized(
    const Halide::Expr &x, const Halide::Expr &quantized_multiplier, const Halide::Expr &shift);

}  // namespace hannk

#endif  // COMMON_HALIDE_H_