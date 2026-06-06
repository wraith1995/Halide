#include "Halide.h"
#include <cstdio>

using namespace Halide;

// Shared-memory swizzle (Func::swizzle_storage) stress/correctness test.
//
// A swizzle is a bijection applied to *every* access of a shared allocation, at
// the codegen address seam. So a pipeline that stages a Func in GPUShared with a
// swizzle must produce *bit-identical* results to the same pipeline without the
// swizzle -- the swizzle only changes which physical bank each element lands in.
// That makes the un-swizzled pipeline a perfect oracle, independent of any CPU
// reference. (Bank-conflict reduction itself is measured separately with ncu.)

namespace {

// A transpose through shared memory: the canonical bank-conflict generator, and
// the case swizzles exist to fix. Mirrors test/correctness/gpu_transpose.cpp,
// optionally swizzling the staged tile.
template<typename T>
Buffer<T> run_transpose(const Buffer<T> &input, bool swizzled, bool use_raw,
                        Swizzle mode, SwizzleLayout raw) {
    Var x, y;
    Func in_func("in_func"), out("out");
    in_func(x, y) = input(x, y);
    out(x, y) = in_func(y, x);

    Var xi, yi, xo, yo, xii, xio, yii, yio, tile_idx, subtile_idx;
    out.tile(x, y, xo, yo, xi, yi, 64, 64)
        .fuse(xo, yo, tile_idx)
        .tile(xi, yi, xio, yio, xii, yii, 16, 16)
        .fuse(xio, yio, subtile_idx)
        .gpu_blocks(subtile_idx, tile_idx)
        .gpu_threads(xii, yii);

    in_func.compute_at(out, subtile_idx)
        .store_in(MemoryType::GPUShared)
        .gpu_threads(x, y);
    if (swizzled) {
        if (use_raw) {
            in_func.swizzle_storage(raw);
        } else {
            in_func.swizzle_storage(mode);
        }
    }

    Buffer<T> result(input.height(), input.width());
    out.realize(result);
    result.copy_to_host();
    return result;
}

// A multi-neighbour stencil staged in shared: exercises many distinct shared
// addresses per thread (each lands in a swizzled bank).
template<typename T>
Buffer<T> run_blur(const Buffer<T> &input, bool swizzled, Swizzle mode) {
    Var x, y;
    Func clamped("clamped"), prod("prod"), cons("cons");
    clamped(x, y) = input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1));
    prod(x, y) = clamped(x, y);
    cons(x, y) = cast<T>(prod(x - 1, y) + prod(x + 1, y) + prod(x, y - 1) +
                         prod(x, y + 1) + prod(x, y));

    Var xo, yo, xi, yi;
    cons.gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
    prod.compute_at(cons, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y);
    if (swizzled) {
        prod.swizzle_storage(mode);
    }

    Buffer<T> result(input.width(), input.height());
    cons.realize(result);
    result.copy_to_host();
    return result;
}

// Vectorized staging: the producer's store into shared and the consumer's load
// from shared are both 4-wide 32-bit, which triggers the dense v4 shared path in
// CodeGen_PTX_Dev (Store/Load reinterpreted as a single 128-bit access at index/4).
// The swizzle must rescale its element-unit params to those coarser units and keep
// the access contiguous (granule == 16 B == the v4 width for 4-byte elements).
template<typename T>
Buffer<T> run_vectorized(const Buffer<T> &input, bool swizzled, Swizzle mode) {
    Var x, y, xo, yo, xi, yi;
    Func clamped("clamped"), prod("prod"), cons("cons");
    clamped(x, y) = input(clamp(x, 0, input.width() - 1), clamp(y, 0, input.height() - 1));
    prod(x, y) = clamped(x, y) + cast<T>(1);
    // Read prod contiguously in x (so the consumer load vectorizes 4-wide).
    cons(x, y) = prod(x, y) + prod(x, y + 1);

    cons.gpu_tile(x, y, xo, yo, xi, yi, 64, 8, TailStrategy::GuardWithIf)
        .vectorize(xi, 4, TailStrategy::GuardWithIf);
    prod.compute_at(cons, xo)
        .store_in(MemoryType::GPUShared)
        .gpu_threads(y)
        .vectorize(x, 4);
    if (swizzled) {
        prod.swizzle_storage(mode);
    }

    Buffer<T> result(input.width(), input.height());
    cons.realize(result);
    result.copy_to_host();
    return result;
}

template<typename T>
bool compare(const Buffer<T> &a, const Buffer<T> &b, const char *label) {
    if (a.width() != b.width() || a.height() != b.height()) {
        printf("FAIL [%s]: size mismatch\n", label);
        return false;
    }
    for (int y = 0; y < a.height(); y++) {
        for (int x = 0; x < a.width(); x++) {
            if (a(x, y) != b(x, y)) {
                printf("FAIL [%s] at (%d,%d): swizzled=%g ref=%g\n",
                       label, x, y, (double)a(x, y), (double)b(x, y));
                return false;
            }
        }
    }
    printf("  ok: %s\n", label);
    return true;
}

template<typename T>
Buffer<T> make_input(int w, int h) {
    Buffer<T> in(w, h);
    for (int y = 0; y < h; y++) {
        for (int x = 0; x < w; x++) {
            in(x, y) = (T)((x * 17 + y * 3 + 1) % 101);
        }
    }
    return in;
}

template<typename T>
bool transpose_modes(const char *tname) {
    Buffer<T> in = make_input<T>(128, 128);
    Buffer<T> ref = run_transpose<T>(in, /*swizzled*/ false, false, Swizzle::None, {});
    bool ok = true;
    for (Swizzle m : {Swizzle::XOR_32B, Swizzle::XOR_64B, Swizzle::XOR_128B}) {
        Buffer<T> got = run_transpose<T>(in, true, false, m, {});
        char label[128];
        snprintf(label, sizeof(label), "transpose %s mode=%d", tname, (int)m);
        ok &= compare<T>(got, ref, label);
    }
    return ok;
}

}  // namespace

int main(int argc, char **argv) {
    if (!get_jit_target_from_environment().has_gpu_feature()) {
        printf("[SKIP] No GPU target enabled.\n");
        return 0;
    }

    bool ok = true;

    // Named modes across element widths (1/2/4 bytes + float).
    ok &= transpose_modes<uint8_t>("u8");
    ok &= transpose_modes<uint16_t>("u16");
    ok &= transpose_modes<uint32_t>("u32");
    ok &= transpose_modes<float>("f32");

    // A multi-neighbour stencil staged in shared.
    {
        Buffer<uint32_t> in = make_input<uint32_t>(160, 96);
        Buffer<uint32_t> ref = run_blur<uint32_t>(in, false, Swizzle::None);
        Buffer<uint32_t> got = run_blur<uint32_t>(in, true, Swizzle::XOR_128B);
        ok &= compare<uint32_t>(got, ref, "blur u32 XOR_128B");
    }

    // Vectorized staging: exercises the dense v4 (128-bit) shared store/load path
    // under swizzle, for both integer and float 32-bit elements.
    {
        Buffer<uint32_t> in = make_input<uint32_t>(256, 64);
        Buffer<uint32_t> ref = run_vectorized<uint32_t>(in, false, Swizzle::None);
        for (Swizzle m : {Swizzle::XOR_32B, Swizzle::XOR_64B, Swizzle::XOR_128B}) {
            Buffer<uint32_t> got = run_vectorized<uint32_t>(in, true, m);
            char label[96];
            snprintf(label, sizeof(label), "vectorized u32 mode=%d", (int)m);
            ok &= compare<uint32_t>(got, ref, label);
        }
    }
    {
        Buffer<float> in = make_input<float>(256, 64);
        Buffer<float> ref = run_vectorized<float>(in, false, Swizzle::None);
        Buffer<float> got = run_vectorized<float>(in, true, Swizzle::XOR_128B);
        ok &= compare<float>(got, ref, "vectorized f32 XOR_128B");
    }

    // Raw SwizzleLayout (control field disjoint from target field => bijection).
    {
        Buffer<uint32_t> in = make_input<uint32_t>(128, 128);
        Buffer<uint32_t> ref = run_transpose<uint32_t>(in, false, false, Swizzle::None, {});
        // bits=2, base=2, shift=4 (shift >= base+bits).
        Buffer<uint32_t> got = run_transpose<uint32_t>(in, true, true, Swizzle::None,
                                                       SwizzleLayout{2, 2, 4});
        ok &= compare<uint32_t>(got, ref, "transpose u32 raw{2,2,4}");
    }

    if (!ok) {
        printf("gpu_shared_swizzle: FAILED\n");
        return 1;
    }
    printf("Success!\n");
    return 0;
}
