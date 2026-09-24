#include "Halide.h"
#include "halide_benchmark.h"

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

// A 1-D heat equation iterated over time, written as an inductive Func:
// f(x, t) reads f(x - 1, t - 1), f(x, t - 1) and f(x + 1, t - 1), and the
// output is the last slice. Its storage folds to the two slices the step
// reads, with nothing in the schedule asking for it.
//
// Computing f per timestep, in parallel over x, streams the two slices
// through memory at every step. Computing f per tile of the output tiles the
// computation in time instead: each tile computes the cone of the earlier
// slices it depends on, and its two-slice window stays in cache for every
// step. This compares the two.

using namespace Halide;
using namespace Halide::Tools;

namespace {

const float C = 0.1f;

Func build(bool temporal, int W, int T, int tile, const ImageParam &in) {
    Func f(Float(32), "f");
    Var x("x"), t("t"), xo("xo"), xi("xi");
    f(x, t) = select(t <= 0 || x <= 0 || x >= W - 1, in(clamp(x, 0, W - 1)),
                     likely(f(x, t - 1) + C * (f(x - 1, t - 1) - 2 * f(x, t - 1) + f(x + 1, t - 1))));
    Func out("out");
    out(x) = f(x, T - 1);
    out.bound(x, 0, W);
    out.split(x, xo, xi, tile).parallel(xo).vectorize(xi, 8);
    if (temporal) {
        f.compute_at(out, xo).vectorize(x, 8);
    } else {
        f.compute_root().split(x, xo, xi, tile).parallel(xo).vectorize(xi, 8);
    }
    return out;
}

}  // namespace

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (target.arch == Target::WebAssembly) {
        printf("[SKIP] Performance is meaningless under the WebAssembly "
               "interpreter.\n");
        return 0;
    }

    const int W = 1 << 21, T = 64;
    Buffer<float> input(W);
    for (int x = 0; x < W; x++) {
        input(x) = std::sin(0.001f * x) + (x % 97) * 0.01f;
    }
    ImageParam in(Float(32), 1);
    in.set(input);

    std::vector<float> a(input.data(), input.data() + W), b(W);
    for (int t = 1; t < T; t++) {
        b[0] = a[0];
        b[W - 1] = a[W - 1];
        for (int x = 1; x < W - 1; x++) {
            b[x] = a[x] + C * (a[x - 1] - 2 * a[x] + a[x + 1]);
        }
        std::swap(a, b);
    }

    Func per_step = build(false, W, T, 65536, in);
    Func temporal = build(true, W, T, 8192, in);
    Buffer<float> out_step(W), out_temporal(W);
    per_step.realize(out_step);
    temporal.realize(out_temporal);
    for (int x = 0; x < W; x++) {
        if (std::abs(out_step(x) - a[x]) > 1e-4f || std::abs(out_temporal(x) - a[x]) > 1e-4f) {
            printf("Mismatch at %d: per-step %f, temporal %f, reference %f\n",
                   x, out_step(x), out_temporal(x), a[x]);
            return 1;
        }
    }

    const double t_step = benchmark(5, 5, [&]() { per_step.realize(out_step); });
    const double t_temporal = benchmark(5, 5, [&]() { temporal.realize(out_temporal); });
    const double speedup = t_step / t_temporal;
    printf("1-D heat, %d points, %d steps\n", W, T);
    printf("per timestep, parallel over x = %.2f ms\n", t_step * 1e3);
    printf("tiled in time, parallel tiles = %.2f ms\n", t_temporal * 1e3);
    printf("speedup = %.2fx\n", speedup);

    const unsigned hw = std::thread::hardware_concurrency();
    if (hw >= 4) {
        if (speedup <= 1.0) {
            printf("Tiling in time was no faster than per timestep (%.2fx) on %u "
                   "cores\n",
                   speedup, hw);
            return 1;
        }
    } else {
        printf("[SKIP] speedup assertion (only %u hardware threads)\n", hw);
    }

    printf("Success!\n");
    return 0;
}
