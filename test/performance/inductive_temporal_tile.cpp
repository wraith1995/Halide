#include "Halide.h"
#include "halide_benchmark.h"

#include <cmath>
#include <cstdio>
#include <thread>
#include <vector>

// The heat equation iterated over time, in 1, 2 and 3 dimensions, written as
// an inductive Func: f(x, ..., t) reads its neighbours at t - 1, and the
// output is the last slice. Its storage folds to the two slices the step
// reads, with nothing in the schedule asking for it.
//
// Computing f per timestep, in parallel, streams the two slices through
// memory at every step. Computing f per tile of the output tiles the
// computation in time instead: each tile computes the cone of the earlier
// slices it depends on, and its two-slice window stays in cache for every
// step. The cone grows by a point per step on every side, so the extra work
// grows with the number of steps and of dimensions, and the 3-D case takes
// fewer steps. This compares the two.

using namespace Halide;
using namespace Halide::Tools;

namespace {

const float C = 0.05f;

// The heat equation on a D-dimensional grid of n points along each dimension,
// over T slices. With tile > 0 it is computed per tile of the output;
// otherwise per timestep, in parallel over chunks of the outermost dimension.
Func build(int D, int n, int T, int tile, const ImageParam &in) {
    Func f(Float(32), "f");
    std::vector<Var> xs;
    for (int d = 0; d < D; d++) {
        xs.emplace_back("x" + std::to_string(d));
    }
    Var t("t");
    std::vector<Var> args = xs;
    args.push_back(t);
    std::vector<Expr> prev(xs.begin(), xs.end()), clamped;
    prev.push_back(t - 1);
    Expr edge = t <= 0, lap = -2 * D * f(prev);
    for (int d = 0; d < D; d++) {
        edge = edge || xs[d] <= 0 || xs[d] >= n - 1;
        for (int dir : {-1, 1}) {
            std::vector<Expr> nb = prev;
            nb[d] = xs[d] + dir;
            lap += f(nb);
        }
        clamped.push_back(clamp(xs[d], 0, n - 1));
    }
    f(args) = select(edge, in(clamped), likely(f(prev) + C * lap));

    Func out("out");
    std::vector<Expr> last(xs.begin(), xs.end());
    last.push_back(T - 1);
    out(xs) = f(last);
    for (int d = 0; d < D; d++) {
        out.bound(xs[d], 0, n);
    }
    if (tile > 0) {
        std::vector<VarOrRVar> dims, outer, inner;
        std::vector<Expr> factors;
        for (int d = 0; d < D; d++) {
            dims.emplace_back(xs[d]);
            outer.emplace_back(Var("o" + std::to_string(d)));
            inner.emplace_back(Var("i" + std::to_string(d)));
            factors.emplace_back(tile);
        }
        out.tile(dims, outer, inner, factors);
        for (int d = D - 1; d > 0; d--) {
            out.fuse(outer[d - 1], outer[d], outer[d - 1]);
        }
        out.parallel(outer[0]).vectorize(inner[0], 8);
        f.compute_at(out, outer[0].var).vectorize(xs[0], 8);
    } else {
        // Chunks of a few rows or planes, so that each task is big enough.
        const int chunk = D == 1 ? 65536 : D == 2 ? 32 :
                                                    1;
        Var o("o"), i("i");
        const Var &v = D == 1 ? i : xs[0];
        out.split(xs[D - 1], o, i, chunk).parallel(o).vectorize(v, 8);
        f.compute_root().split(xs[D - 1], o, i, chunk).parallel(o).vectorize(v, 8);
    }
    return out;
}

// Returns false on a wrong result or, on 4 or more cores, if tiling in time
// is no faster.
bool run(int D, int n, int T, int tile) {
    std::vector<int> sizes(D, n);
    Buffer<float> input(sizes);
    input.for_each_element([&](const int *p) {
        float v = 0;
        for (int d = 0; d < D; d++) {
            v += std::sin(0.01f * (d + 1) * p[d]) + (p[d] % 7) * 0.01f;
        }
        input(p) = v;
    });
    ImageParam in(Float(32), D);
    in.set(input);

    // The reference. Points on the edge keep their initial values.
    const size_t total = input.number_of_elements();
    std::vector<float> a(input.data(), input.data() + total), b = a;
    std::vector<size_t> stride(D, 1);
    for (int d = 1; d < D; d++) {
        stride[d] = stride[d - 1] * n;
    }
    for (int s = 1; s < T; s++) {
        for (size_t i = 0; i < total; i++) {
            bool interior = true;
            float lap = -2 * D * a[i];
            for (int d = 0; d < D && interior; d++) {
                const int c = (int)((i / stride[d]) % n);
                interior = c > 0 && c < n - 1;
                if (interior) {
                    lap += a[i - stride[d]] + a[i + stride[d]];
                }
            }
            b[i] = interior ? a[i] + C * lap : a[i];
        }
        std::swap(a, b);
    }

    Func per_step = build(D, n, T, 0, in);
    Func temporal = build(D, n, T, tile, in);
    Buffer<float> out_step(sizes), out_temporal(sizes);
    per_step.realize(out_step);
    temporal.realize(out_temporal);
    for (size_t i = 0; i < total; i++) {
        const float s = out_step.data()[i], tt = out_temporal.data()[i];
        if (std::abs(s - a[i]) > 1e-4f || std::abs(tt - a[i]) > 1e-4f) {
            printf("%d-D: mismatch at %zu: per-step %f, temporal %f, reference %f\n",
                   D, i, s, tt, a[i]);
            return false;
        }
    }

    const double t_step = benchmark(5, 5, [&]() { per_step.realize(out_step); });
    const double t_temporal = benchmark(5, 5, [&]() { temporal.realize(out_temporal); });
    const double speedup = t_step / t_temporal;
    printf("%d-D heat, %d^%d points, %d steps, tiles of %d^%d\n", D, n, D, T - 1, tile, D);
    printf("  per timestep, parallel  = %.2f ms\n", t_step * 1e3);
    printf("  tiled in time, parallel = %.2f ms\n", t_temporal * 1e3);
    printf("  speedup = %.2fx\n", speedup);

    const unsigned hw = std::thread::hardware_concurrency();
    if (hw < 4) {
        printf("[SKIP] speedup assertion (only %u hardware threads)\n", hw);
    } else if (speedup <= 1.0) {
        printf("Tiling in time was no faster than per timestep (%.2fx) on %u cores\n",
               speedup, hw);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (target.arch == Target::WebAssembly) {
        printf("[SKIP] Performance is meaningless under the WebAssembly "
               "interpreter.\n");
        return 0;
    }

    if (!run(1, 1 << 21, 65, 8192) ||
        !run(2, 2048, 17, 128) ||
        !run(3, 192, 5, 32)) {
        return 1;
    }

    printf("Success!\n");
    return 0;
}
