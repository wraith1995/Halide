#include "Halide.h"
#include <functional>
#include <stdio.h>

using namespace Halide;

// Correctness tests for GPU warp-specialized async producers.
//
// A producer scheduled with async() + store_in(MemoryType::GPUShared), computed
// between the GPU block and thread loops, is consumed within the same block.
// This is the GPU analogue of CPU .async(): instead of producer/consumer host
// threads, a group of warps in each block produces (e.g. stages data into
// shared memory) while the rest consume. Until device-side warp specialization
// is fully implemented these schedules degrade to synchronous shared-memory
// staging; either way results must match a reference.
//
// The goal is coverage as dense as the CPU async tests and lesson_24_async:
// many equivalent scenarios, several producer/consumer split strategies,
// different producer warp counts, non-tile-aligned ("non-perfect") loops,
// stencils with halos, 1D/2D, multiple consumers, and producer chains.
//
// Each scenario defines a pure algorithm twice: once with a default (CPU)
// schedule to produce the reference, once with a GPU warp-spec schedule. The
// algorithm builder takes fresh Funcs so the reference is guaranteed to match.

namespace {

Target target;
int num_failures = 0;

template<typename T>
bool compare(const Buffer<T> &got, const Buffer<T> &ref, const char *name) {
    if (got.dimensions() != ref.dimensions()) {
        printf("[%s] dimension mismatch\n", name);
        return false;
    }
    for (int i = 0; i < got.dimensions(); i++) {
        if (got.dim(i).extent() != ref.dim(i).extent()) {
            printf("[%s] extent mismatch in dim %d\n", name, i);
            return false;
        }
    }
    bool ok = true;
    got.for_each_element([&](const int *pos) {
        if (!ok) {
            return;
        }
        T a = got(pos);
        T b = ref(pos);
        if (a != b) {
            if (got.dimensions() == 2) {
                printf("[%s] (%d, %d): %f vs %f\n", name, pos[0], pos[1],
                       (double)a, (double)b);
            } else {
                printf("[%s] (%d): %f vs %f\n", name, pos[0], (double)a, (double)b);
            }
            ok = false;
        }
    });
    return ok;
}

// Run a 2D scenario: `algo` builds the (producer, consumer) algorithm into the
// passed Funcs; `sched` applies a GPU warp-spec schedule. Compares GPU output
// to a CPU reference of the same algorithm.
void check2d(const char *name,
             const std::function<void(Func &p, Func &c, Var x, Var y)> &algo,
             const std::function<void(Func &p, Func &c, Var x, Var y)> &sched,
             int w, int h) {
    Var x("x"), y("y");

    Func p_ref("p_ref"), c_ref("c_ref");
    algo(p_ref, c_ref, x, y);
    Buffer<int> ref = c_ref.realize({w, h});  // default CPU schedule = oracle

    Func p("producer"), c("consumer");
    algo(p, c, x, y);
    sched(p, c, x, y);
    Buffer<int> got = c.realize({w, h}, target);
    got.copy_to_host();

    if (!compare(got, ref, name)) {
        num_failures++;
    } else {
        printf("[%s] ok\n", name);
    }
}

void check1d(const char *name,
             const std::function<void(Func &p, Func &c, Var x)> &algo,
             const std::function<void(Func &p, Func &c, Var x)> &sched,
             int w) {
    Var x("x");

    Func p_ref("p_ref"), c_ref("c_ref");
    algo(p_ref, c_ref, x);
    Buffer<int> ref = c_ref.realize({w});

    Func p("producer"), c("consumer");
    algo(p, c, x);
    sched(p, c, x);
    Buffer<int> got = c.realize({w}, target);
    got.copy_to_host();

    if (!compare(got, ref, name)) {
        num_failures++;
    } else {
        printf("[%s] ok\n", name);
    }
}

}  // namespace

int main(int argc, char **argv) {
    target = get_jit_target_from_environment();
    if (!target.has_feature(Target::CUDA)) {
        printf("[SKIP] GPU warp specialization currently requires CUDA.\n");
        return 0;
    }

    // Sizes deliberately not multiples of the tile, to exercise tails.
    const int W = 100, H = 80;

    // 1. Elementwise, tile-aligned extent.
    check2d(
        "elementwise_aligned",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + 2 * y;
            c(x, y) = p(x, y) + 1;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        },
        96, 80);

    // 2. Elementwise, non-tile-aligned extent (tails via GuardWithIf default).
    check2d(
        "elementwise_tails",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x * 3 - y;
            c(x, y) = p(x, y) + 7;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        },
        W, H);

    // 3. Non-square tile (different split aspect ratio), with tails.
    check2d(
        "nonsquare_tile",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + y * y;
            c(x, y) = p(x, y) * 2;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 32, 8);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        },
        W, H);

    // 4. 3x3-ish stencil: consumer reads a halo of the producer; the producer's
    //    per-block region is larger than the tile.
    check2d(
        "stencil_halo",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + 2 * y;
            c(x, y) = p(x - 1, y) + p(x, y) + p(x + 1, y) + p(x, y - 1) + p(x, y + 1);
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        },
        W, H);

    // 5. Multiple producer warps via gpu_producer_warps(2).
    check2d(
        "producer_warps_2",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x - 2 * y;
            c(x, y) = p(x, y) + 100;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).gpu_producer_warps(2).async();
        },
        W, H);

    // 6. Multiple producer warps via gpu_producer_warps(4).
    check2d(
        "producer_warps_4",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = 5 * x + y;
            c(x, y) = p(x, y) - 3;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).gpu_producer_warps(4).async();
        },
        W, H);

    // 7. Producer with a 1D thread decomposition only (gpu_threads(x)).
    check2d(
        "producer_threads_1d",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + 11 * y;
            c(x, y) = p(x, y) + 1;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x).async();
        },
        W, H);

    // 8. 1D pipeline.
    check1d(
        "elementwise_1d",
        [](Func &p, Func &c, Var x) {
            p(x) = 2 * x + 1;
            c(x) = p(x) * p(x);
        },
        [](Func &p, Func &c, Var x) {
            Var xo("xo"), xi("xi");
            c.compute_root().gpu_tile(x, xo, xi, 32);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x).async();
        },
        1000);

    // 9. Two consumers of a single async producer.
    check2d(
        "two_consumers",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + 2 * y;
            Func c1("c1");
            c1(x, y) = p(x, y) + 1;
            c(x, y) = c1(x, y) + p(x, y);
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), yo("yo"), xi("xi"), yi("yi");
            c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        },
        W, H);

    // 10. Chain of two async shared-memory producers feeding the consumer.
    //     Written inline so both producers in the chain can be scheduled.
    {
        Var x("x"), y("y"), xo("xo"), yo("yo"), xi("xi"), yi("yi");
        auto algo = [](Func &p, Func &q, Func &c, Var x, Var y) {
            p(x, y) = x + 2 * y;
            q(x, y) = p(x, y) * 2;
            c(x, y) = q(x, y) + 1;
        };

        Func p_ref("p_ref"), q_ref("q_ref"), c_ref("c_ref");
        algo(p_ref, q_ref, c_ref, x, y);
        Buffer<int> ref = c_ref.realize({W, H});

        Func p("p"), q("q"), c("consumer");
        algo(p, q, c, x, y);
        c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        q.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "async_chain")) {
            num_failures++;
        } else {
            printf("[async_chain] ok\n");
        }
    }

    if (num_failures > 0) {
        printf("FAILED: %d scenario(s)\n", num_failures);
        return 1;
    }
    printf("Success!\n");
    return 0;
}
