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

    // 10. Serial loop inside the block: each block (a y-row) loops serially over
    //     x-chunks, staging each chunk through the shared producer. This is the
    //     structure Phase 2 will software-pipeline; here it must be correct at
    //     depth 1 (warp-specialized per serial iteration).
    check2d(
        "serial_loop_in_block",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x * 2 + y;
            c(x, y) = p(x, y) * 3 + 1;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), xi("xi");
            c.compute_root()
                .split(x, xo, xi, 32)
                .reorder(xi, xo, y)
                .gpu_blocks(y)
                .gpu_threads(xi);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x).async();
        },
        128, 8);

    // 11. Same, with a non-chunk-aligned width (serial-loop tails).
    check2d(
        "serial_loop_tails",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x - y;
            c(x, y) = p(x, y) + 5;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), xi("xi");
            c.compute_root()
                .split(x, xo, xi, 32)
                .reorder(xi, xo, y)
                .gpu_blocks(y)
                .gpu_threads(xi);
            p.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x).async();
        },
        100, 8);

    // 12. Ring-buffered serial loop: storage hoisted to the block level with 2
    //     shared slots indexed by (serial iteration % 2). This is the storage
    //     Phase 2 overlap needs; here it must still be correct at depth 1.
    check2d(
        "ring_buffer_serial",
        [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x * 2 + y;
            c(x, y) = p(x, y) * 3 + 1;
        },
        [](Func &p, Func &c, Var x, Var y) {
            Var xo("xo"), xi("xi");
            c.compute_root()
                .split(x, xo, xi, 32)
                .reorder(xi, xo, y)
                .gpu_blocks(y)
                .gpu_threads(xi);
            p.compute_at(c, xo)
                .store_in(MemoryType::GPUShared)
                .gpu_threads(x)
                .hoist_storage(c, y)
                .ring_buffer(2)
                .async();
        },
        128, 8);

    // 13. Two parallel async shared-memory producers feeding one consumer.
    {
        Var x("x"), y("y"), xo("xo"), yo("yo"), xi("xi"), yi("yi");
        auto algo = [](Func &p1, Func &p2, Func &c, Var x, Var y) {
            p1(x, y) = x + y;
            p2(x, y) = x - y;
            c(x, y) = p1(x, y) * 2 + p2(x, y);
        };

        Func p1r("p1r"), p2r("p2r"), cr("cr");
        algo(p1r, p2r, cr, x, y);
        Buffer<int> ref = cr.realize({W, H});

        Func p1("p1"), p2("p2"), c("consumer");
        algo(p1, p2, c, x, y);
        c.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        p1.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        p2.compute_at(c, xo).store_in(MemoryType::GPUShared).gpu_threads(x, y).async();
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "parallel_producers")) {
            num_failures++;
        } else {
            printf("[parallel_producers] ok\n");
        }
    }

    // 14. Chain of two async shared-memory producers feeding the consumer.
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

    // 15. Two *ring-buffered* async shared-memory producers feeding one consumer.
    //     Unlike #13, both producers use ring_buffer(2), so this is the path that
    //     exercises the N-ary warp-async fork (2 producer warp groups + 1 consumer,
    //     each producer with its own block of per-slot named barriers) — the
    //     GEMM-shaped double-buffered staging.
    {
        Var x("x"), y("y"), xo("xo"), xi("xi");
        auto algo = [](Func &a, Func &b, Func &c, Var x, Var y) {
            a(x, y) = x + y;
            b(x, y) = x - y;
            c(x, y) = a(x, y) * 2 + b(x, y);
        };

        Func a_ref("a_ref"), b_ref("b_ref"), c_ref("c_ref");
        algo(a_ref, b_ref, c_ref, x, y);
        Buffer<int> ref = c_ref.realize({W, H});

        Func a("As"), b("Bs"), c("consumer");
        algo(a, b, c, x, y);
        c.compute_root()
            .split(x, xo, xi, 32)
            .reorder(xi, xo, y)
            .gpu_blocks(y)
            .gpu_threads(xi);
        for (Func *p : {&a, &b}) {
            p->compute_at(c, xo)
                .store_in(MemoryType::GPUShared)
                .gpu_threads(x)
                .hoist_storage(c, y)
                .ring_buffer(2)
                .async();
        }
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "ring_buffer_two_producers")) {
            num_failures++;
        } else {
            printf("[ring_buffer_two_producers] ok\n");
        }
    }

    // 16. Per-role binding via compute_with: two staged producers folded onto ONE
    //     producer warp group. Only the anchor (b) is async; a is compute_with'd
    //     into it (still ring-buffered for double buffering, but not separately
    //     async), so both are computed by the same warp group and a single
    //     semaphore guards the whole cluster -> 2 warp groups (producer + consumer)
    //     instead of the 3 of #15. This is the "same group, independent" cell of
    //     the allocation algebra (plan §9).
    {
        Var x("x"), y("y"), xo("xo"), xi("xi");
        auto algo = [](Func &a, Func &b, Func &c, Var x, Var y) {
            a(x, y) = x + y;
            b(x, y) = x - y;
            c(x, y) = a(x, y) * 2 + b(x, y);
        };

        Func a_ref("a_ref"), b_ref("b_ref"), c_ref("c_ref");
        algo(a_ref, b_ref, c_ref, x, y);
        Buffer<int> ref = c_ref.realize({W, H});

        Func a("As"), b("Bs"), c("consumer");
        algo(a, b, c, x, y);
        c.compute_root()
            .split(x, xo, xi, 32)
            .reorder(xi, xo, y)
            .gpu_blocks(y)
            .gpu_threads(xi);
        for (Func *p : {&a, &b}) {
            p->compute_at(c, xo)
                .store_in(MemoryType::GPUShared)
                .gpu_threads(x)
                .hoist_storage(c, y)
                .ring_buffer(2);
        }
        a.compute_with(b, x);  // fold a's producer into b's warp group
        b.async();             // only the anchor is async
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "compute_with_role")) {
            num_failures++;
        } else {
            printf("[compute_with_role] ok\n");
        }
    }

    // 17. Asymmetric warp groups: the producer stages a 64-wide tile with only 32
    //     threads (an inner serial loop covers the rest), while the consumer reads it
    //     with 64. In the rectangular warp-group model blockDim.x is the max (64) and
    //     the smaller group's padded lanes still execute the barriers (which sit
    //     outside the per-tile thread guard), so the 2*max participant count stays
    //     correct — asymmetric WORK already lowers. The cost is launching the padded
    //     lanes; the flat-partition optimization (plan §9.1) would remove it.
    {
        Var x("x"), y("y"), xo("xo"), xi("xi"), pxo("pxo"), pxi("pxi");
        auto algo = [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x * 3 + y;
            c(x, y) = p(x, y) * 2;
        };

        Func p_ref("p_ref"), c_ref("c_ref");
        algo(p_ref, c_ref, x, y);
        Buffer<int> ref = c_ref.realize({W, H});

        Func p("p"), c("consumer");
        algo(p, c, x, y);
        c.compute_root()
            .split(x, xo, xi, 64)
            .reorder(xi, xo, y)
            .gpu_blocks(y)
            .gpu_threads(xi);
        p.compute_at(c, xo)
            .store_in(MemoryType::GPUShared)
            .split(x, pxo, pxi, 32)
            .gpu_threads(pxi)
            .hoist_storage(c, y)
            .ring_buffer(2)
            .async();
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "asymmetric_groups")) {
            num_failures++;
        } else {
            printf("[asymmetric_groups] ok\n");
        }
    }

    // 18. 2D thread groups through the ring path. Producer and consumer both use
    //     gpu_threads(x, y) (an 8x8 = 64-thread tile), staged via ring_buffer(2).
    //     Under the Fork-aware fuser this exercises the multi-dim decompose in
    //     FlattenBranchThreads (flat id -> x = local%8, y = local/8) and a >1-warp
    //     group (64 lanes); blockDim should be 128 = 64 producer + 64 consumer.
    {
        Var x("x"), y("y"), xo("xo"), xi("xi"), yo("yo"), yi("yi");
        auto algo = [](Func &p, Func &c, Var x, Var y) {
            p(x, y) = x + 2 * y;
            c(x, y) = p(x, y) * 3;
        };

        Func p_ref("p_ref"), c_ref("c_ref");
        algo(p_ref, c_ref, x, y);
        Buffer<int> ref = c_ref.realize({W, H});

        Func p("p"), c("consumer");
        algo(p, c, x, y);
        c.compute_root()
            .split(x, xo, xi, 8)
            .split(y, yo, yi, 8)
            .reorder(xi, yi, xo, yo)
            .gpu_blocks(yo)
            .gpu_threads(xi, yi);
        p.compute_at(c, xo)
            .store_in(MemoryType::GPUShared)
            .gpu_threads(x, y)
            .hoist_storage(c, yo)
            .ring_buffer(2)
            .async();
        Buffer<int> got = c.realize({W, H}, target);
        got.copy_to_host();
        if (!compare(got, ref, "ring_buffer_2d")) {
            num_failures++;
        } else {
            printf("[ring_buffer_2d] ok\n");
        }
    }

    if (num_failures > 0) {
        printf("FAILED: %d scenario(s)\n", num_failures);
        return 1;
    }
    printf("Success!\n");
    return 0;
}
