#include "Halide.h"
#include <cstdio>

using namespace Halide;

// Exercise the gpu_warps work-distribution axis: a block's output tile is
// partitioned across N warp groups (a SUM partition of the block's warps,
// warp-aligned per group), each group computing its own sub-tile with a
// gpu_threads tile inside. Verified against the closed-form result. This is
// the data-symmetric source of warp groups (vs the role-asymmetric .async()
// Fork); both lower through the same flat partition (partition_warp_groups).

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (!target.has_gpu_feature()) {
        printf("[SKIP] gpu_warps test requires a GPU target.\n");
        return 0;
    }

    // (groups, threads_per_group) cases. The last has threads_per_group not a
    // multiple of the warp size (32), so each group is rounded up to 2 warps
    // with the tail lanes masked — exercises the per-group warp-alignment +
    // FlattenBranchThreads guard.
    struct Case {
        int groups;
        int threads_per_group;
        int explicit_warps;  // 0 = derive group size from the tile; N = force N warps/group
    };
    const Case cases[] = {
        {2, 32, 0},
        {4, 32, 0},
        {3, 48, 0},
        {2, 64, 0},
        {2, 32, 2},  // explicit 2-warp groups, only 32 lanes work (32 idle) — wgmma-scope sizing
        {3, 32, 4},  // explicit 4-warp groups, 32 lanes work (96 idle)
    };

    for (const Case &c : cases) {
        const int per_block = c.groups * c.threads_per_group;
        const int blocks = 5;
        const int n = blocks * per_block;

        Func f("f");
        Var x("x"), xo("xo"), xb("xb"), wg("wg"), tx("tx");

        f(x) = x * 2 + 1;

        // x -> xo (block) x xb (within-block); xb -> wg (group) x tx (thread).
        f.compute_root()
            .split(x, xo, xb, per_block)
            .split(xb, wg, tx, c.threads_per_group)
            .gpu_blocks(xo)
            .gpu_warps(wg, c.explicit_warps)  // c.groups warp groups; size derived or explicit
            .gpu_threads(tx);

        Buffer<int> out = f.realize({n}, target);
        out.copy_to_host();

        for (int i = 0; i < n; i++) {
            int correct = i * 2 + 1;
            if (out(i) != correct) {
                printf("gpu_warps(groups=%d, threads_per_group=%d, explicit_warps=%d): out(%d) = %d instead of %d\n",
                       c.groups, c.threads_per_group, c.explicit_warps, i, out(i), correct);
                return 1;
            }
        }
        printf("gpu_warps groups=%d threads_per_group=%d explicit_warps=%d (per_block=%d): OK\n",
               c.groups, c.threads_per_group, c.explicit_warps, per_block);
    }

    // Shared-memory consumer: each warp group stages a tile into shared and reads a
    // neighbour (x+1), forcing an intra-group barrier between the store and the read.
    // A derived-size gpu_warps axis lowers as a thread sub-dimension (no guarded peel),
    // so the standard pipeline inserts a correct whole-CTA barrier every lane reaches.
    {
        const int G = 2, T = 32, blocks = 4;
        const int per_block = G * T;
        const int n = blocks * per_block;

        Buffer<int> in(n);
        for (int i = 0; i < n; i++) {
            in(i) = i * 3 + 5;
        }
        Func in_b = BoundaryConditions::repeat_edge(in);

        Func staged("staged"), cons("cons");
        Var x("x"), xo("xo"), xb("xb"), wg("wg"), tx("tx");
        staged(x) = in_b(x);
        cons(x) = staged(x) + staged(x + 1);

        cons.compute_root()
            .split(x, xo, xb, per_block)
            .split(xb, wg, tx, T)
            .gpu_blocks(xo)
            .gpu_warps(wg)
            .gpu_threads(tx);
        staged.compute_at(cons, wg).store_in(MemoryType::GPUShared).gpu_threads(x);

        Buffer<int> out = cons.realize({n}, target);
        out.copy_to_host();
        for (int i = 0; i < n; i++) {
            int xp = (i + 1 < n) ? (i + 1) : (n - 1);  // repeat_edge
            int correct = (i * 3 + 5) + (xp * 3 + 5);
            if (out(i) != correct) {
                printf("gpu_warps shared-staging: out(%d) = %d instead of %d\n", i, out(i), correct);
                return 1;
            }
        }
        printf("gpu_warps shared-staging (stencil, %d groups x %d threads): OK\n", G, T);
    }

    printf("Success!\n");
    return 0;
}
