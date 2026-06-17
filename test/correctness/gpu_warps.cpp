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
    };
    const Case cases[] = {{2, 32}, {4, 32}, {3, 48}, {2, 64}};

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
            .gpu_warps(wg)        // c.groups warp groups, size derived from tx
            .gpu_threads(tx);

        Buffer<int> out = f.realize({n}, target);
        out.copy_to_host();

        for (int i = 0; i < n; i++) {
            int correct = i * 2 + 1;
            if (out(i) != correct) {
                printf("gpu_warps(groups=%d, threads_per_group=%d): out(%d) = %d instead of %d\n",
                       c.groups, c.threads_per_group, i, out(i), correct);
                return 1;
            }
        }
        printf("gpu_warps groups=%d threads_per_group=%d (per_block=%d): OK\n",
               c.groups, c.threads_per_group, per_block);
    }

    printf("Success!\n");
    return 0;
}
