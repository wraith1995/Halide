#include "Halide.h"

#include <cstdio>

// A reindexed reduction composes with GPU scheduling. An independent scatter is
// reindexed by a transpose (both axes proven parallel-safe), then the two
// PureRVars are mapped to GPU blocks/threads with gpu_tile(). Runs only when a
// GPU target is enabled; otherwise it reports [SKIP].

using namespace Halide;

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (!target.has_gpu_feature()) {
        printf("[SKIP] No GPU target enabled.\n");
        return 0;
    }

    const int W = 32, H = 32;
    Func in("in");
    Var x("x"), y("y");
    in(x, y) = x * 10 + y;

    Func f("f");
    f(x, y) = 0;
    RDom r(0, W, 0, H, "r");
    f(r.x, r.y) = in(r.x, r.y);

    Var u("u"), v("v");
    // Transpose (box -> box, both axes independent): u = r.y, v = r.x.
    ReindexResult res = f.update().reindex({u, v}, {r.y, r.x}, {v, u});
    RVar ui("ui"), vi("vi");
    f.update().gpu_tile(res.vars[0], res.vars[1], ui, vi, 8, 8);

    Buffer<int> out = f.realize({W, H}, target);
    out.copy_to_host();
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (out(xx, yy) != xx * 10 + yy) {
                printf("mismatch at (%d,%d): %d vs %d\n", xx, yy, out(xx, yy),
                       xx * 10 + yy);
                return 1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
