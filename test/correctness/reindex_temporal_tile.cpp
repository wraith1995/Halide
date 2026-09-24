#include "Halide.h"
#include <cstdio>
#include <vector>

using namespace Halide;

// Temporal tiling of a 1-D three-point stencil iterated over time, built from
// two reindex() calls. The first skews space by time (u = x + t) so that every
// dependence points to an earlier v. The second diamond-tiles the skewed (u, v)
// space: bv (the block anti-diagonal) runs serially and p (the block position
// on it) runs in parallel.
//
// The loop over bv is partitioned, and the parallel loop over p inside it is
// partitioned again into an if-tree. Trimming the no-op iterations of bv then
// has to solve a condition relaxed over p, which is a min/max of booleans.

namespace {

int stencil(int l, int c, int r) {
    return (l + 2 * c + r + 1) % 1009;
}

bool run(int W, int T, int tile) {
    std::vector<int> in(W);
    for (int x = 0; x < W; x++) {
        in[x] = (x * 7 + 3) % 1009;
    }

    std::vector<std::vector<int>> ref(T, std::vector<int>(W, 0));
    ref[0] = in;
    for (int t = 1; t < T; t++) {
        for (int x = 1; x < W - 1; x++) {
            ref[t][x] = stencil(ref[t - 1][x - 1], ref[t - 1][x], ref[t - 1][x + 1]);
        }
    }

    Buffer<int> input(in.data(), W);
    Func f("f");
    Var x("x"), t("t");
    f(x, t) = select(t == 0, input(x), 0);
    RDom r(1, W - 2, 1, T - 1, "r");
    f(r.x, r.y) = (f(r.x - 1, r.y - 1) + 2 * f(r.x, r.y - 1) + f(r.x + 1, r.y - 1) + 1) % 1009;

    Var u("u"), v("v");
    ReindexResult skew = f.update().reindex({u, v}, {r.x + r.y, r.y}, {u - v, v});
    RVar uo("uo"), ui("ui"), vo("vo"), vi("vi");
    f.update().split(skew.vars[0], uo, ui, tile).split(skew.vars[1], vo, vi, tile);

    Var lui("ui"), luo("uo"), lvi("vi"), lvo("vo");
    Var nui("nui"), bv("bv"), nvi("nvi"), p("p");
    ReindexResult diamond = f.update().reindex(
        {nui, bv, nvi, p},
        {lui, luo + lvo, lvi, lvo},
        {nui, bv - p, nvi, p},
        {nui, nvi, p, bv});
    if (!diamond.parallel_safe[3]) {
        printf("expected the block position p to be parallel-safe\n");
        return false;
    }
    f.update().parallel(diamond.vars[3]);

    Buffer<int> out = f.realize({W, T});
    for (int tt = 0; tt < T; tt++) {
        for (int xx = 0; xx < W; xx++) {
            if (out(xx, tt) != ref[tt][xx]) {
                printf("mismatch W=%d T=%d tile=%d at (%d, %d): %d vs %d\n",
                       W, T, tile, xx, tt, out(xx, tt), ref[tt][xx]);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    struct Case {
        int W, T, tile;
    };
    const Case cases[] = {
        {96, 48, 16},
        {32, 16, 8},
        {50, 21, 8},
        {17, 40, 4},
    };
    for (const Case &c : cases) {
        if (!run(c.W, c.T, c.tile)) {
            return 1;
        }
    }
    printf("Success!\n");
    return 0;
}
