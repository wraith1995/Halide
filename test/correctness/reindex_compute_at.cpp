#include "Halide.h"

#include <cstdio>

// A reindexed reduction must still schedule normally against its consumers.
// f(x) sums in(x, r.x, r.y); after reindexing the two rvars by a skew, f is
// computed at a split loop of its consumer g. The result is unchanged.

using namespace Halide;

int main(int argc, char **argv) {
    const int W = 6, N = 5;
    Buffer<int> in(W, N, N, "in");
    for (int a = 0; a < W; a++) {
        for (int b = 0; b < N; b++) {
            for (int c = 0; c < N; c++) {
                in(a, b, c) = a + 2 * b + 3 * c + 1;
            }
        }
    }

    auto run = [&](bool skew) {
        Func f("f");
        Var x("x");
        f(x) = 0;
        RDom r(0, N, 0, N, "r");
        f(x) += in(x, r.x, r.y);
        if (skew) {
            Var u("u"), v("v");
            f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        }
        Func g("g");
        g(x) = f(x) * 2;
        Var xo("xo"), xi("xi");
        g.split(x, xo, xi, 2);
        f.compute_at(g, xo);
        return g.realize({W});
    };

    Buffer<int> ref = run(false);
    Buffer<int> got = run(true);
    for (int x = 0; x < W; x++) {
        if (ref(x) != got(x)) {
            printf("mismatch x=%d: %d vs %d\n", x, ref(x), got(x));
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
