#include "Halide.h"

#include <cstdio>

// Reindex a reduction whose update also has a PURE Var dimension. f(x) sums
// in(x, r.x, r.y) over a 2-D RDom, so x is a pure loop of the update while r.x/
// r.y are the reduction. reindex() skews the two rvars and must leave the pure
// dimension x untouched; the sum is unchanged.

using namespace Halide;

int main(int argc, char **argv) {
    const int N = 6, W = 5;
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
        return f.realize({W});
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
