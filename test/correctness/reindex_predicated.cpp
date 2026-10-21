#include "Halide.h"

#include <cstdio>

// Reindex a *predicated* reduction domain. The user's .where() predicate (in old
// rvar terms) must compose with reindex's image-carving predicate, so the skewed
// reduction sums over exactly the same points as the unreindexed one.

using namespace Halide;

namespace {

int run(const Buffer<int> &in, bool skew) {
    const int W = in.width(), H = in.height();
    const int K = 10;
    Func f("f");
    RDom r(0, W, 0, H, "r");
    r.where(r.x + r.y < K);  // a triangular sub-region of the box
    f() = 0;
    f() += in(r.x, r.y);
    if (skew) {
        Var u("u"), v("v");
        // u = r.x + 2*r.y, v = r.y ; inverse r.x = u - 2*v, r.y = v.
        f.update().reindex({u, v}, {r.x + 2 * r.y, r.y}, {u - 2 * v, v});
    }
    Realization out = f.realize();
    Buffer<int> b = out[0];
    return b();
}

}  // namespace

int main(int argc, char **argv) {
    const int W = 12, H = 12;
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = x + 7 * y + 1;
        }
    }

    const int ref = run(in, false);
    const int got = run(in, true);
    if (ref != got) {
        printf("predicated reindex mismatch: %d vs %d\n", ref, got);
        return 1;
    }

    // The RVars reindex() returns carry the predicate too, in their own
    // terms, so a new definition over them visits the same points.
    {
        const int K = 10;
        Func f("f_pred"), count("count");
        RDom r(0, W, 0, H, "r");
        r.where(r.x + r.y < K);
        f() = 0;
        f() += in(r.x, r.y);
        Var u("u"), v("v");
        ReindexResult res = f.update().reindex({u, v}, {r.x + 2 * r.y, r.y}, {u - 2 * v, v});
        count() = 0;
        count() += select(res.vars[0] >= 0, 1, 0);
        Buffer<int> c = count.realize();
        const int points = K * (K + 1) / 2;
        if (c() != points) {
            printf("a new definition over the returned RVars visits %d points, not %d\n", c(), points);
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
