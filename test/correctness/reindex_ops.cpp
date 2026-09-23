#include "Halide.h"

#include <cstdio>
#include <utility>

// Extra reindex coverage: an associative op other than sum (tuple argmin), a
// 3-D reduction domain, and reorder() composition on reindexed PureRVars.

using namespace Halide;

namespace {

// Tuple argmin over a box, reindexed by a skew. With strictly distinct values the
// argmin (value and index) is order-independent, so the skew must not change it.
bool argmin() {
    const int W = 7, H = 6;
    Func in("in");
    Var x("x"), y("y");
    in(x, y) = x * 7 + y * 13 + 1;  // distinct across the box

    auto run = [&](bool skew) {
        Func f("f");
        RDom r(0, W, 0, H, "r");
        f() = Tuple(in(0, 0), 0);
        Expr val = in(r.x, r.y), idx = r.x + W * r.y;
        Expr better = val < f()[0];
        f() = Tuple(select(better, val, f()[0]), select(better, idx, f()[1]));
        if (skew) {
            Var u("u"), v("v");
            f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        }
        Realization o = f.realize();
        Buffer<int> mv = o[0], mi = o[1];
        return std::make_pair(mv(), mi());
    };

    if (run(false) != run(true)) {
        printf("argmin: reindex changed the result\n");
        return false;
    }
    return true;
}

// A 3-D reduction domain reindexed by a 3-D skew; the sum is invariant.
bool sum3d() {
    const int A = 4, B = 5, C = 6;
    Buffer<int> in(A, B, C, "in");
    for (int a = 0; a < A; a++) {
        for (int b = 0; b < B; b++) {
            for (int c = 0; c < C; c++) {
                in(a, b, c) = a + 2 * b + 3 * c + 1;
            }
        }
    }
    auto run = [&](bool skew) {
        Func f("f");
        RDom r(0, A, 0, B, 0, C, "r");
        f() = 0;
        f() += in(r.x, r.y, r.z);
        if (skew) {
            Var u("u"), v("v"), w("w");
            // u=rx, v=rx+ry, w=ry+rz ; inverse rx=u, ry=v-u, rz=w-(v-u).
            f.update().reindex({u, v, w}, {r.x, r.x + r.y, r.y + r.z},
                               {u, v - u, w - (v - u)});
        }
        Realization o = f.realize();
        Buffer<int> b = o[0];
        return b();
    };
    if (run(false) != run(true)) {
        printf("sum3d: reindex changed the result\n");
        return false;
    }
    return true;
}

// After reindex, two proven-safe axes are PureRVars, which reorder() may swap
// freely (no associativity/commutativity proof required for pure dims).
bool reorder_pure() {
    const int W = 6, H = 5;
    Func in("in"), f("f");
    Var x("x"), y("y");
    in(x, y) = x * 10 + y;
    f(x, y) = 0;
    RDom r(0, W, 0, H, "r");
    f(r.x, r.y) = in(r.x, r.y);
    Var u("u"), v("v");
    ReindexResult res = f.update().reindex({u, v}, {r.y, r.x}, {v, u});  // transpose
    f.update().reorder(res.vars[0], res.vars[1]);

    Buffer<int> o = f.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (o(xx, yy) != xx * 10 + yy) {
                printf("reorder_pure: wrong at (%d,%d): %d\n", xx, yy, o(xx, yy));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (!argmin() || !sum3d() || !reorder_pure()) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
