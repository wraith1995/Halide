#include "Halide.h"
#include <cstdio>

using namespace Halide;

namespace {

// Sum in(r.x, r.y) over the WxH box, scheduling the update with `sched`, and
// return the scalar result. `sched` receives the update Stage so it can apply an
// index bijection (reindex) to it.
template<typename Sched>
int skewed_sum(const Buffer<int> &in, Sched sched) {
    const int W = in.width(), H = in.height();
    Func f;
    RDom r(0, W, 0, H, "r");
    f() = 0;
    f() += in(r.x, r.y);
    sched(f.update(), r);
    Buffer<int> out = f.realize();
    return out();
}

}  // namespace

int main() {
    const int W = 13, H = 10;
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = x * 7 + y * 3 + 1;
        }
    }

    // Reference: no reindexing.
    const int ref = skewed_sum(in, [](Stage, RDom) {});

    // Case 1: a standard skew, u = r.x, v = r.y + r.x (inverse r.x = u, r.y = v - u).
    {
        const int got = skewed_sum(in, [](Stage s, RDom r) {
            Var u("u"), v("v");
            s.reindex({u, v}, {r.x, r.y + r.x}, {u, v - u});
        });
        if (got != ref) {
            printf("skew (u=x, v=y+x) mismatch: got %d, expected %d\n", got, ref);
            return 1;
        }
    }

    // Case 2: a shear the other way, u = r.x + 2*r.y, v = r.y
    // (inverse r.y = v, r.x = u - 2*v).
    {
        const int got = skewed_sum(in, [](Stage s, RDom r) {
            Var u("u"), v("v");
            s.reindex({u, v}, {r.x + 2 * r.y, r.y}, {u - 2 * v, v});
        });
        if (got != ref) {
            printf("shear (u=x+2y, v=y) mismatch: got %d, expected %d\n", got, ref);
            return 1;
        }
    }

    // Case 3: identity bijection (u = r.x, v = r.y) -- degenerate but must hold.
    {
        const int got = skewed_sum(in, [](Stage s, RDom r) {
            Var u("u"), v("v");
            s.reindex({u, v}, {r.x, r.y}, {u, v});
        });
        if (got != ref) {
            printf("identity reindex mismatch: got %d, expected %d\n", got, ref);
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
