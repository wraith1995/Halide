#include "Halide.h"
#include <cstdio>

using namespace Halide;

// A 2D neighbour recurrence f(x,y) = f(x-1,y) + f(x,y-1) + in(x,y) has a carried
// dependency along both axes, so neither x nor y is parallel. Skewing the
// iteration space by v = x + y makes each anti-diagonal independent: iterating v
// serially and u = x in parallel is a valid (wavefront) schedule. This exercises
// Stage::reindex() -- the affine index-bijection built on the reduction-rewrite
// core -- on an update with a self-referential (recurrence) body, and then marks
// the exposed dimension parallel.

namespace {

const int W = 11, H = 8;

Buffer<int> make_input() {
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = (x * 3 + y * 5 + 1) % 11;
        }
    }
    return in;
}

// mode: 0 = unskewed, 1 = skewed serial, 2 = skewed with parallel(u).
Buffer<int> compute(const Buffer<int> &in, int mode) {
    Func f("f");
    Var x("x"), y("y");
    f(x, y) = in(x, y);
    RDom r(1, W - 1, 1, H - 1, "r");
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1) + in(r.x, r.y);

    if (mode != 0) {
        Var u("u"), v("v");
        // u = r.x, v = r.x + r.y ; inverse r.x = u, r.y = v - u.
        f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        if (mode == 2) {
            // The reindexed dims are RVars; u is independent within each v.
            f.update().allow_race_conditions().parallel(RVar("u"));
        }
    }
    return f.realize({W, H});
}

}  // namespace

int main() {
    Buffer<int> in = make_input();

    // Serial reference: interior recurrence, boundary = in.
    Buffer<int> ref(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (x == 0 || y == 0) {
                ref(x, y) = in(x, y);
            } else {
                ref(x, y) = ref(x - 1, y) + ref(x, y - 1) + in(x, y);
            }
        }
    }

    const char *labels[] = {"unskewed", "skewed serial", "skewed parallel(u)"};
    for (int mode = 0; mode < 3; mode++) {
        Buffer<int> out = compute(in, mode);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (out(x, y) != ref(x, y)) {
                    printf("%s mismatch at (%d,%d): %d vs %d\n", labels[mode], x, y,
                           out(x, y), ref(x, y));
                    return 1;
                }
            }
        }
    }

    printf("Success!\n");
    return 0;
}
