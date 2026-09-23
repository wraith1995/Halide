#include "Halide.h"

#include <cstdio>

// The reindexed parallel axis (a PureRVar) must compose with ordinary scheduling
// directives -- parallel, vectorize, split -- and still produce the reference
// result. This is the "interacts with the rest of Halide" check for reindex().

using namespace Halide;

namespace {

const int W = 16, H = 12;

Buffer<int> reference() {
    Buffer<int> f(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            f(x, y) = x + y;
        }
    }
    for (int y = 1; y < H; y++) {
        for (int x = 1; x < W; x++) {
            f(x, y) = f(x - 1, y) + f(x, y - 1) + (x + y);
        }
    }
    return f;
}

bool run(int mode, const char *name) {
    Func in("in"), f("f");
    Var x("x"), y("y");
    in(x, y) = x + y;
    f(x, y) = in(x, y);
    RDom r(1, W - 1, 1, H - 1, "r");
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1) + in(r.x, r.y);

    Var u("u"), v("v");
    ReindexResult res = f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
    RVar U = res.vars[0];  // the parallel-safe (anti-diagonal position) axis

    switch (mode) {
    case 0:
        f.update().parallel(U);
        break;
    case 1:
        f.update().vectorize(U, 4);
        break;
    case 2: {
        RVar uo("uo"), ui("ui");
        f.update().split(U, uo, ui, 8).parallel(uo);
        break;
    }
    case 3: {
        RVar uo("uo"), ui("ui");
        f.update().split(U, uo, ui, 8).parallel(uo).vectorize(ui, 4);
        break;
    }
    }

    Buffer<int> out = f.realize({W, H});
    Buffer<int> ref = reference();
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (out(x, y) != ref(x, y)) {
                printf("%s: mismatch at (%d,%d): %d vs %d\n", name, x, y,
                       out(x, y), ref(x, y));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (!run(0, "parallel(u)")) {
        return 1;
    }
    if (!run(1, "vectorize(u, 4)")) {
        return 1;
    }
    if (!run(2, "split(u) + parallel(uo)")) {
        return 1;
    }
    if (!run(3, "split(u) + parallel(uo) + vectorize(ui, 4)")) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
