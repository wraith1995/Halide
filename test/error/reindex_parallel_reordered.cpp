#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    const int W = 16, H = 16;
    Func f("f");
    Var x("x"), y("y"), u("u"), v("v");
    f(x, y) = x + y;
    RDom r(1, W - 1, 1, H - 1);
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1);
    // Skew into wavefronts: v numbers the wavefront and u the position on it.
    // Positions on one wavefront are independent, so u is parallel-safe inside
    // the loop over v.
    ReindexResult res = f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
    // Moving u outside v makes its iterations depend on each other.
    f.update().reorder(res.vars[1], res.vars[0]).parallel(res.vars[0]);
    f.realize({W, H});

    printf("Success!\n");
    return 0;
}
