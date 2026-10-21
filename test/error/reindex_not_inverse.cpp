#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), u("u"), v("v");
    f(x) = 0;
    RDom r(0, 4, 0, 4);
    f(r.x) += r.y * 10 + r.x;
    // The inverse of v = r.y + r.x is r.y = v - u, not v + u.
    f.update().reindex({u, v}, {r.x, r.y + r.x}, {u, v + u});
    f.realize({4});

    printf("Success!\n");
    return 0;
}
