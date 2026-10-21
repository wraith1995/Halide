#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), u("u"), v("v");
    f(x) = 0;
    RDom r(0, 4, 0, 4);
    f(r.x) += r.y + 1;
    // The loop order names u twice.
    f.update().reindex({u, v}, {r.x, r.y}, {u, v}, {u, u});
    f.realize({4});

    printf("Success!\n");
    return 0;
}
