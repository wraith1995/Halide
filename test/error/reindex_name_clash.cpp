#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x");
    f(x) = 0;
    RDom r(0, 4);
    f(x) += r * (x + 1);
    // The new reduction variable has the name of the pure variable x.
    Var also_x("x");
    f.update().reindex({also_x}, {r.x}, {also_x});
    f.realize({4});

    printf("Success!\n");
    return 0;
}
