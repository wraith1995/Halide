#include "Halide.h"

#include <cstdio>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    RDom r(0, 8, 0, 8, "r");
    f() = 0;
    f() += r.x + r.y;

    Var u("u"), v("v"), w("w");
    // Error: loop_order names w, which is not one of the new reduction dims.
    f.update().reindex({u, v}, {r.x, r.y}, {u, v}, {u, w});

    printf("Success!\n");
    return 0;
}
