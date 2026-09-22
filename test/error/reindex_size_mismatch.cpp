#include "Halide.h"

#include <cstdio>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    RDom r(0, 8, 0, 8, "r");
    f() = 0;
    f() += r.x + r.y;

    Var u("u");
    // Error: reindex() expects a square bijection matching the 2 reduction
    // dimensions, but is given only one new var / forward / inverse.
    f.update().reindex({u}, {r.x}, {u});

    printf("Success!\n");
    return 0;
}
