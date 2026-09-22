#include "Halide.h"

#include <cstdio>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x");
    RDom r(0, 8, "r");
    f(x) = 0;
    f(x) += r;  // f now has an update definition

    Var u("u");
    // Error: reindex_storage() currently supports pure Funcs only, but f has
    // update definitions.
    f.reindex_storage({u}, {x}, {u});

    printf("Success!\n");
    return 0;
}
