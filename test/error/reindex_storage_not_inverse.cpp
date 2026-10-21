#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), u("u");
    f(x) = x * 3;
    // The inverse of u = x + 2 is x = u - 2, not u + 2.
    Func stored = f.reindex_storage({u}, {x + 2}, {u + 2});
    stored.compute_root();
    f.realize({8});

    printf("Success!\n");
    return 0;
}
