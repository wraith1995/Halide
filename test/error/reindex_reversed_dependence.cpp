#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), u("u");
    f(x) = x;
    RDom r(0, 10);
    // Each iteration reads the element the next one writes. Reversing the
    // loop would run the next one first.
    f(r) = f(r + 1) * 2 + 1;
    f.update().reindex({u}, {9 - r.x}, {9 - u});
    f.realize({11});

    printf("Success!\n");
    return 0;
}
