#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), y("y");
    f(x, y) = x + y;
    RDom r(0, 15, 1, 15);
    // Each row reads the row before it, one column to the right.
    f(r.x, r.y) = f(r.x + 1, r.y - 1) * 2;
    // With the loop over x outermost, f(x + 1, y - 1) is read before the
    // iteration that writes it has run.
    f.update().reorder(r.y, r.x);
    f.realize({16, 16});

    printf("Success!\n");
    return 0;
}
