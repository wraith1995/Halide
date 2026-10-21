#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f("f");
    Var x("x"), u("u");
    f(x) = 0;
    RDom r(0, 4, 0, 4);
    f(r.x) += r.y + 1;
    // Both new reduction variables are named u.
    f.update().reindex({u, u}, {r.x, r.y}, {u, u});
    f.realize({4});

    printf("Success!\n");
    return 0;
}
