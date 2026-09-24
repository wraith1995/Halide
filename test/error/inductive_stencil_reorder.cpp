#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func f(Int(32), "f"), g("g");
    Var x("x"), t("t");

    // f reads its neighbours along x in the previous step along t, so the
    // loop over x has to be inside the loop over t.
    f(x, t) = select(t <= 0 || x <= 0 || x >= 9, x,
                     likely(f(x - 1, t - 1) + f(x + 1, t - 1)));
    g(x) = f(x, 5);
    f.compute_root().reorder(t, x);

    g.realize({10});

    printf("Success!\n");
    return 0;
}
