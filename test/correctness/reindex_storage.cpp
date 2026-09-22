#include "Halide.h"

#include <cstdio>

// Func::reindex_storage() changes a Func's storage layout by an affine bijection
// on its pure dimensions -- the storage analogue of Stage::reindex(). Here we
// reflect the storage of an intermediate along x (store index u = W-1-x), which
// reorder_storage() cannot express. The original Func becomes a trivial wrapper
// reading the reindexed store, so the consumer is unaffected and the result is
// unchanged; the data physically lives reflected in the "_storage" Func.

using namespace Halide;

int main(int argc, char **argv) {
    const int W = 20, H = 12;
    Var x("x"), y("y"), u("u"), v("v");

    Func in("in");
    in(x, y) = x + 100 * y;

    Func f("f");
    f(x, y) = in(x, y) + 1;

    // u = W-1-x, v = y (a reflection: box -> box, so no out-of-domain corners);
    // inverse x = W-1-u, y = v.
    Func fs = f.reindex_storage({u, v}, {W - 1 - x, y}, {W - 1 - u, v});
    fs.compute_root();

    // The reindexed store holds f's value at the reflected coordinate.
    Buffer<int> fsb = fs.realize({W, H});
    for (int vv = 0; vv < H; vv++) {
        for (int uu = 0; uu < W; uu++) {
            const int expected = (W - 1 - uu) + 100 * vv + 1;
            if (fsb(uu, vv) != expected) {
                printf("f_storage mismatch at (%d,%d): %d vs %d\n", uu, vv,
                       fsb(uu, vv), expected);
                return 1;
            }
        }
    }

    // A consumer that reads f normally sees the unchanged logical values.
    Func g("g");
    g(x, y) = f(x, y) * 2;

    Buffer<int> out = g.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            const int expected = (xx + 100 * yy + 1) * 2;
            if (out(xx, yy) != expected) {
                printf("g mismatch at (%d,%d): %d vs %d\n", xx, yy, out(xx, yy),
                       expected);
                return 1;
            }
        }
    }

    printf("Success!\n");
    return 0;
}
