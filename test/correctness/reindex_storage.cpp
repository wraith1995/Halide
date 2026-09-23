#include "Halide.h"

#include <cstdio>

// Func::reindex_storage() changes a Func's storage layout by a bijection phi on
// its pure dimensions -- the storage analogue of Stage::reindex(). phi need not
// be affine and its validity is not checked; the caller supplies matching
// forward/inverse maps. The original Func becomes a trivial wrapper reading the
// reindexed store, so consumers are unaffected and results are unchanged; the
// data physically lives in the reindexed coordinate system.

using namespace Halide;

namespace {

// Reflection along x (u = W-1-x): an affine, box->box map that reorder_storage()
// cannot express.
bool test_reflection() {
    const int W = 20, H = 12;
    Var x("x"), y("y"), u("u"), v("v");

    Func in("in_r");
    in(x, y) = x + 100 * y;
    Func f("f_r");
    f(x, y) = in(x, y) + 1;

    Func fs = f.reindex_storage({u, v}, {W - 1 - x, y}, {W - 1 - u, v});
    fs.compute_root();

    Buffer<int> fsb = fs.realize({W, H});
    for (int vv = 0; vv < H; vv++) {
        for (int uu = 0; uu < W; uu++) {
            const int expected = (W - 1 - uu) + 100 * vv + 1;
            if (fsb(uu, vv) != expected) {
                printf("reflection store mismatch at (%d,%d): %d vs %d\n", uu, vv,
                       fsb(uu, vv), expected);
                return false;
            }
        }
    }

    Func g("g_r");
    g(x, y) = f(x, y) * 2;
    Buffer<int> out = g.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            const int expected = (xx + 100 * yy + 1) * 2;
            if (out(xx, yy) != expected) {
                printf("reflection consumer mismatch at (%d,%d): %d vs %d\n", xx,
                       yy, out(xx, yy), expected);
                return false;
            }
        }
    }
    return true;
}

// A cyclic shift along x (u = (x+K) mod W): a NON-affine bijection with a clean
// inverse (x = (u + W-K) mod W). Demonstrates that phi need not be affine.
bool test_modular_shift() {
    const int W = 16, H = 8, K = 5;
    Var x("x"), y("y"), u("u"), v("v");

    Func in("in_m");
    in(x, y) = x + 100 * y;
    Func f("f_m");
    f(x, y) = in(x, y) + 1;

    Func fs = f.reindex_storage({u, v},
                                {(x + K) % W, y},
                                {(u + (W - K)) % W, v});
    fs.compute_root();

    // The reindexed store holds f's value at the inverse coordinate.
    Buffer<int> fsb = fs.realize({W, H});
    for (int vv = 0; vv < H; vv++) {
        for (int uu = 0; uu < W; uu++) {
            const int src_x = (uu + (W - K)) % W;
            const int expected = src_x + 100 * vv + 1;
            if (fsb(uu, vv) != expected) {
                printf("modular store mismatch at (%d,%d): %d vs %d\n", uu, vv,
                       fsb(uu, vv), expected);
                return false;
            }
        }
    }

    Func g("g_m");
    g(x, y) = f(x, y) * 2;
    Buffer<int> out = g.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            const int expected = (xx + 100 * yy + 1) * 2;
            if (out(xx, yy) != expected) {
                printf("modular consumer mismatch at (%d,%d): %d vs %d\n", xx, yy,
                       out(xx, yy), expected);
                return false;
            }
        }
    }
    return true;
}

// A tuple-valued Func: reindex_storage must rewrite every component and rebuild
// the wrapper per component. Reflect along x and read both components back.
bool test_tuple() {
    const int W = 10, H = 8;
    Var x("x"), y("y"), u("u"), v("v");

    Func f("ft");
    f(x, y) = Tuple(x + 100 * y, x - y);

    Func fs = f.reindex_storage({u, v}, {W - 1 - x, y}, {W - 1 - u, v});
    fs.compute_root();

    Func g("gt");
    g(x, y) = f(x, y)[0] * 2 + f(x, y)[1];

    Buffer<int> out = g.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            const int expected = (xx + 100 * yy) * 2 + (xx - yy);
            if (out(xx, yy) != expected) {
                printf("tuple consumer mismatch at (%d,%d): %d vs %d\n", xx, yy,
                       out(xx, yy), expected);
                return false;
            }
        }
    }
    return true;
}

// Two independent consumers read the reindexed Func; both must see the unchanged
// logical values (each inlines through the wrapper to the reindexed store).
bool test_multi_consumer() {
    const int W = 10, H = 8;
    Var x("x"), y("y"), u("u"), v("v");

    Func f("fm");
    f(x, y) = x + 100 * y;

    Func fs = f.reindex_storage({u, v}, {W - 1 - x, y}, {W - 1 - u, v});
    fs.compute_root();

    Func g1("g1"), g2("g2");
    g1(x, y) = f(x, y) + 1;
    g2(x, y) = f(x, y) * 3;

    Buffer<int> o1 = g1.realize({W, H});
    Buffer<int> o2 = g2.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            if (o1(xx, yy) != (xx + 100 * yy) + 1 ||
                o2(xx, yy) != (xx + 100 * yy) * 3) {
                printf("multi-consumer mismatch at (%d,%d): %d, %d\n", xx, yy,
                       o1(xx, yy), o2(xx, yy));
                return false;
            }
        }
    }
    return true;
}

// A skew (u = x, v = x+y) maps the storage box to a parallelogram, so the store
// is computed over a bounding box whose corners fall outside the original domain.
// Reading through a boundary condition makes those corner evaluations well
// defined; the values consumers actually read are unchanged.
bool test_skew_boundary_clamped() {
    const int W = 12, H = 10;
    Var x("x"), y("y"), u("u"), v("v");

    Func in("in");
    in(x, y) = x + 100 * y;
    Func inb = BoundaryConditions::repeat_edge(in, {{0, W}, {0, H}});

    Func f("fk");
    f(x, y) = inb(x, y) + 1;

    // u = x, v = x + y ; inverse x = u, y = v - u.
    Func fs = f.reindex_storage({u, v}, {x, x + y}, {u, v - u});
    fs.compute_root();

    Func g("gk");
    g(x, y) = f(x, y) * 2;

    Buffer<int> out = g.realize({W, H});
    for (int yy = 0; yy < H; yy++) {
        for (int xx = 0; xx < W; xx++) {
            const int expected = ((xx + 100 * yy) + 1) * 2;
            if (out(xx, yy) != expected) {
                printf("skew-bc mismatch at (%d,%d): %d vs %d\n", xx, yy,
                       out(xx, yy), expected);
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    if (!test_reflection()) {
        return 1;
    }
    if (!test_modular_shift()) {
        return 1;
    }
    if (!test_tuple()) {
        return 1;
    }
    if (!test_multi_consumer()) {
        return 1;
    }
    if (!test_skew_boundary_clamped()) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
