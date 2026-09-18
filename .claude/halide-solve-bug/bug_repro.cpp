// Minimal reproducer: likely() is documented to be a value-identity
// (see remove_likelies), affecting only loop partitioning. But when a likely()
// tail guard couples a SERIAL outer reduction loop with a PARALLEL inner one,
// loop partitioning miscompiles: it hoists a partition of the inner (parallel)
// loop out of the outer (serial) loop, dropping the outer variable, so a whole
// region of the reduction is never computed.
//
// This is a diamond/wavefront-tiled 2D recurrence expressed directly with the
// public front-end (RDom/where/likely/unsafe_promise_clamped/parallel). The
// parallelization is provably race-free (independent proof available), so
// allow_race_conditions() is truthful and the wrong result is not a race.
//
// Build:  c++ -std=c++17 -I <halide>/include bug_repro.cpp -o bug_repro \
//              -L <halide>/lib -lHalide -Wl,-rpath,<halide>/lib
//
// Expected: "plain" and "likely" both equal the serial reference.
// Actual:   with a NON-tile-multiple size, the likely+parallel version is wrong.

#include "Halide.h"
#include <cstdio>
using namespace Halide;

static Buffer<int> run(const Buffer<int> &in, int W, int H, int T, bool use_likely) {
    Func f("f");
    Var x("x"), y("y");
    f(x, y) = in(x, y);  // boundary (x==0 || y==0) stays = in

    // Tiled+skewed iteration domain, innermost-first dims: nxi, nyi, nyo, bv.
    // block-x index rxo = bv - nyo, block-y index ryo = nyo.
    //   X = T*(bv-nyo) + nxi + 1,   Y = T*nyo + nyi + 1
    int nbx = (W - 1 + T - 1) / T, nby = (H - 1 + T - 1) / T;
    RDom d({{0, T}, {0, T}, {0, nby}, {0, nbx + nby - 1}}, "d");
    RVar nxi = d[0], nyi = d[1], nyo = d[2], bv = d[3];
    Expr rxo = bv - nyo;
    Expr tailx = T * rxo + nxi;  // = X-1, valid iff <= W-2
    Expr taily = T * nyo + nyi;  // = Y-1, valid iff <= H-2
    if (use_likely) { tailx = likely(tailx); taily = likely(taily); }
    d.where(0 <= rxo && rxo <= nbx - 1 && tailx <= W - 2 && taily <= H - 2);

    Expr X = unsafe_promise_clamped(T * (bv - nyo) + nxi + 1, 1, W - 1);
    Expr Y = unsafe_promise_clamped(T * nyo + nyi + 1, 1, H - 1);
    f(X, Y) = f(X - 1, Y) + f(X, Y - 1) + in(X, Y);

    // nyo (the block index along a block-anti-diagonal) is race-free to
    // parallelize for a fixed bv; bv is the serial wavefront axis.
    f.update().allow_race_conditions().parallel(nyo);
    return f.realize({W, H});
}

static int check(const Buffer<int> &out, const Buffer<int> &ref, const char *label) {
    int mism = 0;
    for (int y = 0; y < ref.height(); y++)
        for (int x = 0; x < ref.width(); x++)
            mism += (out(x, y) != ref(x, y));
    printf("  %-24s : %s (%d mismatches)\n", label, mism ? "WRONG" : "correct", mism);
    return mism;
}

int main() {
    const int W = 16, H = 12, T = 4;  // interior extents 15,11 are NOT multiples of T
    Buffer<int> in(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            in(x, y) = (x * 3 + y * 5 + 1) % 11;

    Buffer<int> ref(W, H);
    for (int y = 0; y < H; y++)
        for (int x = 0; x < W; x++)
            ref(x, y) = (x == 0 || y == 0) ? in(x, y)
                                           : ref(x - 1, y) + ref(x, y - 1) + in(x, y);

    printf("diamond-tiled recurrence, parallel(nyo), W=%d H=%d T=%d:\n", W, H, T);
    int a = check(run(in, W, H, T, /*use_likely=*/false), ref, "plain predicate");
    int b = check(run(in, W, H, T, /*use_likely=*/true), ref, "likely tail guard");

    printf("\nlikely() is documented as a value-identity, so both must match the "
           "reference.\n");
    if (a == 0 && b != 0) {
        printf("BUG CONFIRMED: identical program is correct without likely(), wrong with it.\n");
        return 0;
    }
    printf("bug not reproduced here.\n");
    return 1;
}
