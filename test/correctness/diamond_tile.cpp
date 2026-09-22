#include "Halide.h"
#include <cstdio>

using namespace Halide;

// Diamond (wavefront) tiling of a 2D neighbour recurrence, built by composing
// split() with Stage::reindex(): tile the reduction domain, then skew the *block*
// indices so blocks on a block-anti-diagonal are independent. The block-diagonal
// (bv) runs serially, the block position on it (nyo) runs in parallel, and each
// block is computed serially (good locality). This is the coarse-grained form
// that turns the correct-but-slow point-wise wavefront into a real speedup.
//
// Here we only check correctness of the schedule for several sizes and tile
// sizes (including non-divisible ones, which exercise the split predicates).

namespace {

Buffer<int> reference(const Buffer<int> &in) {
    const int W = in.width(), H = in.height();
    Buffer<int> ref(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            ref(x, y) = (x == 0 || y == 0) ? in(x, y)
                                           : ref(x - 1, y) + ref(x, y - 1) + in(x, y);
        }
    }
    return ref;
}

bool run(int W, int H, int T) {
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = (x * 3 + y * 5 + 1) % 11;
        }
    }
    Buffer<int> ref = reference(in);

    Func f("f");
    Var x("x"), y("y");
    f(x, y) = in(x, y);
    RDom r(1, W - 1, 1, H - 1, "r");
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1) + in(r.x, r.y);

    RVar rxo("rxo"), rxi("rxi"), ryo("ryo"), ryi("ryi");
    f.update().split(r.x, rxo, rxi, T).split(r.y, ryo, ryi, T);
    // Refer to the (schedule-only) split leaves by name-matched Var placeholders.
    Var lxi("rxi"), lxo("rxo"), lyi("ryi"), lyo("ryo");
    // bv = block anti-diagonal (serial), nyo = block position on it (parallel),
    // nxi/nyi = within-block. Loop order (innermost first): nxi, nyi, nyo, bv.
    Var nxi("nxi"), bv("bv"), nyi("nyi"), nyo("nyo");
    ReindexResult res = f.update().reindex(
        {nxi, bv, nyi, nyo},
        {lxi, lxo + lyo, lyi, lyo},
        {nxi, bv - nyo, nyi, nyo},
        {nxi, nyi, nyo, bv});
    f.update().allow_race_conditions().parallel(res.vars[3]);  // nyo

    Buffer<int> out = f.realize({W, H});
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (out(x, y) != ref(x, y)) {
                printf("mismatch W=%d H=%d T=%d at (%d,%d): %d vs %d\n", W, H, T, x, y,
                       out(x, y), ref(x, y));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main() {
    // Both tile-multiple and non-tile-multiple interior extents. The latter make
    // the split emit a `likely` tail guard on the (parallel) block index; these
    // used to miscompile via trim_no_ops / solve_for_outer_interval mishandling
    // min/max of booleans (fixed in src/Solve.cpp).
    struct Case {
        int W, H, T;
    };
    const Case cases[] = {
        {9, 9, 2},     // interior 8x8, T=2   (divisible)
        {17, 9, 4},    // interior 16x8, T=4  (divisible)
        {33, 17, 8},   // interior 32x16, T=8 (divisible)
        {16, 12, 4},   // interior 15x11, T=4 (NON-divisible -> likely tail guard)
        {17, 13, 4},   // interior 16x12, T=4 (mixed)
        {23, 19, 8},   // interior 22x18, T=8 (NON-divisible)
        {5, 5, 4},     // one tile per axis (interior 4x4, T=4)
    };
    for (const Case &c : cases) {
        if (!run(c.W, c.H, c.T)) {
            return 1;
        }
    }
    printf("Success!\n");
    return 0;
}
