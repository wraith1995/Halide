#include "Halide.h"
#include <cstdio>

// Whether reindex() reports a reduction variable safe to parallelize depends on
// the loops outside it: two iterations of a parallel loop share the values of
// the loops outside it, and nothing about the loops inside it.

using namespace Halide;

namespace {

const int N = 32;

bool same(const Buffer<int> &a, const Buffer<int> &b) {
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (a(x, y) != b(x, y)) {
                printf("mismatch at (%d, %d): %d vs %d\n", x, y, a(x, y), b(x, y));
                return false;
            }
        }
    }
    return true;
}

// f(x, y) reads f(x - 1, y) and f(x - 1, y + 1).
Func reads_up_and_left(const char *name, RDom &r) {
    Func f(name);
    Var x("x"), y("y");
    f(x, y) = x + 3 * y;
    r = RDom(1, N - 1, 0, N - 1);
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x - 1, r.y + 1);
    return f;
}

// f(x, y) reads f(x - 1, y) and f(x, y - 1).
Func wavefront(const char *name, RDom &r) {
    Func f(name);
    Var x("x"), y("y");
    f(x, y) = x + y;
    r = RDom(1, N - 1, 1, N - 1);
    f(r.x, r.y) = (f(r.x - 1, r.y) + f(r.x, r.y - 1)) % 1009;
    return f;
}

}  // namespace

int main(int argc, char **argv) {
    // The loop over y is outside the loop over x, which carries a dependence.
    // Iterations of y read what other iterations of y write at other x, so y
    // isn't safe to parallelize even though x is serial.
    {
        RDom r;
        Func f = reads_up_and_left("f_up_left", r);
        Var u("u"), v("v");
        ReindexResult res = f.update().reindex({u, v}, {r.x, r.y}, {u, v});
        if (res.parallel_safe[0] || res.parallel_safe[1]) {
            printf("expected neither axis to be parallel-safe, got {%d, %d}\n",
                   (int)res.parallel_safe[0], (int)res.parallel_safe[1]);
            return 1;
        }
    }

    // In wavefront order, the position on a wavefront is parallel-safe inside
    // the loop over wavefronts.
    {
        RDom r;
        Func serial = wavefront("f_wave_serial", r);
        Buffer<int> expected = serial.realize({N, N});

        Func f = wavefront("f_wave", r);
        Var u("u"), v("v");
        ReindexResult res = f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        if (!res.parallel_safe[0] || res.parallel_safe[1]) {
            printf("wavefront: expected {1, 0}, got {%d, %d}\n",
                   (int)res.parallel_safe[0], (int)res.parallel_safe[1]);
            return 1;
        }
        f.update().parallel(res.vars[0]);
        if (!same(f.realize({N, N}), expected)) {
            return 1;
        }
    }

    // In a diamond tiling, the position of a tile on a diagonal of tiles is
    // parallel-safe inside the loop over diagonals.
    {
        RDom r;
        Func serial = wavefront("f_diamond_serial", r);
        Buffer<int> expected = serial.realize({N, N});

        Func f = wavefront("f_diamond", r);
        RVar rxo("rxo"), rxi("rxi"), ryo("ryo"), ryi("ryi");
        f.update().split(r.x, rxo, rxi, 8).split(r.y, ryo, ryi, 8);
        Var lxi("rxi"), lxo("rxo"), lyi("ryi"), lyo("ryo");
        Var nxi("nxi"), bv("bv"), nyi("nyi"), nyo("nyo");
        ReindexResult res = f.update().reindex({nxi, bv, nyi, nyo}, {lxi, lxo + lyo, lyi, lyo},
                                               {nxi, bv - nyo, nyi, nyo}, {nxi, nyi, nyo, bv});
        if (!res.parallel_safe[3] || res.parallel_safe[1]) {
            printf("diamond: expected the tile position and not the diagonal to be "
                   "parallel-safe\n");
            return 1;
        }
        f.update().parallel(res.vars[3]);
        if (!same(f.realize({N, N}), expected)) {
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
