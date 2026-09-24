#include "Halide.h"
#include "halide_benchmark.h"

#include <algorithm>
#include <cstdio>
#include <random>
#include <thread>
#include <vector>

// Pairwise sequence alignment (Gotoh, affine gap) -- the dynamic program from
// Tang, Chowdhury, Luk & Leiserson, "Coding Stencil Computations Using the
// Pochoir Stencil-Specification Language" (HotPar'11), Section 4. Each cell of
// the (G, D, I) grid depends on its up, left, and diagonal neighbours, so the DP
// cannot be parallelized by loops -- it is serial. Reindex performs the same
// diamond transform the paper does by hand (their Figure 3), tiling and skewing
// the block grid into a block-anti-diagonal wavefront so the block-position axis
// is parallel. This benchmarks the DP with and without that transform; the two
// must be bit-exact.

using namespace Halide;
using namespace Halide::Tools;

namespace {

const int GO = 4, GE = 1, MATCH = 2, MISS = -1, NEG = -1000000;

Func build(bool diamond, int N, int M, int T, const ImageParam &X,
           const ImageParam &Y) {
    Var i("i"), j("j");
    Func f("f");
    Expr Gb = select(i == 0 && j == 0, 0,
                     select(i == 0, -(GO + GE * j),
                            select(j == 0, -(GO + GE * i), 0)));
    f(i, j) = Tuple(Gb, NEG, NEG);
    RDom r(1, N, 1, M, "r");
    Expr s = select(X(r.x - 1) == Y(r.y - 1), MATCH, MISS);
    Expr Gup = f(r.x - 1, r.y)[0], Dup = f(r.x - 1, r.y)[1];
    Expr Gle = f(r.x, r.y - 1)[0], Ile = f(r.x, r.y - 1)[2];
    Expr Gdi = f(r.x - 1, r.y - 1)[0];
    Expr D = max(Gup - (GO + GE), Dup - GE);
    Expr I = max(Gle - (GO + GE), Ile - GE);
    Expr G = max(max(Gdi + s, D), I);
    f(r.x, r.y) = Tuple(G, D, I);
    if (diamond) {
        RVar rxo("rxo"), rxi("rxi"), ryo("ryo"), ryi("ryi");
        f.update().split(r.x, rxo, rxi, T).split(r.y, ryo, ryi, T);
        Var lxi("rxi"), lxo("rxo"), lyi("ryi"), lyo("ryo");
        Var nxi("nxi"), bv("bv"), nyi("nyi"), nyo("nyo");
        ReindexResult res = f.update().reindex(
            {nxi, bv, nyi, nyo}, {lxi, lxo + lyo, lyi, lyo},
            {nxi, bv - nyo, nyi, nyo}, {nxi, nyi, nyo, bv});
        f.update().parallel(res.vars[3]);  // block position on the anti-diagonal
    }
    return f;
}

}  // namespace

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (target.arch == Target::WebAssembly) {
        printf("[SKIP] Performance is meaningless under the WebAssembly "
               "interpreter.\n");
        return 0;
    }

    const int N = 2048, M = 2048, T = 64;  // interior divisible by T
    Buffer<int> xb(N), yb(M);
    std::mt19937 rng(7);
    std::uniform_int_distribution<int> d(0, 3);
    for (int k = 0; k < N; k++) xb(k) = d(rng);
    for (int k = 0; k < M; k++) yb(k) = d(rng);
    ImageParam X(Int(32), 1), Y(Int(32), 1);
    X.set(xb);
    Y.set(yb);

    Func serial = build(false, N, M, T, X, Y);
    Func diamond = build(true, N, M, T, X, Y);
    Realization so = serial.realize({N + 1, M + 1});
    Realization di = diamond.realize({N + 1, M + 1});
    Buffer<int> sG = so[0], dG = di[0];

    // C++ reference score for the same recurrence.
    std::vector<int> pG(M + 1), pD(M + 1), pI(M + 1), cG(M + 1), cD(M + 1), cI(M + 1);
    for (int j = 0; j <= M; j++) {
        pG[j] = (j == 0) ? 0 : -(GO + GE * j);
        pD[j] = NEG;
        pI[j] = NEG;
    }
    for (int i = 1; i <= N; i++) {
        cG[0] = -(GO + GE * i);
        cD[0] = NEG;
        cI[0] = NEG;
        for (int j = 1; j <= M; j++) {
            int sc = (xb(i - 1) == yb(j - 1)) ? MATCH : MISS;
            cD[j] = std::max(pG[j] - (GO + GE), pD[j] - GE);
            cI[j] = std::max(cG[j - 1] - (GO + GE), cI[j - 1] - GE);
            cG[j] = std::max(std::max(pG[j - 1] + sc, cD[j]), cI[j]);
        }
        std::swap(pG, cG);
        std::swap(pD, cD);
        std::swap(pI, cI);
    }
    const int refscore = pG[M];

    if (sG(N, M) != refscore || dG(N, M) != refscore) {
        printf("Wrong score: serial %d, diamond %d, ref %d\n", sG(N, M),
               dG(N, M), refscore);
        return 1;
    }
    for (int j = 0; j <= M; j++) {
        for (int i = 0; i <= N; i++) {
            if (sG(i, j) != dG(i, j)) {
                printf("Mismatch at (%d,%d): serial %d vs diamond %d\n", i, j,
                       sG(i, j), dG(i, j));
                return 1;
            }
        }
    }

    const double ts = benchmark(3, 3, [&]() { serial.realize(so); });
    const double td = benchmark(3, 3, [&]() { diamond.realize(di); });
    const double speedup = ts / td;
    printf("Gotoh alignment %dx%d (score=%d)\n", N, M, refscore);
    printf("without reindex (serial loops)   = %.1f ms\n", ts * 1e3);
    printf("with reindex (diamond wavefront) = %.1f ms\n", td * 1e3);
    printf("speedup = %.2fx\n", speedup);

    const unsigned hw = std::thread::hardware_concurrency();
    if (hw >= 4) {
        if (speedup <= 1.0) {
            printf("Diamond parallelism no faster than serial (%.2fx) on %u "
                   "cores\n",
                   speedup, hw);
            return 1;
        }
    } else {
        printf("[SKIP] speedup assertion (only %u hardware threads)\n", hw);
    }

    printf("Success!\n");
    return 0;
}
