#include "Halide.h"
#include "halide_benchmark.h"

#include <cstdio>
#include <thread>

// Performance of diamond (wavefront-of-tiles) parallelism built with
// Stage::reindex(). A 2-D recurrence f(x,y)=f(x-1,y)+f(x,y-1)+heavy(in) carries a
// dependence along both axes, so the plain form is fully serial. Splitting into
// tiles and reindexing the block indices into a block-anti-diagonal wavefront
// exposes a parallel block axis. The result must stay bit-exact with the serial
// form, and the parallel version should be meaningfully faster on a multicore
// machine.

using namespace Halide;
using namespace Halide::Tools;

namespace {

Expr heavy(Expr v, int c) {
    for (int i = 0; i < c; i++) {
        v = v * 3 + (v >> 1) + 1;
    }
    return v;
}

Func build(bool diamond, int W, int H, int T, const ImageParam &in) {
    Func f("f");
    Var x("x"), y("y");
    f(x, y) = in(x, y);
    RDom r(1, W - 1, 1, H - 1, "r");
    f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1) + heavy(in(r.x, r.y), 32);
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

    const int W = 1025, H = 1025, T = 64;  // interior 1024x1024, divisible by T
    Buffer<int> input(W, H);
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            input(x, y) = (x * 3 + y * 5 + 1) % 11;
        }
    }
    ImageParam in(Int(32), 2);
    in.set(input);

    Func serial = build(false, W, H, T, in);
    Func diamond = build(true, W, H, T, in);

    Buffer<int> so(W, H), dio(W, H);
    serial.realize(so);
    diamond.realize(dio);

    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            if (so(x, y) != dio(x, y)) {
                printf("Mismatch at (%d,%d): serial %d vs diamond %d\n", x, y,
                       so(x, y), dio(x, y));
                return 1;
            }
        }
    }

    const double ts = benchmark(3, 3, [&]() { serial.realize(so); });
    const double td = benchmark(3, 3, [&]() { diamond.realize(dio); });
    const double speedup = ts / td;
    printf("serial  = %.2f ms\ndiamond = %.2f ms\nspeedup = %.2fx\n", ts * 1e3,
           td * 1e3, speedup);

    const unsigned hw = std::thread::hardware_concurrency();
    if (hw >= 4) {
        const double want = 1.3;
        if (speedup < want) {
            printf("Diamond parallelism too slow: %.2fx < %.2fx on %u cores\n",
                   speedup, want, hw);
            return 1;
        }
    } else {
        printf("[SKIP] speedup assertion (only %u hardware threads)\n", hw);
    }

    printf("Success!\n");
    return 0;
}
