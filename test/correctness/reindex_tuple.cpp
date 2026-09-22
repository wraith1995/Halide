#include "Halide.h"

#include <cstdio>
#include <utility>

// Reindex a *tuple* reduction: sum and sum-of-squares over a box. Skewing the
// reduction domain must leave both tuple components unchanged.

using namespace Halide;

namespace {

std::pair<int, int> run(const Buffer<int> &in, bool skew) {
    Func f("f");
    RDom r(0, in.width(), 0, in.height(), "r");
    f() = Tuple(0, 0);
    Expr val = in(r.x, r.y);
    f() = Tuple(f()[0] + val, f()[1] + val * val);
    if (skew) {
        Var u("u"), v("v");
        // u = r.x, v = r.y + r.x ; inverse r.x = u, r.y = v - u.
        f.update().reindex({u, v}, {r.x, r.y + r.x}, {u, v - u});
    }
    Realization out = f.realize();
    Buffer<int> s = out[0], q = out[1];
    return {s(), q()};
}

}  // namespace

int main(int argc, char **argv) {
    const int W = 11, H = 9;
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = x * 3 + y * 2 + 1;
        }
    }

    const auto ref = run(in, false);
    const auto got = run(in, true);
    if (ref != got) {
        printf("tuple reindex mismatch: (%d,%d) vs (%d,%d)\n", ref.first,
               ref.second, got.first, got.second);
        return 1;
    }

    printf("Success!\n");
    return 0;
}
