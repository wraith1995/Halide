#include "Halide.h"

#include <cstdio>

// reindex composes with store_at / sliding-window scheduling. f is inductive in t
// (a per-element recurrence f(r,t) = f(r,t-1)*2 + 1) with an RVar r. We reindex r
// (a reflection), then schedule f with the inductive sliding window -- computed at
// the consumer's t loop, stored at the root, and folded to just two t-slices
// (fold_storage(t, 2)). The fold is only legal because each step reads only t-1,
// which the reindexing preserves; the result must be unchanged.

using namespace Halide;

namespace {

const int W = 6, T = 5;

Buffer<int> reference() {
    Buffer<int> b(W, T);
    for (int r = 0; r < W; r++) {
        b(r, 0) = r;
    }
    for (int t = 1; t < T; t++) {
        for (int r = 0; r < W; r++) {
            b(r, t) = b(r, t - 1) * 2 + 1;
        }
    }
    return b;
}

// mode 0: compute_root; 1: reindex + compute_root; 2: reindex + slide/fold(t,2).
Buffer<int> run(int mode) {
    Func f("f");
    Var x("x"), t("t");
    f(x, t) = 0;
    RDom r(0, W, "r");
    f(r, t) = select(t <= 0, r, likely(f(r, t - 1) * 2 + 1));
    if (mode >= 1) {
        Var u("u");
        f.update().reindex({u}, {(W - 1) - r}, {(W - 1) - u});  // reflect r
    }
    Func g("g");
    g(x, t) = f(x, t);
    if (mode == 2) {
        f.compute_at(g, t).store_root().fold_storage(t, 2);
    } else {
        f.compute_root();
    }
    return g.realize({W, T});
}

bool check(const Buffer<int> &got, const Buffer<int> &ref, const char *name) {
    for (int t = 0; t < T; t++) {
        for (int r = 0; r < W; r++) {
            if (got(r, t) != ref(r, t)) {
                printf("%s: mismatch at (%d,%d): %d vs %d\n", name, r, t,
                       got(r, t), ref(r, t));
                return false;
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    Buffer<int> ref = reference();
    if (!check(run(0), ref, "baseline")) {
        return 1;
    }
    if (!check(run(1), ref, "reindex")) {
        return 1;
    }
    if (!check(run(2), ref, "reindex+slide-fold")) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
