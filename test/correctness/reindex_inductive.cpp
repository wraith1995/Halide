#include "Halide.h"

#include <cstdio>

// Reindex the RVar of an *inductive* Func. f is inductive in t (a pure var read
// at t-1) and gathers over the RVar r. reindex() reflects the r axis. Because
// reindex feeds the Func's inductive pure var (t) into the race analysis as
// serial, the reflected axis u is proven parallel-safe (a PureRVar) and can be
// parallelized with no allow_race_conditions(); without that, u would be
// classified unsafe. The result is unchanged throughout.

using namespace Halide;

namespace {

const int W = 6, T = 5;

Buffer<int> reference() {
    Buffer<int> ref(W, T);
    for (int r = 0; r < W; r++) {
        ref(r, 0) = r;
    }
    for (int t = 1; t < T; t++) {
        for (int r = 0; r < W; r++) {
            ref(r, t) = ref(r, t - 1) + ref((r + 1) % W, t - 1);
        }
    }
    return ref;
}

// mode 0: no reindex; 1: reindex(reflect r); 2: reindex then parallel(u).
Buffer<int> run(int mode, int *safe) {
    Func f("f");
    Var x("x"), t("t");
    f(x, t) = 0;
    RDom r(0, W, "r");
    f(r, t) = select(t <= 0, r, likely(f(r, t - 1) + f((r + 1) % W, t - 1)));
    if (mode >= 1) {
        Var u("u");
        // Reflect the r axis: u = W-1-r, inverse r = W-1-u.
        ReindexResult res = f.update().reindex({u}, {(W - 1) - r}, {(W - 1) - u});
        *safe = res.parallel_safe[0];
        if (mode == 2 && res.parallel_safe[0]) {
            f.update().parallel(res.vars[0]);
        }
    }
    // An inductive Func cannot be an output, so realize a trivial consumer.
    Func g("g");
    g(x, t) = f(x, t);
    f.compute_root();
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
    int safe = -1;

    if (!check(run(0, &safe), ref, "baseline")) {
        return 1;
    }
    if (!check(run(1, &safe), ref, "reindex")) {
        return 1;
    }
    if (safe != 1) {
        printf("expected u parallel-safe (inductive var t treated serial), "
               "got %d\n",
               safe);
        return 1;
    }
    if (!check(run(2, &safe), ref, "reindex+parallel")) {
        return 1;
    }

    printf("Success!\n");
    return 0;
}
