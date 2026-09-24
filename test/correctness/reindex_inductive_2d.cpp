#include "Halide.h"

#include <cstdio>

// Reindex the two RVars of a Func that is inductive in t and, at each step,
// gathers over a 2-D reduction domain. Skewing the (r.x, r.y) axes keeps the
// result unchanged, and -- because the inductive var t is fed to the race
// analysis as serial -- at least one skewed axis is proven parallel-safe and can
// be parallelized with no allow_race_conditions().

using namespace Halide;

namespace {

const int W = 5, T = 4;

Buffer<int> reference() {
    Buffer<int> ref(W, W, T);
    for (int a = 0; a < W; a++) {
        for (int b = 0; b < W; b++) {
            ref(a, b, 0) = a + b;
        }
    }
    for (int t = 1; t < T; t++) {
        for (int a = 0; a < W; a++) {
            for (int b = 0; b < W; b++) {
                ref(a, b, t) = ref(a, b, t - 1) + ref((a + 1) % W, b, t - 1) +
                               ref(a, (b + 1) % W, t - 1);
            }
        }
    }
    return ref;
}

// mode 0: no reindex; 1: reindex(skew r.x,r.y); 2: reindex then parallel a safe axis.
Buffer<int> run(int mode, bool *any_safe) {
    Func f("f");
    Var x("x"), y("y"), t("t");
    f(x, y, t) = 0;
    RDom r(0, W, 0, W, "r");
    f(r.x, r.y, t) = select(t <= 0, r.x + r.y,
                            likely(f(r.x, r.y, t - 1) + f((r.x + 1) % W, r.y, t - 1) +
                                   f(r.x, (r.y + 1) % W, t - 1)));
    if (mode >= 1) {
        Var u("u"), v("v");
        ReindexResult res = f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        *any_safe = res.parallel_safe[0] || res.parallel_safe[1];
        if (mode == 2) {
            for (size_t k = 0; k < res.vars.size(); k++) {
                if (res.parallel_safe[k]) {
                    f.update().parallel(res.vars[k]);
                }
            }
        }
    }
    Func g("g");
    g(x, y, t) = f(x, y, t);
    f.compute_root();
    return g.realize({W, W, T});
}

bool check(const Buffer<int> &got, const Buffer<int> &ref, const char *name) {
    for (int t = 0; t < T; t++) {
        for (int a = 0; a < W; a++) {
            for (int b = 0; b < W; b++) {
                if (got(a, b, t) != ref(a, b, t)) {
                    printf("%s: mismatch at (%d,%d,%d): %d vs %d\n", name, a, b, t,
                           got(a, b, t), ref(a, b, t));
                    return false;
                }
            }
        }
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    Buffer<int> ref = reference();
    bool any_safe = false;

    if (!check(run(0, &any_safe), ref, "baseline")) {
        return 1;
    }
    if (!check(run(1, &any_safe), ref, "reindex")) {
        return 1;
    }
    if (!any_safe) {
        printf("expected at least one parallel-safe axis (inductive t serial)\n");
        return 1;
    }
    if (!check(run(2, &any_safe), ref, "reindex+parallel")) {
        return 1;
    }

    printf("Success!\n");
    return 0;
}
