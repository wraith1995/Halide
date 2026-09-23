#include "Halide.h"

#include <cstdio>

// reindex()'s parallel-safety classification (via Halide's own race analysis)
// must be discriminating across the spectrum:
//   A) a 2-D recurrence skewed to a wavefront -> exactly one safe axis;
//   B) a scalar reduction that accumulates into one location -> no safe axis;
//   C) an independent scatter under a permutation -> both axes safe.
// The safe axes are PureRVars, so parallelizing them needs no
// allow_race_conditions().
//
// Note (C uses a transpose, not a skew, deliberately): for a *skewed*
// independent scatter, can_parallelize_rvar()'s simplifier cannot always
// falsify the multi-variable write-write hazard, so it conservatively reports
// only one axis safe. That is safe (never a miscompile), just imprecise; a
// permutation keeps the hazard falsifiable so both axes are proven safe.

using namespace Halide;

namespace {

bool expect(const std::vector<bool> &got, bool a, bool b, const char *name) {
    if (got.size() == 2 && got[0] == a && got[1] == b) {
        return true;
    }
    printf("%s: expected {%d,%d}, got {%d,%d}\n", name, (int)a, (int)b,
           (int)got[0], (int)got[1]);
    return false;
}

}  // namespace

int main(int argc, char **argv) {
    const int W = 8, H = 8;
    Var x("x"), y("y"), u("u"), v("v");

    // A) recurrence: only u (anti-diagonal position) is safe after the skew.
    {
        Func in("ina"), f("fa");
        in(x, y) = x + y;
        f(x, y) = in(x, y);
        RDom r(1, W - 1, 1, H - 1, "r");
        f(r.x, r.y) = f(r.x - 1, r.y) + f(r.x, r.y - 1) + in(r.x, r.y);
        ReindexResult res =
            f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        if (!expect(res.parallel_safe, true, false, "A (recurrence)")) {
            return 1;
        }
    }

    // B) scalar accumulation into f(): every iteration touches the same
    // location, so neither skewed axis is safe.
    {
        Func in("inb"), f("fb");
        in(x, y) = x * 10 + y;
        f() = 0;
        RDom r(0, W, 0, H, "r");
        f() += in(r.x, r.y);
        ReindexResult res =
            f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        if (!expect(res.parallel_safe, false, false, "B (scalar sum)")) {
            return 1;
        }
    }

    // C) independent scatter under a transpose: both axes are safe and can be
    // parallelized together with no race assertion.
    {
        Func in("inc"), f("fc");
        in(x, y) = x * 10 + y;
        f(x, y) = 0;
        RDom r(0, W, 0, H, "r");
        f(r.x, r.y) = in(r.x, r.y);
        // u = r.y, v = r.x ; inverse r.x = v, r.y = u.
        ReindexResult res = f.update().reindex({u, v}, {r.y, r.x}, {v, u});
        if (!expect(res.parallel_safe, true, true, "C (scatter transpose)")) {
            return 1;
        }
        f.update().parallel(res.vars[0]).parallel(res.vars[1]);
        Buffer<int> out = f.realize({W, H});
        for (int yy = 0; yy < H; yy++) {
            for (int xx = 0; xx < W; xx++) {
                const int e = xx * 10 + yy;
                if (out(xx, yy) != e) {
                    printf("C: value wrong at (%d,%d): %d vs %d\n", xx, yy,
                           out(xx, yy), e);
                    return 1;
                }
            }
        }
    }

    printf("Success!\n");
    return 0;
}
