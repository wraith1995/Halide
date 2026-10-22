#include "Halide.h"
#include <cstdio>

// reorder() accepts a new order of the RVars of an update that isn't
// associative and commutative when every dependence of the update still
// points forward in it.

using namespace Halide;

namespace {

const int N = 24;

// f(x, y) reads f(x - 1, y) and f(x, y - 1).
Func wavefront(const char *name, RDom &r) {
    Func f(name);
    Var x("x"), y("y");
    f(x, y) = x + y;
    r = RDom(1, N - 1, 1, N - 1);
    f(r.x, r.y) = (f(r.x - 1, r.y) * 3 + f(r.x, r.y - 1)) % 1009;
    return f;
}

bool same(const Buffer<int> &a, const Buffer<int> &b, const char *what) {
    for (int y = 0; y < N; y++) {
        for (int x = 0; x < N; x++) {
            if (a(x, y) != b(x, y)) {
                printf("%s: mismatch at (%d, %d): %d vs %d\n", what, x, y, a(x, y), b(x, y));
                return false;
            }
        }
    }
    return true;
}

Internal::DimType dim_type(const Stage &s, const std::string &var) {
    for (const auto &d : s.get_schedule().dims()) {
        if (d.var == var || Internal::ends_with(d.var, "." + var)) {
            return d.dim_type;
        }
    }
    return Internal::DimType::PureVar;
}

}  // namespace

int main(int argc, char **argv) {
    RDom r;
    Buffer<int> expected = wavefront("f_serial", r).realize({N, N});

    // Both dependences point forward along x and along y, so either loop can
    // be outermost.
    {
        Func f = wavefront("f_swapped", r);
        f.update().reorder(r.y, r.x);
        if (!same(f.realize({N, N}), expected, "swapped")) {
            return 1;
        }
    }

    // The same through a split: a dependence along x either stays within a
    // tile or crosses into the next one, which runs later.
    {
        Func f = wavefront("f_tiled", r);
        RVar xo("xo"), xi("xi");
        f.update().split(r.x, xo, xi, 4).reorder(xi, r.y, xo);
        if (!same(f.realize({N, N}), expected, "tiled")) {
            return 1;
        }
    }

    // In wavefront order the position on a wavefront is a NestedPureRVar.
    // Moving it outside the loop over wavefronts runs the update in an order
    // that keeps its dependences, but the position now carries one, so it
    // becomes an ImpureRVar.
    {
        Func f = wavefront("f_wave", r);
        Var u("u"), v("v");
        ReindexResult res = f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        if (dim_type(f.update(), "u") != Internal::DimType::NestedPureRVar) {
            printf("expected u to be a NestedPureRVar after reindex\n");
            return 1;
        }
        f.update().reorder(res.vars[1], res.vars[0]);
        if (dim_type(f.update(), "u") != Internal::DimType::ImpureRVar) {
            printf("expected u to be an ImpureRVar after moving it outside v\n");
            return 1;
        }
        if (!same(f.realize({N, N}), expected, "wavefront reordered")) {
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
}
