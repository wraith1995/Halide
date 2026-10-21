// An inductive Func whose step is a stencil over the previous slice, like an
// explicit time-stepped PDE: f(x, t) reads f(x - 1, t - 1), f(x, t - 1) and
// f(x + 1, t - 1). Only t is inductive; x is a neighbourhood within a slice.
//
// When a consumer only wants the final slice, the storage for t folds to the
// two slices the recurrence needs, over f's own loop, with nothing in the
// schedule asking for it. Computing f per tile of the consumer then gives
// overlapped temporal tiling: each tile computes the cone of the previous
// slices it depends on, in a two-slice window.

#include "Halide.h"
#include <cstdio>
#include <vector>

using namespace Halide;
using namespace Halide::Internal;

namespace {

const int W = 200, T = 24;

// Records the extent along t of the allocation of one Func.
class RecordSlices : public IRMutator {
    using IRMutator::visit;
    Stmt visit(const Allocate *op) override {
        if (op->name == func) {
            auto e = as_const_int(op->extents.back());
            *slices = e ? (int)*e : -1;
        }
        return IRMutator::visit(op);
    }

public:
    std::string func;
    int *slices;
    RecordSlices(std::string f, int *s)
        : func(std::move(f)), slices(s) {
    }
};

std::vector<std::vector<int>> reference() {
    std::vector<std::vector<int>> ref(T, std::vector<int>(W));
    for (int x = 0; x < W; x++) {
        ref[0][x] = (x * 7 + 3) % 101;
    }
    for (int t = 1; t < T; t++) {
        for (int x = 0; x < W; x++) {
            ref[t][x] = (x == 0 || x == W - 1) ? ref[0][x] : (ref[t - 1][x - 1] + 2 * ref[t - 1][x] + ref[t - 1][x + 1] + 1) % 101;
        }
    }
    return ref;
}

Func stencil(const std::string &name) {
    Func f(Int(32), name);
    Var x("x"), t("t");
    f(x, t) = select(t <= 0 || x <= 0 || x >= W - 1, (x * 7 + 3) % 101,
                     likely((f(x - 1, t - 1) + 2 * f(x, t - 1) + f(x + 1, t - 1) + 1) % 101));
    return f;
}

// The consumer reads the last `last` slices, summed. Returns the number of
// slices allocated, or -2 on a wrong answer.
int run(int schedule, int last) {
    const auto ref = reference();
    std::string name = "f_" + std::to_string(schedule) + "_" + std::to_string(last);
    Func f = stencil(name);
    Var x("x"), xo("xo"), xi("xi");
    Func out("out");
    Expr e = 0;
    for (int k = 0; k < last; k++) {
        e += f(x, T - 1 - k);
    }
    out(x) = e;
    out.bound(x, 0, W);
    if (schedule == 0) {
        f.compute_root();
    } else {
        out.split(x, xo, xi, 32);
        f.compute_at(out, xo);
        if (schedule == 2) {
            out.parallel(xo);
            f.vectorize(x, 8);
        }
        if (schedule == 3) {
            // Store f once for all tiles, so sliding window computes only the
            // new part of f for each tile, using what earlier tiles computed.
            // Those parts of earlier slices have to survive.
            f.store_root();
        }
    }

    int slices = -1;
    out.add_custom_lowering_pass(new RecordSlices(name, &slices));
    Buffer<int> result = out.realize({W});
    for (int x = 0; x < W; x++) {
        int want = 0;
        for (int k = 0; k < last; k++) {
            want += ref[T - 1 - k][x];
        }
        if (result(x) != want) {
            printf("schedule %d, last %d: out(%d) = %d instead of %d\n", schedule, last, x, result(x), want);
            return -2;
        }
    }
    return slices;
}

// Records the extent along x of the allocations of a Func (each of its values,
// if it has several).
class RecordWidth : public IRMutator {
    using IRMutator::visit;
    Stmt visit(const Allocate *op) override {
        if (op->name == func || starts_with(op->name, func + ".")) {
            auto e = as_const_int(op->extents[0]);
            *width = e ? (int)*e : -1;
        }
        return IRMutator::visit(op);
    }

public:
    std::string func;
    int *width;
    RecordWidth(std::string f, int *w)
        : func(std::move(f)), width(w) {
    }
};

// A stencil with no boundary along x widens by its reach at every step back.
// With two values, each widens it from the region required, not one after the
// other. Returns the widths allocated for one value and for two, or -2 on a
// wrong answer.
std::pair<int, int> widths_for_one_and_two_values() {
    Var x("x"), t("t");
    Func one(Int(32), "f_one");
    one(x, t) = select(t <= 0, x, likely(one(x - 1, t - 1) + one(x + 1, t - 1)));
    Func two({Int(32), Int(32)}, "f_two");
    two(x, t) = Tuple(select(t <= 0, x, likely(two(x - 1, t - 1)[0] + two(x + 1, t - 1)[0])),
                      select(t <= 0, 2 * x, likely(two(x - 1, t - 1)[1] + two(x + 1, t - 1)[1])));
    Func out_one("out_one"), out_two("out_two");
    out_one(x) = one(x, 6);
    out_two(x) = two(x, 6)[0] + two(x, 6)[1];
    out_one.bound(x, 0, 16);
    out_two.bound(x, 0, 16);
    one.compute_root();
    two.compute_root();
    int w1 = -1, w2 = -1;
    out_one.add_custom_lowering_pass(new RecordWidth("f_one", &w1));
    out_two.add_custom_lowering_pass(new RecordWidth("f_two", &w2));
    Buffer<int> r1 = out_one.realize({16});
    Buffer<int> r2 = out_two.realize({16});
    for (int i = 0; i < 16; i++) {
        if (r2(i) != 3 * r1(i)) {
            printf("two values: out(%d) = %d instead of %d\n", i, r2(i), 3 * r1(i));
            return {-2, -2};
        }
    }
    return {w1, w2};
}

// Every slice is read, so nothing can fold.
int run_all_slices() {
    const auto ref = reference();
    Func f = stencil("f_all");
    Var x("x"), t("t");
    Func out("out_all");
    out(x, t) = f(x, t);
    out.bound(x, 0, W).bound(t, 0, T);
    f.compute_root();
    int slices = -1;
    out.add_custom_lowering_pass(new RecordSlices("f_all", &slices));
    Buffer<int> result = out.realize({W, T});
    for (int t = 0; t < T; t++) {
        for (int x = 0; x < W; x++) {
            if (result(x, t) != ref[t][x]) {
                printf("all slices: out(%d, %d) = %d instead of %d\n", x, t, result(x, t), ref[t][x]);
                return -2;
            }
        }
    }
    return slices;
}

}  // namespace

int main(int argc, char **argv) {
    struct Case {
        int schedule, last, slices;
    };
    // The step reaches one slice back, so the window keeps two. A consumer
    // reading the last two slices fits in that window; one reading the last
    // three needs a window of four. A Func slid along by a loop outside it
    // (schedule 3) doesn't fold in place: 0 means every slice is kept.
    const Case cases[] = {
        {0, 1, 2},
        {1, 1, 2},
        {2, 1, 2},
        {0, 2, 2},
        {1, 3, 4},
        {3, 1, 0},
    };
    for (const Case &c : cases) {
        int slices = run(c.schedule, c.last);
        if (slices == -2) {
            return 1;
        }
        if (c.slices == 0 ? (slices >= 0 && slices < T) : slices != c.slices) {
            printf("schedule %d, last %d: expected %d slices allocated, got %d\n",
                   c.schedule, c.last, c.slices, slices);
            return 1;
        }
    }

    auto [w1, w2] = widths_for_one_and_two_values();
    if (w1 == -2) {
        return 1;
    }
    if (w2 != w1) {
        printf("one value allocates %d along x, but two allocate %d\n", w1, w2);
        return 1;
    }

    int slices = run_all_slices();
    if (slices == -2) {
        return 1;
    }
    if (slices < T) {
        printf("every slice is read, but only %d were allocated\n", slices);
        return 1;
    }

    printf("Success!\n");
    return 0;
}
