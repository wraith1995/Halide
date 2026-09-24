#include "Halide.h"
#include <cstdio>

// A select guarded by conditions on two different loops, like the boundary
// of a time-stepped stencil: t <= 0 || x <= 0 || x >= W - 1. The simplifier
// merges the first two into min(t, x) <= 0. Loop partitioning should still
// find the steady state of each loop from the part of the condition that
// depends on it: t >= 1 for the loop over t, and 1 <= x <= W - 2 for the loop
// over x, which has to run over the whole interior rather than up to t.

using namespace Halide;
using namespace Halide::Internal;

namespace {

const int T = 6;

// Inspects every store to func: whether its value has a select in it, and
// whether the loops around it have bounds that depend on the loop over t.
class InspectStores : public IRVisitor {
    using IRVisitor::visit;

    std::map<std::string, Expr> lets;
    std::vector<const For *> loops;

    void visit(const LetStmt *op) override {
        lets[op->name] = op->value;
        IRVisitor::visit(op);
    }

    bool depends_on_t(Expr e) {
        // Substitute the enclosing lets until only loop vars and params remain.
        for (int i = 0; i < 20; i++) {
            Expr next = substitute(lets, e);
            if (next.same_as(e)) {
                break;
            }
            e = next;
        }
        bool found = false;
        visit_with(e, [&](auto *self, const Variable *v) {
            if (starts_with(v->name, func + ".s0.t")) {
                found = true;
            }
        });
        return found;
    }

    void visit(const For *op) override {
        loops.push_back(op);
        IRVisitor::visit(op);
        loops.pop_back();
    }

    void visit(const Store *op) override {
        IRVisitor::visit(op);
        if (op->name != func) {
            return;
        }
        bool has_select = false;
        visit_with(op->value, [&](auto *self, const Select *) { has_select = true; });
        if (has_select || loops.empty()) {
            return;
        }
        steady_stores++;
        const For *inner = loops.back();
        if (!depends_on_t(inner->min) && !depends_on_t(inner->max)) {
            steady_stores_independent_of_t++;
        }
    }

public:
    std::string func;
    int steady_stores = 0, steady_stores_independent_of_t = 0;
};

class Check : public IRMutator {
public:
    using IRMutator::mutate;
    InspectStores inspect;
    Stmt mutate(const Stmt &s) override {
        s.accept(&inspect);
        return s;
    }
};

int run(bool vectorize, int W) {
    Buffer<float> in(W, T);
    for (int t = 0; t < T; t++) {
        for (int x = 0; x < W; x++) {
            in(x, t) = x + 100 * t;
        }
    }
    Func g("g");
    Var x("x"), t("t");
    g(x, t) = select(t <= 0 || x <= 0 || x >= W - 1, in(x, t) * 2, likely(in(x, t) + 1));
    g.bound(x, 0, W).bound(t, 0, T);
    if (vectorize) {
        // When W isn't a multiple of 8, the vectors shift inwards at the
        // end, which puts a likely on the clamp of the vector's base.
        g.vectorize(x, 8);
    }
    Check *check = new Check;
    check->inspect.func = g.name();
    g.add_custom_lowering_pass(check);
    g.compile_jit();
    Buffer<float> result = g.realize({W, T});
    for (int tt = 0; tt < T; tt++) {
        for (int xx = 0; xx < W; xx++) {
            float want = (tt <= 0 || xx <= 0 || xx >= W - 1) ? in(xx, tt) * 2 : in(xx, tt) + 1;
            if (result(xx, tt) != want) {
                printf("g(%d, %d) = %f instead of %f\n", xx, tt, result(xx, tt), want);
                return 1;
            }
        }
    }
    if (check->inspect.steady_stores_independent_of_t == 0) {
        printf("vectorize=%d, W=%d: no steady state over the interior of x "
               "(%d select-free stores, all bounded by t)\n",
               vectorize, W, check->inspect.steady_stores);
        return 1;
    }
    return 0;
}

}  // namespace

int main(int argc, char **argv) {
    // Scalar; vectorized over a multiple of the vector width; and vectorized
    // with a tail that shifts inwards.
    int failures = run(false, 100) + run(true, 96) + run(true, 100);
    if (failures) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
