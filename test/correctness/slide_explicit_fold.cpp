#include "Halide.h"
#include <stdio.h>
#include <string>

using namespace Halide;

// When slide() targets a dimension spread across a split, sliding window
// tells storage folding how wide the window is, and the producer folds to
// that. An explicit fold_storage of the same dimension can widen the fold,
// up to keeping the whole extent, but never narrow it below the window.

namespace {

// The size of a producer's allocation.
class AllocationSizeOf : public Internal::IRMutator {
    const std::string producer;
    using IRMutator::visit;
    Internal::Stmt visit(const Internal::Allocate *op) override {
        if (op->name == producer) {
            int64_t total = 1;
            for (const auto &e : op->extents) {
                total *= Internal::as_const_int(e).value_or(0);
            }
            size = (int)total;
        }
        return IRMutator::visit(op);
    }

public:
    int size = 0;
    AllocationSizeOf(std::string producer)
        : producer(std::move(producer)) {
    }
};

const int N = 16;

// The allocation of a producer that slides along a split dimension of its
// consumer, two values live at a time, with fold_storage(x, fold) when fold
// is positive. Returns -1 on a wrong result.
int allocation_with_fold(int fold) {
    Func f("f"), g("g");
    Var x("x"), xo("xo"), xi("xi");

    f(x) = x * x;
    g(x) = f(x) + f(clamp(x - 1, 0, N - 1));
    g.bound(x, 0, N);

    g.align_bounds(x, 2).split(x, xo, xi, 2).unroll(xi);
    f.store_root().compute_at(g, xi).slide(g, x);
    if (fold > 0) {
        f.fold_storage(x, fold);
    }

    AllocationSizeOf alloc(f.name());
    g.add_custom_lowering_pass(&alloc, nullptr);
    Buffer<int> out = g.realize({N});

    for (int i = 0; i < N; i++) {
        int p = std::max(i - 1, 0);
        if (out(i) != i * i + p * p) {
            printf("fold %d: out(%d) = %d instead of %d\n", fold, i, out(i), i * i + p * p);
            return -1;
        }
    }
    return alloc.size;
}

}  // namespace

int main(int argc, char **argv) {
    struct {
        int fold, expected;
    } cases[] = {
        {0, 2},   // the window
        {2, 2},   // an explicit fold as wide as the window
        {8, 8},   // wider
        {N, N},   // the whole extent: no folding
    };
    for (const auto &c : cases) {
        int size = allocation_with_fold(c.fold);
        if (size != c.expected) {
            printf("fold_storage(x, %d): allocation of %d instead of %d\n", c.fold, size, c.expected);
            return 1;
        }
    }
    printf("Success!\n");
    return 0;
}
