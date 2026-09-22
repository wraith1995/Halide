<!-- Title: solve_for_outer_interval / solve_for_inner_interval mishandle min/max of booleans -->

### Description

`SolveForInterval` in `src/Solve.cpp` implements `visit(And)`, `visit(Or)`, `visit(Not)`, and `visit(Select)`, but not `visit(Min)` / `visit(Max)`. Boolean `min`/`max` — the `&&`/`||` encodings that `and_condition_over_domain` and the simplifier both produce — are therefore not recognized as logical connectives, so `Not` does not distribute over them (De Morgan), and `solve_for_outer_interval` / `solve_for_inner_interval` return a wrong interval.

Concretely, `min(x<0, x<2)` is equivalent to `x<0`, so `!min(x<0, x<2)` is `x>=0` and should solve to `[0, +inf)`. Instead it solves to `[2, +inf)` (as if `!min(a,b)` were `!a && !b` instead of `!a || !b`).

This is not just a precision issue in a helper: it causes a silent miscompile via `trim_no_ops`, which builds a loop's no-op condition by relaxing over inner loops with `and_condition_over_domain` (producing `min`/`max` of comparisons) and then trims the loop with `solve_for_outer_interval(!is_no_op, var)`. The wrong interval can trim a **serial** loop down to a single iteration and drop real work — e.g. a wavefront/diamond-tiled recurrence with a `GuardWithIf` (`likely`) split tail over a non-tile-multiple extent produces incorrect output under parallelism.

### Reproducing case

```cpp
#include "Halide.h"
#include "Solve.h"  // Halide::Internal

#include <iostream>

using namespace Halide;
using namespace Halide::Internal;

int main() {
    Expr x = Variable::make(Int(32), "x");

    // min(x < 0, x < 2) is equivalent to (x < 0), so !min(...) is (x >= 0),
    // which must solve to [0, +inf).
    Interval i = solve_for_outer_interval(!min(x < 0, x < 2), "x");
    std::cout << "got [" << simplify(i.min) << ", " << simplify(i.max) << "]\n";
    // Expected: [0, +inf)
    // Actual:   [2, +inf)

    // The equivalent forms both solve correctly, confirming the discrepancy:
    Interval j = solve_for_outer_interval(x >= 0, "x");       // [0, +inf)
    Interval k = solve_for_outer_interval(!(x < 0), "x");     // [0, +inf)
    return 0;
}
```

### How did you get Halide?

Built from source

### Halide version

22.0.0

### Halide commit (if known)

d33fd254f

### Target

arm-64-osx (host; the bug is target-independent)

### Operating system

macOS (arm64)

### Additional context

- `simplify(min(x<0, x<2))` also does not fold to `x<0`, so the `min` survives to `solve`.
- End-to-end, the same defect surfaces as wrong output (not a crash) from a valid pipeline; the parallelization in the diamond-tiled case is provably race-free, so it is a miscompile rather than a race.
- Fix: handle `Min`/`Max` of boolean operands in `SolveForInterval` as `And`/`Or` (mirroring the existing `visit(Select)`) so that `Not` distributes correctly. PR to follow.
