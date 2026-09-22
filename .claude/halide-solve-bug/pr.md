<!-- Title: Solve: treat min/max of booleans as And/Or in SolveForInterval -->

`SolveForInterval` in `src/Solve.cpp` handles `And`, `Or`, `Not`, and `Select`, but
not `Min` / `Max`. Boolean `min`/`max` — the `&&`/`||` encodings that
`and_condition_over_domain` and the simplifier both emit — were therefore not
treated as logical connectives, so `Not` did not distribute over them (De Morgan)
and `solve_for_outer_interval` / `solve_for_inner_interval` returned a wrong
interval. For example `solve_for_outer_interval(!min(x < 0, x < 2), "x")` returned
`[2, +inf)` instead of `[0, +inf)` (`min(x<0, x<2)` is `x<0`, so `!min(...)` is
`x>=0`).

This caused a silent miscompile via `trim_no_ops`, which builds a loop's no-op
condition by relaxing over inner loops with `and_condition_over_domain` (yielding
`min`/`max` of comparisons) and trims the loop using `solve_for_outer_interval`.
The wrong interval trimmed a serial loop to a single iteration and dropped work
— e.g. a wavefront/diamond-tiled recurrence with a `GuardWithIf` (`likely`) split
tail over a non-tile-multiple extent produced incorrect results under parallelism.

Fix: add `visit(Min)` / `visit(Max)` to `SolveForInterval`, rewriting a boolean
`min` to `And` and `max` to `Or` and re-solving — exactly as the existing
`visit(Select)` already does — so `Not` distributes correctly.

Regression checks for `!min`, `!max`, and the non-negated forms are added to
`test/correctness/solve.cpp`.

Validated locally on `main` @ `d33fd254f` (LLVM 21, macOS arm64): confirmed the
bug on unmodified `main` and its fix afterward; `ctest -L correctness` passes
471/471 (GPU tests skipped, no GPU target), `ctest -L 'error|warning'` passes
168/168, and the `solve` fuzzer runs clean.

Fixes #<!-- issue number -->

## Breaking changes

None. The change is confined to `Halide::Internal` (`SolveForInterval` in
`Solve.cpp`); there are no changes to the `Halide::` public API.

## Checklist

- [x] Tests added or updated (not required for docs, CI config, or typo fixes)
- [ ] Documentation updated (if public API changed) — N/A, `Halide::Internal` only
- [ ] Python bindings updated (if public API changed) — N/A, `Halide::Internal` only
- [ ] Benchmarks are included here if the change is intended to affect performance
      — N/A, correctness fix (no intended performance change)
- [x] Commits include AI attribution where applicable (see Code of Conduct)
