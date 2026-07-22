# chebyshev_inductive

Chebyshev semi-iteration for an SPD system `A x = b`, written two ways -- **with**
inductively defined functions and **without** -- in the style of `apps/iir_cascade`.
It is a worked example of using an inductive Func for an *iterative solver* recurrence.

## The recurrence

Chebyshev semi-iteration has a single-sequence three-term form

```
x_{k+1} = (1 + w_k) x_k - w_k x_{k-1} + a_k (b - A x_k)
```

The coefficients `a_k`, `w_k` come only from the spectral bounds `[lmin, lmax]` (no
inner products), so a step is exactly **one mat-vec** plus a bounded-lag, same-
component combination:

* one reduction per step (the mat-vec) -> fits an inductive update's single RDom;
* bounded lag (2) -> folds to O(1) columns of storage;
* the mat-vec `A x_k` is a self-reference at a *shifted* iteration index, read at
  every component.

## Inductive vs. non-inductive

The generator (`chebyshev_inductive_generator.cpp`) has a `GeneratorParam<bool>
inductive`, exactly like `iir_cascade`:

* **inductive = true** -- the whole iteration is one inductive Func; the endpoint is
  extracted with the iteration index as the outermost loop, and `fold_storage(k, 3)`
  keeps only three live columns. One pipeline, O(n) working memory.

* **inductive = false** -- there is a crucial difference from a 1D IIR filter:
  the Chebyshev iterate is a **mat-vec recurrence** (column `k` reads the previous
  iterate at *every* component), which **cannot** be written as a reduction-domain
  scan -- the self-reference would put a reduction variable where a pure variable must
  be (*"recursive references ... must contain the same pure variables in the same
  places"*), and routing around it needs a second reduction domain (*"Multiple
  reduction domains"*). So there is no rolled RDom form of the iteration; the non-
  inductive version has to **unroll** the fixed iteration count and materialise the
  intermediate iterates.

`test.cpp` runs both, checks they agree with each other and with a plain-C++
reference, and reports timing. As with `iir_cascade`, the inductive version's win is
reduced I/O -- it does not write the whole trajectory to memory -- which shows up on
large problems and on GPU rather than on small CPU cases.

## Build & run

With CMake (from the apps build):

```
cmake --build . --target chebyshev_inductive_test
ctest -R chebyshev_inductive_test
```

or with the Makefile:

```
make test
```
