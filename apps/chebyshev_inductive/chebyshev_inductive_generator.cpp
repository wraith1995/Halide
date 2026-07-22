// Chebyshev semi-iteration for an SPD system A x = b.
// Contains a version that uses inductive functions, and a version that does not.
//
// The Chebyshev iterate obeys a single-sequence three-term recurrence
//
//     x_{k+1} = (1 + w_k) x_k - w_k x_{k-1} + a_k (b - A x_k)
//
// whose coefficients a_k, w_k depend only on the spectral bounds of A (there are no
// inner products). So a step is one mat-vec plus a bounded-lag, same-component
// combination -- exactly the shape an inductive Func can express and fold.
//
// Unlike a 1D IIR filter (see apps/iir_cascade), the recurrence here is a *mat-vec*
// recurrence: column k reads the previous iterate at *every* component. That cannot
// be written as a reduction-domain scan (the self-reference would put a reduction
// variable where a pure variable must be), so there is no "rolled" RDom form of the
// iteration. The non-inductive version therefore has to UNROLL the fixed iteration
// count and materialise the intermediate iterates, whereas the inductive version
// keeps the whole iteration in one pipeline and folds its storage to O(1) columns.

#include "Halide.h"

#include <string>
#include <vector>

using namespace Halide;

class ChebyshevInductive : public Generator<ChebyshevInductive> {
public:
    Input<Buffer<double, 2>> A{"A"};          // n x n SPD matrix (column-major)
    Input<Buffer<double, 1>> b{"b"};          // right-hand side, length n
    Input<Buffer<double, 1>> alpha{"alpha"};  // per-iteration coefficient a_k, length M
    Input<Buffer<double, 1>> omega{"omega"};  // per-iteration coefficient w_k, length M

    GeneratorParam<int> M{"M", 60};                // number of Chebyshev iterations
    GeneratorParam<bool> inductive{"inductive", true};

    Output<Buffer<double, 1>> x{"x"};  // solution iterate x_M, length n

    void generate() {
        Var t("t"), k("k");
        Expr n = A.dim(0).extent();
        RDom j(0, n, "j");  // the single reduction: the mat-vec over components

        if (inductive) {
            // ---- inductive: the whole iteration is ONE folded pipeline ----
            Func X(Float(64), "X");
            // Pure definition: base case (k <= 0) is x0 = 0; accumulator starts at 0.
            X(t, k) = cast<double>(0);

            Expr km1 = max(0, k - 1), km2 = max(0, k - 2);
            Expr once = (Expr(1.0) + omega(km1)) * X(t, km1) - omega(km1) * X(t, km2) +
                        alpha(km1) * b(t);
            // One update; every self-reference (incl. the accumulator X(t,k)) sits
            // inside ONE select. Once-per-column terms are gated by cast(j == 0); the
            // mat-vec of the previous column X(j, k-1) is accumulated over all j.
            X(t, k) = select(k <= 0, cast<double>(0),
                             X(t, k) + cast<double>(j == 0) * once -
                                 alpha(km1) * A(t, j) * X(j, km1));

            // Endpoint-extract consumer: keep only column M (= x_M), so the OUTPUT is
            // O(n). The iteration index rk is the OUTERMOST loop (the mat-vec couples
            // all t within a column), which is also the sweep that lets X fold.
            RDom rk(0, (int)M + 1, "rk");
            x(t) = cast<double>(0);
            x(t) += select(rk == (int)M, X(t, rk), cast<double>(0));
            x.update(0).reorder(t, rk);
            X.compute_at(x, rk).store_root().fold_storage(k, 3);
            X.update(0).vectorize(t, 4);
        } else {
            // ---- non-inductive: unroll the fixed iteration; mat-vec via an RDom ----
            // There is no reduction-domain form of the mat-vec recurrence, so we build
            // the iterates x_0, x_1, ..., x_M as a chain of pure Funcs and materialise
            // them (compute_root) -- the intermediate iterates are written to memory,
            // which is exactly the I/O the inductive version's folding avoids.
            std::vector<Func> xs((int)M + 1);
            xs[0] = Func(Float(64), "x0");
            xs[0](t) = cast<double>(0);  // x_0 = 0
            for (int step = 0; step < (int)M; step++) {
                Func xn(Float(64), "x" + std::to_string(step + 1));
                Expr Ax = sum(A(t, j) * xs[step](j));  // mat-vec via a reduction domain
                Expr xk = xs[step](t);
                Expr xkm1 = (step == 0) ? cast<double>(0) : xs[step - 1](t);
                xn(t) = (Expr(1.0) + omega(step)) * xk - omega(step) * xkm1 +
                        alpha(step) * (b(t) - Ax);
                xs[step + 1] = xn;
            }
            x(t) = xs[(int)M](t);
            for (int step = 1; step <= (int)M; step++) {
                xs[step].compute_root().vectorize(t, 4);
            }
        }

        // Output is length n.
        x.dim(0).set_bounds(0, A.dim(0).extent());
    }
};

HALIDE_REGISTER_GENERATOR(ChebyshevInductive, chebyshev_inductive)
