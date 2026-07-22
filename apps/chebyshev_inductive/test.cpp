// Runs the inductive and non-inductive Chebyshev solvers, checks they agree with
// each other and with a plain-C++ reference, and reports timing.

#include <cmath>
#include <cstdint>
#include <cstdio>
#include <vector>

#include "HalideBuffer.h"
#include "HalideRuntime.h"

#include "chebyshev_inductive.h"
#include "chebyshev_inductive_noninductive.h"

#include "halide_benchmark.h"

using Halide::Runtime::Buffer;
using Halide::Tools::benchmark;

namespace {

// Must match the generator's M GeneratorParam.
constexpr int M = 60;

void host_matvec(const std::vector<double> &A, int n, const std::vector<double> &x,
                 std::vector<double> &y) {
    for (int i = 0; i < n; i++) y[i] = 0.0;
    for (int j = 0; j < n; j++)
        for (int i = 0; i < n; i++) y[i] += A[(size_t)j * n + i] * x[j];
}

// Deterministic SPD system: A = M^T M + n I (so lambda_min >= n), b = A x*.
void make_spd(int n, std::vector<double> &A, std::vector<double> &b, std::vector<double> &x_exact) {
    uint64_t s = 1;
    auto rnd = [&] {
        s ^= s << 13;
        s ^= s >> 7;
        s ^= s << 17;
        return (double)(s >> 11) / (double)(1ULL << 53) * 2.0 - 1.0;
    };
    std::vector<double> Mm((size_t)n * n);
    for (auto &v : Mm) v = rnd();
    A.assign((size_t)n * n, 0.0);
    for (int i = 0; i < n; i++)
        for (int j = 0; j < n; j++) {
            double acc = 0.0;
            for (int kk = 0; kk < n; kk++) acc += Mm[(size_t)i * n + kk] * Mm[(size_t)j * n + kk];
            A[(size_t)j * n + i] = acc + (i == j ? (double)n : 0.0);
        }
    x_exact.assign(n, 0.0);
    for (int i = 0; i < n; i++) x_exact[i] = rnd();
    b.assign(n, 0.0);
    host_matvec(A, n, x_exact, b);
}

double lambda_max_est(const std::vector<double> &A, int n, int iters = 60) {
    std::vector<double> v(n, 1.0), w(n);
    double lam = 0.0;
    for (int it = 0; it < iters; it++) {
        host_matvec(A, n, v, w);
        double nrm = 0.0;
        for (int i = 0; i < n; i++) nrm += w[i] * w[i];
        nrm = std::sqrt(nrm);
        if (nrm == 0) break;
        for (int i = 0; i < n; i++) v[i] = w[i] / nrm;
        lam = nrm;
    }
    return lam;
}

// Three-term Chebyshev coefficients from the spectral bounds.
void make_coeffs(double lmin, double lmax, int m, std::vector<double> &alpha,
                 std::vector<double> &omega) {
    double d = 0.5 * (lmax + lmin), c = 0.5 * (lmax - lmin);
    alpha.assign(m, 0.0);
    omega.assign(m, 0.0);
    double a_prev = 0.0;
    for (int k = 0; k < m; k++) {
        if (k == 0) {
            alpha[k] = 1.0 / d;
            omega[k] = 0.0;
        } else {
            double beta = (c * a_prev * 0.5) * (c * a_prev * 0.5);
            alpha[k] = 1.0 / (d - beta / a_prev);
            omega[k] = alpha[k] * beta / a_prev;
        }
        a_prev = alpha[k];
    }
}

// Plain-C++ ground truth (three-term form).
std::vector<double> host_chebyshev(const std::vector<double> &A, int n, const std::vector<double> &b,
                                   const std::vector<double> &alpha, const std::vector<double> &omega,
                                   int m) {
    std::vector<double> xp(n, 0.0), xc(n, 0.0), xn(n), Ax(n);
    for (int k = 0; k < m; k++) {
        host_matvec(A, n, xc, Ax);
        double a = alpha[k], w = omega[k];
        for (int i = 0; i < n; i++) xn[i] = (1.0 + w) * xc[i] - w * xp[i] + a * (b[i] - Ax[i]);
        xp = xc;
        xc = xn;
    }
    return xc;
}

double rel_error(const std::vector<double> &x, const std::vector<double> &y) {
    double e = 0, nn = 0;
    for (size_t i = 0; i < x.size(); i++) {
        double dd = x[i] - y[i];
        e += dd * dd;
        nn += y[i] * y[i];
    }
    return std::sqrt(e / (nn > 0 ? nn : 1.0));
}

}  // namespace

int main(int argc, char **argv) {
    const int n = 256;

    std::vector<double> Ah, bh, x_exact;
    make_spd(n, Ah, bh, x_exact);
    double lmax = lambda_max_est(Ah, n) * 1.02;
    double lmin = (double)n * 0.98;  // A = M^T M + n I => lambda_min >= n
    std::vector<double> alpha_h, omega_h;
    make_coeffs(lmin, lmax, M, alpha_h, omega_h);
    std::vector<double> ref = host_chebyshev(Ah, n, bh, alpha_h, omega_h, M);

    // Halide buffers. A is column-major (dim0 = row, fastest), matching make_spd.
    Buffer<double> A(Ah.data(), n, n);
    Buffer<double> b(bh.data(), n);
    Buffer<double> alpha(alpha_h.data(), M);
    Buffer<double> omega(omega_h.data(), M);
    Buffer<double> x_inductive(n), x_noninductive(n);

    double t_inductive = benchmark([&]() { chebyshev_inductive(A, b, alpha, omega, x_inductive); });
    printf("inductive time:     %gms (1 pipeline, storage folds to 3 columns)\n", t_inductive * 1e3);

    double t_noninductive =
        benchmark([&]() { chebyshev_inductive_noninductive(A, b, alpha, omega, x_noninductive); });
    printf("non-inductive time: %gms (%d iterations unrolled, iterates materialised)\n",
           t_noninductive * 1e3, M);

    std::vector<double> xi(n), xn(n);
    for (int i = 0; i < n; i++) {
        xi[i] = x_inductive(i);
        xn[i] = x_noninductive(i);
    }

    double diff = rel_error(xi, xn);
    double err_i = rel_error(xi, x_exact), err_n = rel_error(xn, x_exact);
    double ref_i = rel_error(xi, ref), ref_n = rel_error(xn, ref);
    printf("inductive vs non-inductive: %g\n", diff);
    printf("error vs exact:   inductive %g   non-inductive %g\n", err_i, err_n);
    printf("error vs C++ ref: inductive %g   non-inductive %g\n", ref_i, ref_n);

    if (diff > 1e-9 || err_i > 1e-6 || err_n > 1e-6 || ref_i > 1e-9 || ref_n > 1e-9) {
        printf("Chebyshev solvers disagree or did not converge!\n");
        return 1;
    }

    printf("Success!\n");
    return 0;
}
