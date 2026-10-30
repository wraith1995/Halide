// Allen-Cahn phase-field dynamics on a 1-D grid (f = eps*laplacian(y) + y -
// y^3), integrated by Adams-Bashforth 2 from a classical RK4 start, with an
// observer reading the mean order parameter <y> of every slice. The Halide
// forms:
//   A. inductive FOLDED   (y is inductive in n; its history folds to a 2-slice
//                          window; the observer is folded into the step)
//   B. inductive UNFOLDED (the same, with the history pinned to the full T extent)
//   C. non-inductive      (an RDom scan materializes the full O(D*B*T)
//                          trajectory; the observer is a second pass over it)
// and two baselines:
//   D. Boost.odeint's adams_bashforth<2> with a runge_kutta4 start and its
//      observer, and a fused C++ loop: the AB2 update, the right-hand side
//      of the new slice, and the slice's sum in one pass per step.
// Everything timed is float, baselines included. The batch is B independent
// trajectories, one per thread when B > 1, for the Halide forms and the
// baselines alike: the baselines' trajectories run as tasks on the Halide
// runtime's thread pool, the same pool the Halide forms use.
//
// With a fourth argument `rk4`, the same problem is integrated by classical
// RK4 at the same fixed step instead, a method SUNDIALS also provides (its
// integrators have no Adams-Bashforth): the textbook C++ loop, Boost.odeint's
// runge_kutta4, SUNDIALS ARKODE (when built WITH_SUNDIALS), and an inductive
// Halide form that computes a whole step in one pass.
//
// usage: ode_observer_sparse_fused_test [D] [B] [T] [rk4]

#include "Halide.h"

#include "../support/bench_harness.h"
#include "../support/jit_support.h"
#include <boost/numeric/odeint.hpp>
#ifdef WITH_SUNDIALS
#include <arkode/arkode_erkstep.h>
#include <nvector/nvector_serial.h>
#endif
#include <cmath>
#include <cstdio>
#include <cstring>
#include <functional>
#include <map>
#include <vector>
using namespace Halide;

namespace {

typedef int (*par_for_t)(void *, halide_task_t, int, int, uint8_t *);

// Runs body(0) .. body(n - 1) as tasks of the Halide runtime's thread pool,
// which exists once a pipeline has been JIT-compiled.
void pool_for(par_for_t par_for, int n, const std::function<void(int)> &body) {
    par_for(
        nullptr, [](void *, int i, uint8_t *closure) {
            (*(const std::function<void(int)> *)closure)(i);
            return 0;
        },
        0, n, (uint8_t *)&body);
}

// The same problem, observer and batch, integrated by classical RK4 with a
// fixed step h: a method Halide, Boost.odeint and SUNDIALS can all run as
// written, so every form does the same arithmetic. The textbook C++ loop
// (four right-hand sides and the stage sums, a pass over the slice each) is
// the reference the others are checked against.
int run_rk4(int D, int B, int T, float h, float eps, const Buffer<float> &y0) {
    auto rhs_at = [=](float lm, float c, float lp) { return eps * (lm - 2.f * c + lp) + c - c * c * c; };
    auto rhs = [=](const float *__restrict__ y, float *__restrict__ f) {
        f[0] = rhs_at(y[0], y[0], y[1]);
        for (int i = 1; i < D - 1; i++)
            f[i] = rhs_at(y[i - 1], y[i], y[i + 1]);
        f[D - 1] = rhs_at(y[D - 2], y[D - 1], y[D - 1]);
    };
    auto energy = [=](const float *y) {
        float e = 0;
        for (int i = 0; i < D; i++)
            e += y[i];
        return e / D;
    };

    // Halide: inductive in the step, one step per slice. The stages are
    // recomputed inline from slice n - 1 at offsets -4..4 (k1 at -3..3, k2
    // at -2..2, k3 at -1..1, k4 at 0), so a step reads one slice and writes
    // one, for 16 right-hand sides per element instead of 4. Each stage
    // value at an offset is one shared Expr. At the ends a stage reads its
    // own value in place of the missing neighbour, as rhs() does. The
    // observer is folded into the step, as in the AB2 form.
    Var d("d"), b("b"), n("n");
    Func y({Float(32), Float(32)}, 3, "y_rk4"), E("E_rk4");
    y(d, b, n) = Tuple(undef<float>(), undef<float>());
    RDom r(0, D, "r");
    Expr i = r.x;
    auto stencil = [&](Expr left, Expr center, Expr right) {
        return eps * (left - 2.0f * center + right) + center - center * center * center;
    };
    using Stage = std::map<int, Expr>;
    auto k = [&](Stage &P, int o) {
        Expr c = P[o];
        return stencil(select(i + o <= 0, c, P[o - 1]), c, select(i + o >= D - 1, c, P[o + 1]));
    };
    Stage Y, P2, P3, P4, K1, K2, K3;
    for (int o = -4; o <= 4; o++)
        Y[o] = y(clamp(i + o, 0, D - 1), b, n - 1)[0];
    for (int o = -3; o <= 3; o++) {
        K1[o] = k(Y, o);
        P2[o] = Y[o] + (h / 2) * K1[o];
    }
    for (int o = -2; o <= 2; o++) {
        K2[o] = k(P2, o);
        P3[o] = Y[o] + (h / 2) * K2[o];
    }
    for (int o = -1; o <= 1; o++) {
        K3[o] = k(P3, o);
        P4[o] = Y[o] + h * K3[o];
    }
    Expr next = Y[0] + (h / 6) * (K1[0] + 2.0f * K2[0] + 2.0f * K3[0] + k(P4, 0));
    Expr val = select(n <= 0, y0(i, b), likely(next));
    Expr carried = select(i >= 16, y(max(i - 16, 0), b, n)[1], cast<float>(0));
    y(i, b, n) = Tuple(val, carried + val);
    RDom rl(0, 16, "rl");
    E(b, n) = cast<float>(0);
    E(b, n) += y(D - 16 + rl, b, n)[1] * (1.0f / D);
    E.compute_root();
    E.update(0).reorder(rl, n, b).atomic().vectorize(rl).parallel(b);
    E.reorder(n, b).vectorize(n, 16).parallel(b);
    E.bound(b, 0, B).bound(n, 0, T);
    E.output_buffer().dim(0).set_stride(T).dim(1).set_stride(1);
    y.compute_at(E, n).store_at(E, b).fold_storage(n, 2);
    y.update(0).allow_race_conditions().vectorize(r.x, 16);
    E.compile_jit();

    par_for_t par_for = (par_for_t)Internal::JITSharedRuntime::find_symbol(
        get_jit_target_from_environment(), "halide_do_par_for");
    if (!par_for) {
        fprintf(stderr, "halide_do_par_for not found in the JIT runtime\n");
        return 1;
    }

    std::vector<float> ec((size_t)B * T), eb((size_t)B * T);
    hb::Stats s_cpp = hb::bench([&]() {
        pool_for(par_for, B, [&](int bb) {
            std::vector<float> yv(D), k1(D), k2(D), k3(D), k4(D), t(D);
            for (int j = 0; j < D; j++)
                yv[j] = y0(j, bb);
            float *e = &ec[(size_t)bb * T];
            e[0] = energy(yv.data());
            for (int nn = 1; nn < T; nn++) {
                rhs(yv.data(), k1.data());
                for (int j = 0; j < D; j++)
                    t[j] = yv[j] + 0.5f * h * k1[j];
                rhs(t.data(), k2.data());
                for (int j = 0; j < D; j++)
                    t[j] = yv[j] + 0.5f * h * k2[j];
                rhs(t.data(), k3.data());
                for (int j = 0; j < D; j++)
                    t[j] = yv[j] + h * k3[j];
                rhs(t.data(), k4.data());
                for (int j = 0; j < D; j++)
                    yv[j] += (h / 6.f) * (k1[j] + 2 * k2[j] + 2 * k3[j] + k4[j]);
                e[nn] = energy(yv.data());
            }
        });
    });

    namespace odeint = boost::numeric::odeint;
    using state = std::vector<float>;
    typedef odeint::runge_kutta4<state, float, state, float, odeint::range_algebra,
                                 odeint::default_operations, odeint::initially_resizer>
        rk4_f;
    hb::Stats s_boost = hb::bench([&]() {
        pool_for(par_for, B, [&](int bb) {
            state x(D);
            for (int j = 0; j < D; j++)
                x[j] = y0(j, bb);
            int step = 0;
            auto sys = [&](const state &yv, state &dy, float) { rhs(yv.data(), dy.data()); };
            auto obs = [&](const state &xs, float) {
                if (step < T) eb[(size_t)bb * T + step++] = energy(xs.data());
            };
            odeint::integrate_n_steps(rk4_f(), sys, x, 0.0f, h, T - 1, obs);
        });
    });

#ifdef WITH_SUNDIALS
    // ARKODE's explicit integrator with the classical RK4 table and a fixed
    // step, one step per call, in a single-precision build of SUNDIALS.
    static_assert(sizeof(sunrealtype) == sizeof(float), "SUNDIALS must be built with SUNDIALS_PRECISION=SINGLE");
    std::vector<float> es((size_t)B * T);
    std::function<void(const float *, float *)> rhs_fn = rhs;
    auto f_sun = [](sunrealtype, N_Vector yv, N_Vector fv, void *user) -> int {
        (*(std::function<void(const float *, float *)> *)user)(N_VGetArrayPointer(yv), N_VGetArrayPointer(fv));
        return 0;
    };
    hb::Stats s_sun = hb::bench([&]() {
        pool_for(par_for, B, [&](int bb) {
            SUNContext ctx;
            SUNContext_Create(SUN_COMM_NULL, &ctx);
            N_Vector x = N_VNew_Serial(D, ctx);
            float *xp = N_VGetArrayPointer(x);
            for (int j = 0; j < D; j++)
                xp[j] = y0(j, bb);
            void *ark = ERKStepCreate(f_sun, 0.0f, x, ctx);
            ARKodeSetUserData(ark, &rhs_fn);
            ERKStepSetTableNum(ark, ARKODE_KUTTA_RK4a_4_4);
            ARKodeSetFixedStep(ark, h);
            ARKodeSetMaxNumSteps(ark, T);
            float *e = &es[(size_t)bb * T];
            e[0] = energy(xp);
            for (int nn = 1; nn < T; nn++) {
                sunrealtype t;
                ARKodeEvolve(ark, nn * h, x, &t, ARK_ONE_STEP);
                e[nn] = energy(xp);
            }
            ARKodeFree(&ark);
            N_VDestroy(x);
            SUNContext_Free(&ctx);
        });
    });
#endif

    Buffer<float> ef = Buffer<float>(T, B).transposed(0, 1);
    E.realize(ef);
    hb::Stats s_fold = hb::bench([&]() { E.realize(ef); });

    auto abserr = [&](auto get) {
        double e = 0;
        for (int bb = 0; bb < B; bb++)
            for (int nn = 0; nn < T; nn++)
                e = std::max(e, std::abs((double)get(bb, nn) - ec[(size_t)bb * T + nn]));
        return e;
    };
    const double tol = 1e-4;
    double err_f = abserr([&](int bb, int nn) { return ef(bb, nn); });
    double err_b = abserr([&](int bb, int nn) { return eb[(size_t)bb * T + nn]; });
    const double thr = (double)D * (double)T * B / 1e6;
    char note[128];
    snprintf(note, sizeof(note), "RK4, fixed h=%g  D=%d B=%d T=%d", h, D, B, T);
    hb::print_spec_header("ode_observer_sparse_fused_rk4", "host", note);
    hb::print_row("RK4 textbook C++ loop", s_cpp, thr / (s_cpp.min * 1e-3), "Mupd/s", 0.0, 0.0, true);
    hb::print_row("RK4 Boost.odeint", s_boost, thr / (s_boost.min * 1e-3), "Mupd/s", 0.0, err_b, err_b < tol);
    bool pass = err_f < tol && err_b < tol;
#ifdef WITH_SUNDIALS
    double err_s = abserr([&](int bb, int nn) { return es[(size_t)bb * T + nn]; });
    hb::print_row("RK4 SUNDIALS ARKODE (fixed h)", s_sun, thr / (s_sun.min * 1e-3), "Mupd/s", 0.0, err_s, err_s < tol);
    pass = pass && err_s < tol;
#endif
    hb::print_row("RK4 inductive FOLDED (fold n -> 2)", s_fold, thr / (s_fold.min * 1e-3), "Mupd/s", 0.0, err_f, err_f < tol);
    printf("  -> %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}

}  // namespace

int main(int argc, char **argv) {
    int D = argc > 1 ? atoi(argv[1]) : 1024;
    int B = argc > 2 ? atoi(argv[2]) : 1;
    int T = argc > 3 ? atoi(argv[3]) : 8192;
    const bool rk4 = argc > 4 && !strcmp(argv[4], "rk4");
    const float h = 0.02f, eps = 0.1f;
    // The inductive observer keeps 16 lane partials along the slice.
    if (D % 16 != 0) {
        fprintf(stderr, "D must be a multiple of 16\n");
        return 1;
    }

    Buffer<float> y0(D, B);
    srand(5);
    for (int bb = 0; bb < B; bb++)
        for (int i = 0; i < D; i++)
            y0(i, bb) = (float)(rand() % 200) / 100.0f - 1.0f;
    if (rk4) {
        return run_rk4(D, B, T, h, eps, y0);
    }

    Var d("d"), b("b"), n("n");

    // The right-hand side and one classical RK4 step, on the host: the
    // integrator is Adams-Bashforth 2, which needs two initial slices, so
    // y1 is one RK4 step from y0, computed here and handed to every
    // pipeline as an input. The fused C++ loop uses the same two functions
    // for its start; Boost's stepper initializes itself with its own
    // runge_kutta4.
    // The two boundary elements apart, so the interior vectorizes: with the
    // neighbours read through conditional indices the whole stencil stays
    // scalar.
    auto rhs_at = [&](float lm, float c, float lp) { return eps * (lm - 2.f * c + lp) + c - c * c * c; };
    auto rhs = [&](const std::vector<float> &yv, std::vector<float> &fv) {
        const float *__restrict__ y = yv.data();
        float *__restrict__ f = fv.data();
        f[0] = rhs_at(y[0], y[0], y[1]);
        for (int i = 1; i < D - 1; i++)
            f[i] = rhs_at(y[i - 1], y[i], y[i + 1]);
        f[D - 1] = rhs_at(y[D - 2], y[D - 1], y[D - 1]);
    };
    auto rk4_step = [&](std::vector<float> &y) {
        std::vector<float> k1(D), k2(D), k3(D), k4(D), tmp(D);
        rhs(y, k1);
        for (int i = 0; i < D; i++)
            tmp[i] = y[i] + 0.5f * h * k1[i];
        rhs(tmp, k2);
        for (int i = 0; i < D; i++)
            tmp[i] = y[i] + 0.5f * h * k2[i];
        rhs(tmp, k3);
        for (int i = 0; i < D; i++)
            tmp[i] = y[i] + h * k3[i];
        rhs(tmp, k4);
        for (int i = 0; i < D; i++)
            y[i] += (h / 6.f) * (k1[i] + 2 * k2[i] + 2 * k3[i] + k4[i]);
    };

    Buffer<float> y1(D, B);
    {
        std::vector<float> yv(D);
        for (int bb = 0; bb < B; bb++) {
            for (int i = 0; i < D; i++)
                yv[i] = y0(i, bb);
            rk4_step(yv);
            for (int i = 0; i < D; i++)
                y1(i, bb) = yv[i];
        }
    }

    // The stencil right-hand side of a slice, as the host computes it:
    // f = eps*lap + c - c^3, left-associated.
    auto stencil = [&](Expr left, Expr center, Expr right) {
        Expr lap = left - 2.0f * center + right;
        return eps * lap + center - center * center * center;
    };

    // The observer as a second pass over a slice: the mean order parameter
    // <y> = (1/D) sum_i y_i, the standard phase-field monitoring diagnostic.
    // Summed in two stages: 16 partial sums across the lanes with plain
    // vector adds over the slice, then those 16 reduced once per step,
    // rather than a horizontal reduction of every vector into the scalar.
    auto observe = [&](Func E, auto slice) {
        RDom rd(0, D, "rd");
        E(b, n) = cast<float>(0);
        E(b, n) += slice(rd.x, b, n) * (1.0f / D);
        RVar rdo("rdo"), rdi("rdi");
        Var l("l");
        E.update(0).split(rd.x, rdo, rdi, 16);
        Func partial = E.update(0).rfactor(rdi, l);
        partial.compute_at(E, n).vectorize(l);
        partial.update(0).reorder(l, rdo).vectorize(l);
        E.compute_root();
        E.update(0).reorder(rdi, n, b).parallel(b);
        E.reorder(n, b).vectorize(n, 16).parallel(b);
        E.bound(b, 0, B).bound(n, 0, T);
    };

    // Build one inductive pipeline, the observer folded into the step.
    // `fold_k` pins the state window (2 = folded, T = unfolded ablation).
    // fold_k = 0 is the fusion-only ablation: y is computed at E's step loop
    // but stored there too, so nothing slides, and each step recomputes the
    // whole trajectory from the base case: O(T^2), run only at small T.
    auto build = [&](const char *tag, int fold_k) {
        // The state carries {y_n, f(y_{n-1}), running sum}: each step
        // evaluates the right-hand side once, on the slice it carries, and
        // reads the one before from the carried component, so nothing
        // reaches back more than one step. The running sum is the observer
        // folded into the step: each element adds its value to the sum 16
        // elements before it in the same slice, so the last 16 elements
        // hold 16 lane partials of the slice's sum, and the vector of 16
        // lanes reads the vector written just before it.
        Func y({Float(32), Float(32), Float(32)}, 3, std::string("y_") + tag), E(std::string("E_") + tag);
        // The walk over the slice writes every element of every step, base
        // cases included, so the pure definition is left undefined: no fill
        // of the window before each step.
        y(d, b, n) = Tuple(undef<float>(), undef<float>(), undef<float>());
        RDom r(0, D, "r");
        Expr i = r.x;
        Expr iL = max(i - 1, 0), iR = min(i + 1, D - 1);
        Expr c1 = y(i, b, n - 1)[0];
        Expr f1 = stencil(y(iL, b, n - 1)[0], c1, y(iR, b, n - 1)[0]);
        Expr f2 = y(i, b, n - 1)[1];
        Expr step_y = c1 + h * (1.5f * f1 - 0.5f * f2);
        // Base cases: n<=0 -> y0 (its derivative is never read); n==1 -> the
        // RK4 slice y1, carrying f(y0) for the first AB2 step.
        Expr f0 = stencil(y0(iL, b), y0(i, b), y0(iR, b));
        Expr val = select(n <= 1, select(n <= 0, y0(i, b), y1(i, b)), likely(step_y));
        Expr fval = select(n <= 1, select(n <= 0, cast<float>(0), f0), likely(f1));
        Expr carried = select(i >= 16, y(max(i - 16, 0), b, n)[2], cast<float>(0));
        y(i, b, n) = Tuple(val, fval, carried + val);

        // The observer reads the 16 lane partials at the end of the slice.
        RDom rl(0, 16, "rl");
        E(b, n) = cast<float>(0);
        E(b, n) += y(D - 16 + rl, b, n)[2] * (1.0f / D);
        E.compute_root();
        // The 16 partials reduce as a shuffle tree rather than a chain of
        // 16 dependent adds per step.
        E.update(0).reorder(rl, n, b).atomic().vectorize(rl).parallel(b);
        // The output is n-dense, so its zero fill runs along n, in parallel
        // over the batch, not as a serial walk striding across trajectories.
        E.reorder(n, b).vectorize(n, 16).parallel(b);
        E.bound(b, 0, B).bound(n, 0, T);

        // Each trajectory's window is its own: stored at its batch index.
        // The output is laid out with n dense and b strided (see the buffer
        // allocation below), so each trajectory's observer strip is contiguous
        // and the threads do not false-share it.
        E.output_buffer().dim(0).set_stride(T).dim(1).set_stride(1);
        if (fold_k == 0) {
            y.compute_at(E, n);
        } else {
            y.compute_at(E, n).store_at(E, b).fold_storage(n, fold_k);
        }
        // The stencil's reads of the previous slice pass the race analysis
        // (n is inductive, so its loop is serial); the lane partial's read
        // sixteen elements back in the same slice does not. It is safe at
        // this vector width, each vector reading the one written before
        // it, which the analysis cannot see.
        y.update(0).allow_race_conditions().vectorize(r.x, 16);
        E.compile_jit();
        return E;
    };

    // Three-way ablation: inductive FOLDED (2-slice window) vs inductive
    // UNFOLDED (fold n -> T, full trajectory, same fusion) vs non-inductive
    // materialize.
    Func E_fold = build("fold", 2);
    Func E_unfold = build("unfold", T);
    // The fusion-only ablation, when HB_FUSION is set.
    const bool fusion = getenv("HB_FUSION") != nullptr;
    Func E_fusion = fusion ? build("fusion", 0) : Func();

    // Non-inductive (materialize full trajectory) with the two-pass observer.
    // It has every slice in memory, so it recomputes the stencil of the slice
    // two back rather than carrying the derivative: another trajectory's
    // worth of traffic would cost more than the arithmetic saves.
    Func E_mat("E_mat");
    {
        Func y_m(Float(32), "y_m");
        // r.x (row) is an RVar, not a plain pure Var, so the stencil's ±1 shifts
        // can't look like a shift of a pure dimension to is_inductive().
        RDom r(0, D, 2, T - 2, "r");  // r.x = row, r.y = rn; slices 0,1 seeded, scan from n=2
        Expr rd_ = r.x, rn = r.y;
        y_m(d, b, n) = undef<float>();
        y_m(d, b, 0) = y0(d, b);
        y_m(d, b, 1) = y1(d, b);  // RK4 startup
        Expr p1 = rn - 1, p2 = rn - 2;
        auto f = [&](Expr t) {
            return stencil(y_m(max(rd_ - 1, 0), b, t), y_m(rd_, b, t),
                           y_m(min(rd_ + 1, D - 1), b, t));
        };
        y_m(rd_, b, rn) = y_m(rd_, b, p1) + h * (1.5f * f(p1) - 0.5f * f(p2));

        observe(E_mat, [&](Expr x, Expr bb, Expr nn) { return y_m(x, bb, nn); });
        E_mat.output_buffer().dim(0).set_stride(T).dim(1).set_stride(1);
        y_m.compute_root();
        y_m.update(0).unscheduled();                                                    // slice 0 init
        y_m.update(1).unscheduled();                                                    // slice 1 (RK4) init
        y_m.update(2).reorder(r.x, r.y, b).allow_race_conditions().vectorize(r.x, 16).parallel(b);  // AB2 scan
        E_mat.compile_jit();
    }

    // The baselines' loops over trajectories run as tasks of the Halide
    // runtime's thread pool, one trajectory per task: the same persistent
    // pool the Halide forms run on, reached through the JIT runtime, which
    // exists once a pipeline has been compiled.
    typedef int (*par_for_t)(void *, halide_task_t, int, int, uint8_t *);
    par_for_t halide_par_for = (par_for_t)Internal::JITSharedRuntime::find_symbol(
        get_jit_target_from_environment(), "halide_do_par_for");
    if (!halide_par_for) {
        fprintf(stderr, "halide_do_par_for not found in the JIT runtime\n");
        return 1;
    }
    auto pool_for = [&](int n, const std::function<void(int)> &body) {
        halide_par_for(
            nullptr, [](void *, int i, uint8_t *closure) {
                (*(const std::function<void(int)> *)closure)(i);
                return 0;
            },
            0, n, (uint8_t *)&body);
    };

    // Each trajectory's observer output is written by its own thread, one
    // value per step. Laid out (b, n) with b contiguous, the B threads write
    // B adjacent floats at each step -- a couple of cache lines that bounce
    // between cores. Give each trajectory a contiguous strip instead (n
    // dense), so a thread's writes stay on its own lines, as the C++
    // baselines' per-trajectory output does.
    Buffer<float> ef = Buffer<float>(T, B).transposed(0, 1);
    Buffer<float> eu = Buffer<float>(T, B).transposed(0, 1);
    Buffer<float> emat = Buffer<float>(T, B).transposed(0, 1);
    E_fold.realize(ef);
    E_unfold.realize(eu);
    E_mat.realize(emat);
    hb::Stats s_fold = hb::bench([&]() { E_fold.realize(ef); });
    hb::Stats s_unfold = hb::bench([&]() { E_unfold.realize(eu); });
    hb::Stats s_mat = hb::bench([&]() { E_mat.realize(emat); });
    Buffer<float> efu = Buffer<float>(T, B).transposed(0, 1);
    hb::Stats s_fusion;
    double bytes_fus = 0;
    if (fusion) {
        E_fusion.realize(efu);
        s_fusion = hb::bench([&]() { E_fusion.realize(efu); });
        bytes_fus = hb::profiled_peak_bytes(E_fusion, efu);
    }

    // Measured peak internal scratch (untimed, separate from the benches above).
    double bytes_ind = hb::profiled_peak_bytes(E_fold, ef);
    double bytes_unf = hb::profiled_peak_bytes(E_unfold, eu);
    double bytes_non = hb::profiled_peak_bytes(E_mat, emat);

    // The mean order parameter <y> of a slice, as a separate pass; the
    // vectorizer keeps the float sum in lane partials.
    auto energy = [&](const std::vector<float> &y) {
        float e = 0;
        for (int i = 0; i < D; i++)
            e += y[i];
        return e / D;
    };
    namespace odeint = boost::numeric::odeint;
    using state = std::vector<float>;
    auto sys = [&](const state &y, state &dy, float) { rhs(y, dy); };
    // Idiomatic Boost.odeint: integrate_n_steps + observer, with runge_kutta4 as the
    // initializing stepper, the same startup as the host's rk4_step. Value, Deriv
    // and Time are all float, so no element is promoted.
    typedef odeint::runge_kutta4<state, float, state, float, odeint::range_algebra,
                                 odeint::default_operations, odeint::initially_resizer>
        rk4_f;
    typedef odeint::adams_bashforth<2, state, float, state, float, odeint::range_algebra,
                                    odeint::default_operations, odeint::initially_resizer, rk4_f>
        ab2_rk4;
    std::vector<float> eb((size_t)B * T);
    hb::Stats s_boost = hb::bench([&]() {
        pool_for(B, [&](int bb) {
            state x(D);
            for (int i = 0; i < D; i++)
                x[i] = y0(i, bb);
            int step = 0;
            auto obs = [&](const state &xs, float) {
                if (step < T) eb[(size_t)bb * T + step++] = energy(xs);
            };
            ab2_rk4 ab;
            odeint::integrate_n_steps(ab, sys, x, 0.0f, h, T - 1, obs);
        });
    });

#ifdef WITH_SUNDIALS
    // SUNDIALS, built in single precision so that every value is float, as
    // above. Its integrators have no Adams-Bashforth 2 (CVODE's Adams family
    // is implicit and adaptive), so it gets two rows: ARKODE's explicit
    // integrator with a fixed step h and Heun's method, the same order as
    // AB2 at two right-hand sides a step; and AB2 itself written with
    // SUNDIALS' vector operations, the same arithmetic as Boost's stepper.
    // Each trajectory has its own context and runs as a task on the Halide
    // pool, as Boost's do. The right-hand side works on the vector's data in
    // place.
    static_assert(sizeof(sunrealtype) == sizeof(float), "SUNDIALS must be built with SUNDIALS_PRECISION=SINGLE");
    struct SunRhs {
        int D;
        float eps;
    };
    auto f_sun = [](sunrealtype, N_Vector yv, N_Vector fv, void *user) -> int {
        const SunRhs *s = (const SunRhs *)user;
        const int D = s->D;
        const float eps = s->eps;
        const float *__restrict__ y = N_VGetArrayPointer(yv);
        float *__restrict__ f = N_VGetArrayPointer(fv);
        auto at = [&](float lm, float c, float lp) { return eps * (lm - 2.f * c + lp) + c - c * c * c; };
        f[0] = at(y[0], y[0], y[1]);
        for (int i = 1; i < D - 1; i++)
            f[i] = at(y[i - 1], y[i], y[i + 1]);
        f[D - 1] = at(y[D - 2], y[D - 1], y[D - 1]);
        return 0;
    };
    SunRhs sun_user{D, eps};
    auto sun_energy = [&](N_Vector x) {
        const float *p = N_VGetArrayPointer(x);
        float e = 0;
        for (int i = 0; i < D; i++)
            e += p[i];
        return e / D;
    };
    std::vector<float> es((size_t)B * T), ev((size_t)B * T);
    hb::Stats s_ark = hb::bench([&]() {
        pool_for(B, [&](int bb) {
            SUNContext ctx;
            SUNContext_Create(SUN_COMM_NULL, &ctx);
            N_Vector x = N_VNew_Serial(D, ctx);
            float *xp = N_VGetArrayPointer(x);
            for (int i = 0; i < D; i++)
                xp[i] = y0(i, bb);
            void *ark = ERKStepCreate(f_sun, 0.0f, x, ctx);
            ARKodeSetUserData(ark, &sun_user);
            ERKStepSetTableNum(ark, ARKODE_HEUN_EULER_2_1_2);
            ARKodeSetFixedStep(ark, h);
            ARKodeSetMaxNumSteps(ark, T);
            es[(size_t)bb * T] = sun_energy(x);
            for (int nn = 1; nn < T; nn++) {
                sunrealtype t;
                ARKodeEvolve(ark, nn * h, x, &t, ARK_ONE_STEP);
                es[(size_t)bb * T + nn] = sun_energy(x);
            }
            ARKodeFree(&ark);
            N_VDestroy(x);
            SUNContext_Free(&ctx);
        });
    });
    hb::Stats s_nvec = hb::bench([&]() {
        pool_for(B, [&](int bb) {
            SUNContext ctx;
            SUNContext_Create(SUN_COMM_NULL, &ctx);
            N_Vector x = N_VNew_Serial(D, ctx), f1 = N_VClone(x), f2 = N_VClone(x);
            float *xp = N_VGetArrayPointer(x);
            // The same start as the fused loop: f(y0), then y1 by RK4.
            std::vector<float> xs(D);
            for (int i = 0; i < D; i++)
                xs[i] = xp[i] = y0(i, bb);
            ev[(size_t)bb * T] = sun_energy(x);
            f_sun(0, x, f2, &sun_user);
            rk4_step(xs);
            std::copy(xs.begin(), xs.end(), xp);
            ev[(size_t)bb * T + 1] = sun_energy(x);
            sunrealtype c[3] = {1.0f, 1.5f * h, -0.5f * h};
            for (int nn = 2; nn < T; nn++) {
                f_sun(0, x, f1, &sun_user);
                N_Vector v[3] = {x, f1, f2};
                N_VLinearCombination(3, c, v, x);  // y += h * (1.5 f1 - 0.5 f2)
                std::swap(f1, f2);
                ev[(size_t)bb * T + nn] = sun_energy(x);
            }
            N_VDestroy(f2);
            N_VDestroy(f1);
            N_VDestroy(x);
            SUNContext_Free(&ctx);
        });
    });
#endif

    // The fused C++ loop. Five slices rotate by pointer: the current y and
    // its right-hand side f1, the previous right-hand side f2, and the new
    // slice and its right-hand side. One pass per step writes the AB2
    // update, the right-hand side of the new slice (its neighbours' updates
    // recomputed from the inputs, so the loop carries nothing through
    // memory), and the slice's sum, which the vectorizer keeps in lane
    // partials. A non-capturing function, its parameters plain locals: the
    // vectorizer does not see through a closure's members.
    auto ab2_steps = [](int D, int T, float h, float eps, float *y, float *f1, float *f2,
                        float *y_new, float *f_new, float *ec_row, auto rhs_at) {
        for (int nn = 2; nn < T; nn++) {
            const float *__restrict__ yi = y, *__restrict__ fa = f1, *__restrict__ fb = f2;
            float *__restrict__ yo = y_new, *__restrict__ fo = f_new;
#define AB2(j) (yi[j] + h * (1.5f * fa[j] - 0.5f * fb[j]))
            float e = 0;
            yo[0] = AB2(0);
            fo[0] = rhs_at(yo[0], yo[0], AB2(1));
            e += yo[0];
            for (int i = 1; i < D - 1; i++) {
                float cl = AB2(i - 1), c = AB2(i), cr = AB2(i + 1);
                yo[i] = c;
                fo[i] = eps * (cl - 2.f * c + cr) + c - c * c * c;
                e += c;
            }
            yo[D - 1] = AB2(D - 1);
            fo[D - 1] = rhs_at(AB2(D - 2), yo[D - 1], yo[D - 1]);
            e += yo[D - 1];
#undef AB2
            ec_row[nn] = e / D;
            std::swap(y, y_new);
            float *dead = f2;
            f2 = f1;
            f1 = f_new;
            f_new = dead;
        }
    };
    std::vector<float> ec((size_t)B * T);
    hb::Stats s_cpp = hb::bench([&]() {
        pool_for(B, [&](int bb) {
            std::vector<float> s0(D), s1(D), s2(D), s3(D), s4(D);
            for (int i = 0; i < D; i++)
                s0[i] = y0(i, bb);
            ec[(size_t)bb * T + 0] = energy(s0);  // n=0
            rhs(s0, s3);                          // f_0
            rk4_step(s0);                         // n=1 via RK4
            ec[(size_t)bb * T + 1] = energy(s0);
            rhs(s0, s2);  // f_1
            ab2_steps(D, T, h, eps, s0.data(), s2.data(), s3.data(), s1.data(), s4.data(),
                      &ec[(size_t)bb * T], rhs_at);
        });
    });

    // The observable <y> is a mean of D values in [-1, 1], so its scale is
    // one and it passes through zero as the field coarsens; a relative check
    // against it blows up at those crossings on a difference that is float
    // round-off. So the forms are compared by absolute error, which on an
    // O(1) observable is the meaningful metric.
    auto abserr = [&](auto get) {
        double e = 0;
        for (int bb = 0; bb < B; bb++)
            for (int nn = 0; nn < T; nn++)
                e = std::max(e, std::abs((double)get(bb, nn) - ec[(size_t)bb * T + nn]));
        return e;
    };
    double err_f = abserr([&](int bb, int nn) { return ef(bb, nn); });
    double err_u = abserr([&](int bb, int nn) { return eu(bb, nn); });
    double err_m = abserr([&](int bb, int nn) { return emat(bb, nn); });
    double err_b = abserr([&](int bb, int nn) { return eb[(size_t)bb * T + nn]; });
    // Float AB2 over thousands of steps against the fused float loop: the
    // observable is O(1) and the forms agree to a few 1e-7.
    const double tol = 1e-4;

    const double thr = (double)D * (double)T * B / 1e6;  // M state-updates
    // Analytic unfolded footprint (full trajectory D*B*T floats) = the roofline
    // x-axis; recurrence-length (T) and batch (B) sweeps collapse onto fp/LLC.
    const double fp_unfold = (double)D * B * T * 4;
    char note[176];
    snprintf(note, sizeof(note),
             "SPARSE observer (fold ablation)  D=%d B=%d T=%d  |  state fold %.0fx (%.2f -> %.2f MB)  |  unfolded fp/LLC=%.3f",
             D, B, T, hb::mem_ratio(bytes_non, bytes_ind), hb::mb(bytes_non), hb::mb(bytes_ind),
             hb::footprint_over_llc(fp_unfold));
    hb::print_spec_header("ode_observer_sparse_fused", "host", note);
    hb::print_row("Boost.odeint (rk4 init + observer)", s_boost, thr / (s_boost.min * 1e-3),
                  "Mupd/s", (double)D * 4, err_b, err_b < tol);
    hb::print_row("fused C++ loop", s_cpp, thr / (s_cpp.min * 1e-3),
                  "Mupd/s", 0.0, 0.0, true);
#ifdef WITH_SUNDIALS
    // Heun is a different method, so its distance from the AB2 trajectory
    // is the two methods' difference, not round-off: shown, not checked.
    double err_s = abserr([&](int bb, int nn) { return es[(size_t)bb * T + nn]; });
    double err_v = abserr([&](int bb, int nn) { return ev[(size_t)bb * T + nn]; });
    hb::print_row("SUNDIALS ARKODE Heun (fixed h)", s_ark, thr / (s_ark.min * 1e-3),
                  "Mupd/s", 0.0, err_s, true);
    hb::print_row("SUNDIALS N_Vector AB2", s_nvec, thr / (s_nvec.min * 1e-3),
                  "Mupd/s", 0.0, err_v, err_v < tol);
#endif
    hb::print_row("non-inductive (materialize)", s_mat, thr / (s_mat.min * 1e-3),
                  "Mupd/s", bytes_non, err_m, err_m < tol, "", fp_unfold);
    double err_fu = fusion ? abserr([&](int bb, int nn) { return efu(bb, nn); }) : 0.0;
    if (fusion) {
        hb::print_row("inductive FUSION ONLY (no slide)", s_fusion, thr / (s_fusion.min * 1e-3),
                      "Mupd/s", bytes_fus, err_fu, err_fu < tol, hb::verdict(s_fusion.min, s_mat.min), fp_unfold);
    }
    hb::print_row("inductive UNFOLDED (fold n -> T)", s_unfold, thr / (s_unfold.min * 1e-3),
                  "Mupd/s", bytes_unf, err_u, err_u < tol, hb::verdict(s_unfold.min, s_mat.min), fp_unfold);
    hb::print_row("inductive FOLDED (fold n -> 2)", s_fold, thr / (s_fold.min * 1e-3),
                  "Mupd/s", bytes_ind, err_f, err_f < tol, hb::verdict(s_fold.min, s_unfold.min), fp_unfold);
    printf("  gap (inductive-fold vs Boost): %+.1f%%   inductive-fold vs non-ind: %.2fx\n",
           100.0 * (s_fold.min - s_boost.min) / s_boost.min, s_mat.min / s_fold.min);
    bool pass = err_f < tol && err_u < tol && err_m < tol && err_b < tol && err_fu < tol;
    printf("  -> %s\n", pass ? "PASS" : "FAIL");
    return pass ? 0 : 1;
}
