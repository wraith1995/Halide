#include "Halide.h"
#include <cstdio>
#include <cstring>
#include <functional>
#include <string>
#include <vector>

// reindex() changes the order an update's iterations run in. Orders that keep
// every dependence of the update pointing forward are accepted silently, and
// give the same result as the original order. When reindex() can't tell, it
// warns. (Orders that provably break a dependence, and maps that provably
// aren't inverses, are errors; see the reindex_* tests in test/error.)

using namespace Halide;

namespace {

class CountWarnings : public CompileTimeErrorReporter {
public:
    std::vector<std::string> warnings;
    void warning(const char *msg) override {
        warnings.emplace_back(msg);
    }
    [[noreturn]] void error(const char *msg) override {
        printf("Unexpected error: %s\n", msg);
        exit(1);
    }
};

CountWarnings reporter;

// Runs `body`, which schedules and realizes something and returns whether the
// result was right, and checks how many warnings mentioning `about` it gave.
bool expect(const char *name, int warnings, const char *about, const std::function<bool()> &body) {
    reporter.warnings.clear();
    bool right = body();
    int n = 0;
    for (const std::string &w : reporter.warnings) {
        if (strstr(w.c_str(), about)) {
            n++;
        }
    }
    if (n != warnings) {
        printf("%s: expected %d warnings about \"%s\", got %d\n", name, warnings, about, n);
        for (const std::string &w : reporter.warnings) {
            printf("  %s", w.c_str());
        }
        return false;
    }
    if (!right) {
        printf("%s: wrong result\n", name);
        return false;
    }
    return true;
}

const char *order_warning = "keeps every dependence";
const char *inverse_warning = "inverses of each other";

}  // namespace

int main(int argc, char **argv) {
    set_custom_compile_time_error_reporter(&reporter);
    bool ok = true;

    // A recurrence reading the next element, reindexed by the identity.
    ok &= expect("identity", 0, order_warning, [] {
        Func f("f_identity");
        Var x("x"), u("u");
        f(x) = x;
        RDom r(0, 10);
        f(r) = f(r + 1) * 2 + 1;
        f.update().reindex({u}, {r.x}, {u});
        Buffer<int> b = f.realize({11});
        int ref[11];
        for (int i = 0; i < 11; i++) {
            ref[i] = i;
        }
        for (int i = 0; i < 10; i++) {
            ref[i] = ref[i + 1] * 2 + 1;
        }
        for (int i = 0; i < 11; i++) {
            if (b(i) != ref[i]) {
                return false;
            }
        }
        return true;
    });

    // A wavefront: skewing a 2-D recurrence keeps both dependences forward.
    const int W = 17, H = 13;
    auto wavefront_ref = [&]() {
        Buffer<int> ref(W, H);
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                ref(x, y) = (x == 0 || y == 0) ? x + y : (ref(x - 1, y) + ref(x, y - 1) + 1) % 1009;
            }
        }
        return ref;
    };
    auto wavefront = [&](const char *name) {
        Func f(name);
        Var x("x"), y("y");
        f(x, y) = x + y;
        RDom r(1, W - 1, 1, H - 1);
        f(r.x, r.y) = (f(r.x - 1, r.y) + f(r.x, r.y - 1) + 1) % 1009;
        return std::make_pair(f, r);
    };
    auto matches = [&](const Buffer<int> &b) {
        Buffer<int> ref = wavefront_ref();
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                if (b(x, y) != ref(x, y)) {
                    return false;
                }
            }
        }
        return true;
    };
    ok &= expect("skew", 0, order_warning, [&] {
        auto [f, r] = wavefront("f_skew");
        Var u("u"), v("v");
        f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        return matches(f.realize({W, H}));
    });

    // A diamond tile: split both dims, then skew the tile indices. The
    // dependences cross tile boundaries, which the check follows through the
    // splits.
    ok &= expect("diamond", 0, order_warning, [&] {
        auto [f, r] = wavefront("f_diamond");
        RVar rxo("rxo"), rxi("rxi"), ryo("ryo"), ryi("ryi");
        f.update().split(r.x, rxo, rxi, 4).split(r.y, ryo, ryi, 4);
        Var lxi("rxi"), lxo("rxo"), lyi("ryi"), lyo("ryo");
        Var nxi("nxi"), bv("bv"), nyi("nyi"), nyo("nyo");
        f.update().reindex({nxi, bv, nyi, nyo}, {lxi, lxo + lyo, lyi, lyo},
                           {nxi, bv - nyo, nyi, nyo}, {nxi, nyi, nyo, bv});
        return matches(f.realize({W, H}));
    });

    // Temporal tiling of a stencil over time: a skew, then a diamond tile of
    // the skewed space, analyzed in the coordinates of the first reindex.
    ok &= expect("temporal tile", 0, order_warning, [&] {
        const int N = 40, T = 12;
        Buffer<int> in(N);
        for (int i = 0; i < N; i++) {
            in(i) = (i * 7 + 3) % 101;
        }
        Func f("f_temporal");
        Var x("x"), t("t");
        f(x, t) = select(t == 0, in(x), 0);
        RDom r(1, N - 2, 1, T - 1);
        f(r.x, r.y) = (f(r.x - 1, r.y - 1) + 2 * f(r.x, r.y - 1) + f(r.x + 1, r.y - 1) + 1) % 101;
        Var u("u"), v("v");
        ReindexResult skew = f.update().reindex({u, v}, {r.x + r.y, r.y}, {u - v, v});
        RVar uo("uo"), ui("ui"), vo("vo"), vi("vi");
        f.update().split(skew.vars[0], uo, ui, 4).split(skew.vars[1], vo, vi, 4);
        Var lui("ui"), luo("uo"), lvi("vi"), lvo("vo");
        Var nui("nui"), bv("bv"), nvi("nvi"), p("p");
        f.update().reindex({nui, bv, nvi, p}, {lui, luo + lvo, lvi, lvo},
                           {nui, bv - p, nvi, p}, {nui, nvi, p, bv});
        Buffer<int> b = f.realize({N, T});
        std::vector<std::vector<int>> ref(T, std::vector<int>(N, 0));
        for (int i = 0; i < N; i++) {
            ref[0][i] = in(i);
        }
        for (int s = 1; s < T; s++) {
            for (int i = 1; i < N - 1; i++) {
                ref[s][i] = (ref[s - 1][i - 1] + 2 * ref[s - 1][i] + ref[s - 1][i + 1] + 1) % 101;
            }
        }
        for (int s = 0; s < T; s++) {
            for (int i = 0; i < N; i++) {
                if (b(i, s) != ref[s][i]) {
                    return false;
                }
            }
        }
        return true;
    });

    // An associative and commutative update may run in any order.
    ok &= expect("sum", 0, order_warning, [&] {
        Func f("f_sum");
        Var x("x"), u("u"), v("v");
        f(x) = 0;
        RDom r(0, 6, 0, 5);
        f(r.x) += r.y * 10 + r.x;
        f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u}, {v, u});
        Buffer<int> b = f.realize({6});
        for (int i = 0; i < 6; i++) {
            if (b(i) != 100 + 5 * i) {
                return false;
            }
        }
        return true;
    });

    // Every iteration reads and writes the same place, and the update isn't
    // commutative, so the order matters, and the check can't analyze it.
    ok &= expect("not commutative", 1, order_warning, [&] {
        Func g("g_nc"), f("f_nc");
        Var a("a"), c("c"), u("u"), v("v");
        g(a, c) = a * 4 + c;
        f() = 0;
        RDom r(0, 3, 0, 3);
        f() = f() * 2 + g(r.x, r.y);
        f.update().reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
        f.realize();
        return true;
    });

    // A rotation within the domain is its own inverse only on the domain, which
    // the simplifier can't show.
    ok &= expect("non-affine", 1, inverse_warning, [&] {
        Func f("f_rot");
        Var x("x"), u("u");
        f(x) = 0;
        RDom r(0, 8);
        f(r) += r;
        f.update().reindex({u}, {(r.x + 3) % 8}, {(u + 5) % 8});
        Buffer<int> b = f.realize({8});
        for (int i = 0; i < 8; i++) {
            if (b(i) != i) {
                return false;
            }
        }
        return true;
    });

    if (!ok) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
