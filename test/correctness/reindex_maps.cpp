#include "Halide.h"

#include <cstdio>
#include <functional>
#include <random>

// Bijection-variety + randomized coverage for Stage::reindex(). A commutative sum
// over a box is invariant under any reindexing of the reduction domain, so we
// reindex it by a spread of maps -- identity, transpose, reflection, skew, shear,
// shift, a non-affine modular shift, and many seeded random unimodular maps
// (integer matrices with det = +/-1, hence integer inverses) -- and require the
// sum to match the unreindexed reference every time.

using namespace Halide;

namespace {

const int W = 7, H = 6;

Buffer<int> make_input() {
    Buffer<int> in(W, H, "in");
    for (int y = 0; y < H; y++) {
        for (int x = 0; x < W; x++) {
            in(x, y) = x * 7 + y * 3 + 1;
        }
    }
    return in;
}

int64_t sum_with(const Buffer<int> &in,
                 const std::function<void(Stage, RDom)> &sched) {
    Func f;
    RDom r(0, W, 0, H, "r");
    f() = 0;
    f() += in(r.x, r.y);
    sched(f.update(), r);
    Buffer<int> out = f.realize();
    return out();
}

bool check(const Buffer<int> &in, int64_t ref, const char *name,
           const std::function<void(Stage, RDom)> &sched) {
    const int64_t got = sum_with(in, sched);
    if (got != ref) {
        printf("[%s] wrong: %lld vs %lld\n", name, (long long)got,
               (long long)ref);
        return false;
    }
    return true;
}

}  // namespace

int main(int argc, char **argv) {
    Buffer<int> in = make_input();
    const int64_t ref = sum_with(in, [](Stage, RDom) {});

    bool ok = true;
    ok &= check(in, ref, "identity", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {r.x, r.y}, {u, v});
    });
    ok &= check(in, ref, "transpose", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {r.y, r.x}, {v, u});
    });
    ok &= check(in, ref, "reflect-x", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {(W - 1) - r.x, r.y}, {(W - 1) - u, v});
    });
    ok &= check(in, ref, "skew", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {r.x, r.x + r.y}, {u, v - u});
    });
    ok &= check(in, ref, "shear", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {r.x + 2 * r.y, r.y}, {u - 2 * v, v});
    });
    ok &= check(in, ref, "shift", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {r.x + 3, r.y - 1}, {u - 3, v + 1});
    });
    ok &= check(in, ref, "modular", [](Stage s, RDom r) {
        Var u("u"), v("v");
        s.reindex({u, v}, {(r.x + 3) % W, r.y}, {(u + (W - 3)) % W, v});
    });

    // Randomized unimodular maps (seeded for reproducibility).
    std::mt19937 rng(12345);
    std::uniform_int_distribution<int> d(-2, 2);
    int tried = 0;
    for (int i = 0; i < 200 && tried < 25; i++) {
        const int a = d(rng), b = d(rng), c = d(rng), e = d(rng);
        const int det = a * e - b * c;
        if (det != 1 && det != -1) {
            continue;
        }
        tried++;
        char name[64];
        snprintf(name, sizeof(name), "unimodular[%d,%d;%d,%d]", a, b, c, e);
        // inverse: rx = det*(e*u - b*v), ry = det*(-c*u + a*v)  (det = +/-1)
        ok &= check(in, ref, name, [=](Stage s, RDom r) {
            Var u("u"), v("v");
            s.reindex({u, v}, {a * r.x + b * r.y, c * r.x + e * r.y},
                      {det * (e * u - b * v), det * (-c * u + a * v)});
        });
    }

    if (!ok) {
        return 1;
    }
    printf("Success!\n");
    return 0;
}
