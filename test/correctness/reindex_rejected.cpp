#include "Halide.h"
#include <cstdio>

// A reindex() that is rejected leaves the stage as it was, even though
// reindexing a split reduction domain starts by lowering its splits.

using namespace Halide;

int main(int argc, char **argv) {
#ifndef HALIDE_WITH_EXCEPTIONS
    printf("[SKIP] reindex_rejected requires exceptions\n");
    return 0;
#else
    Func f("f");
    Var x("x"), u("u"), w("w");
    f(x) = x;
    RDom r(0, 16);
    // Each iteration reads the element the next one writes.
    f(r) = f(r + 1) * 2 + 1;
    RVar ro("ro"), ri("ri");
    f.update().split(r, ro, ri, 4);

    const size_t splits = f.update().get_schedule().splits().size();
    const size_t rvars = f.update().get_schedule().rvars().size();
    const std::string loops = f.update().dump_argument_list();

    // Reversing the order of the tiles runs each tile before the one it
    // reads from.
    Var lri("ri"), lro("ro");
    bool rejected = false;
    try {
        f.update().reindex({u, w}, {lri, 3 - lro}, {u, 3 - w});
    } catch (const CompileError &e) {
        rejected = true;
    }
    if (!rejected) {
        printf("expected the reindex to be rejected\n");
        return 1;
    }

    if (f.update().get_schedule().splits().size() != splits ||
        f.update().get_schedule().rvars().size() != rvars ||
        f.update().dump_argument_list() != loops) {
        printf("the rejected reindex changed the stage:\n%s\nbecame\n%s\n",
               loops.c_str(), f.update().dump_argument_list().c_str());
        return 1;
    }

    Buffer<int> b = f.realize({17});
    int ref[17];
    for (int i = 0; i < 17; i++) {
        ref[i] = i;
    }
    for (int i = 0; i < 16; i++) {
        ref[i] = ref[i + 1] * 2 + 1;
    }
    for (int i = 0; i < 17; i++) {
        if (b(i) != ref[i]) {
            printf("f(%d) = %d instead of %d\n", i, b(i), ref[i]);
            return 1;
        }
    }

    printf("Success!\n");
    return 0;
#endif
}
