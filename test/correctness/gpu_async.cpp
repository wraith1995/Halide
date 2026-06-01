#include "Halide.h"
#include <stdio.h>

using namespace Halide;

// Correctness tests for GPU warp-specialized async producers: a producer
// scheduled with async() + store_in(GPUShared), computed between the GPU block
// and thread loops, is consumed within the same block. Until device-side warp
// specialization is fully implemented these degrade to synchronous shared-memory
// staging; either way the results must match a reference computation.

int main(int argc, char **argv) {
    Target target = get_jit_target_from_environment();
    if (!target.has_feature(Target::CUDA)) {
        printf("[SKIP] GPU warp specialization currently requires CUDA.\n");
        return 0;
    }

    const int W = 100, H = 80;  // deliberately not multiples of the tile size

    // Elementwise: producer staged into shared memory, consumed in-block.
    {
        Func producer("producer"), consumer("consumer");
        Var x("x"), y("y"), xo("xo"), yo("yo"), xi("xi"), yi("yi");
        producer(x, y) = x + 2 * y;
        consumer(x, y) = producer(x, y) + 1;

        consumer.compute_root().gpu_tile(x, y, xo, yo, xi, yi, 16, 16);
        producer.compute_at(consumer, xo)
            .store_in(MemoryType::GPUShared)
            .gpu_threads(x, y)
            .async();

        Buffer<int> out = consumer.realize({W, H}, target);
        out.copy_to_host();
        for (int y = 0; y < H; y++) {
            for (int x = 0; x < W; x++) {
                int correct = x + 2 * y + 1;
                if (out(x, y) != correct) {
                    printf("out(%d, %d) = %d instead of %d\n", x, y, out(x, y), correct);
                    return 1;
                }
            }
        }
    }

    printf("Success!\n");
    return 0;
}
