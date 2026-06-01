#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func producer("producer"), consumer("consumer");
    Var x("x"), y("y");

    producer(x, y) = x + y;
    consumer(x, y) = producer(x, y);

    // async() + store_in(GPUShared) selects GPU warp specialization, which is
    // currently only supported on the CUDA target. Compiling to a non-CUDA
    // target must produce a clear error.
    producer.compute_root().store_in(MemoryType::GPUShared).async();

    consumer.compile_to_module({}, "consumer", Target("host"));

    printf("Success!\n");
    return 0;
}
