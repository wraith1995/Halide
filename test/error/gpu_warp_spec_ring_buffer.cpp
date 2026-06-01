#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func producer("producer"), consumer("consumer");
    Var x("x"), y("y");

    producer(x, y) = x + y;
    consumer(x, y) = producer(x, y);

    // Ring buffering combined with GPU warp specialization (async() +
    // store_in(GPUShared)) is not yet implemented and must error clearly,
    // even on the CUDA target.
    producer.compute_root().store_in(MemoryType::GPUShared).async().ring_buffer(2);

    consumer.compile_to_module({}, "consumer", Target("host-cuda"));

    printf("Success!\n");
    return 0;
}
