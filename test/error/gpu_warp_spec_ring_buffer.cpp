#include "Halide.h"
#include <stdio.h>

using namespace Halide;

int main(int argc, char **argv) {
    Func producer("producer"), consumer("consumer");
    Var x("x"), y("y"), xo("xo"), xi("xi");

    producer(x, y) = x + y;
    consumer(x, y) = producer(x, y);

    consumer.split(x, xo, xi, 16);

    // A well-formed ring-buffered schedule (store level inside hoist_storage
    // level) that also requests GPU warp specialization (async() +
    // store_in(GPUShared)). Ring buffering for warp-specialized producers is
    // not yet implemented and must error clearly on the CUDA target.
    producer.compute_at(consumer, xi)
        .store_in(MemoryType::GPUShared)
        .hoist_storage(consumer, xo)
        .async()
        .ring_buffer(2);

    consumer.compile_to_module({}, "consumer", Target("host-cuda"));

    printf("Success!\n");
    return 0;
}
