# The artifact as a container. Build from the repository root:
#
#   CPU:  docker build -f inductive_artifact/Dockerfile -t inductive-artifact .
#   GPU:  docker build -f inductive_artifact/Dockerfile -t inductive-artifact-gpu \
#             --build-arg BASE=nvidia/cuda:13.0.2-devel-ubuntu24.04 \
#             --build-arg PYTHON_BASELINES=1 .
#
# and run, from /artifact inside the container:
#
#   docker run --rm -it inductive-artifact inductive_artifact/basic_test.sh
#   docker run --rm -it inductive-artifact inductive_artifact/run_benchmarks.sh
#   docker run --rm -it --gpus all inductive-artifact-gpu \
#       inductive_artifact/run_benchmarks.sh --only cuda_mamba2,flash_attention
#
# The image holds everything the README installs: the apt packages, LLVM and
# the Halide build (setup.sh), Julia and parasail, and with PYTHON_BASELINES=1
# the Python and GPU baselines. Timings in a container match the host's when
# it runs with the host's cores (no --cpus limit).
ARG BASE=ubuntu:24.04
FROM ${BASE}

ARG PYTHON_BASELINES=0
ENV DEBIAN_FRONTEND=noninteractive \
    UV_INSTALL_DIR=/opt/uv UV_PYTHON_INSTALL_DIR=/opt/uv-python UV_CACHE_DIR=/tmp/uv-cache \
    JULIAUP_DEPOT_PATH=/opt/juliaup JULIA_DEPOT_PATH=/opt/julia-depot \
    PARASAIL_DIR=/opt/parasail-2.6.2 \
    PATH=/opt/uv:/opt/juliaup-bin/bin:/usr/local/cuda/bin:$PATH \
    LIBRARY_PATH=/usr/local/cuda/lib64/stubs
# (The GPU runners link -lcuda; the CUDA image has only a stub of it, and
# the real driver library comes in at run time with --gpus.)

RUN apt-get update && apt-get install -y --no-install-recommends \
        build-essential cmake pkg-config ca-certificates curl git \
        libpng-dev libjpeg-turbo8-dev zlib1g-dev \
        libjemalloc2 libboost-dev libtbb-dev \
    && rm -rf /var/lib/apt/lists/*

RUN curl -LsSf https://astral.sh/uv/install.sh | sh

# The two baselines installed by hand (see the README).
RUN curl -fsSL https://install.julialang.org | \
        sh -s -- --yes --path /opt/juliaup-bin --add-to-path=no --default-channel 1.12 \
    && julia --version
RUN cd /opt && curl -fsSL https://github.com/jeffdaily/parasail/archive/refs/tags/v2.6.2.tar.gz | tar xz \
    && cmake -S parasail-2.6.2 -B parasail-2.6.2/build -DCMAKE_BUILD_TYPE=Release \
        -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5 \
    && cmake --build parasail-2.6.2/build --target parasail -j"$(nproc)"

COPY . /artifact
WORKDIR /artifact
RUN WITH_PYTHON_BASELINES=${PYTHON_BASELINES} inductive_artifact/setup.sh && rm -rf /tmp/uv-cache

CMD ["inductive_artifact/basic_test.sh"]
