# Inductive Funcs: artifact

Archived at Zenodo: [10.5281/zenodo.24284710](https://doi.org/10.5281/zenodo.24284710).

Scripts to build this fork of Halide and rerun the paper's benchmark table
and optimization ablation. The CGO artifact appendix is in `appendix/`
(`latexmk -pdf main.tex` builds a standalone copy; `artifact_appendix.tex` is
the part to `\input` into the paper, after `\appendix` and before the
bibliography; CGO places the artifact appendix before the References).

## Setup

Halide is built as the top-level `README.md` describes under "Building Halide
with make", with LLVM from Halide's package index ("Acquiring LLVM").
`setup.sh` does both: it creates a Python 3.12 environment in `.venv` holding
LLVM 22, then builds `distrib/`, the Halide distribution every app in `apps/`
links against. It needs Linux x86-64, a C++17 compiler, GNU make, pkg-config,
[uv](https://docs.astral.sh/uv/), jemalloc, and the libpng and libjpeg
headers (Halide's Makefile checks for them):

```shell
sudo apt install build-essential cmake pkg-config libpng-dev libjpeg-turbo8-dev zlib1g-dev \
    libjemalloc2 libboost-dev libtbb-dev
inductive_artifact/setup.sh
# With the Python and GPU baselines (scipy, IPP, oneTBB, torch, Triton, mamba_ssm):
WITH_PYTHON_BASELINES=1 inductive_artifact/setup.sh
```

Two baselines are installed by hand: Julia, for the RNG rows, and parasail,
for the alignment rows. Julia's installer puts `julia` on the `PATH` for new
shells; parasail is found through `PARASAIL_DIR`:

```shell
curl -fsSL https://install.julialang.org | sh -s -- --default-channel 1.12
curl -L https://github.com/jeffdaily/parasail/archive/refs/tags/v2.6.2.tar.gz | tar xz
cmake -S parasail-2.6.2 -B parasail-2.6.2/build -DCMAKE_BUILD_TYPE=Release \
    -DBUILD_SHARED_LIBS=OFF -DCMAKE_POLICY_VERSION_MINIMUM=3.5
cmake --build parasail-2.6.2/build --target parasail -j
export PARASAIL_DIR=$PWD/parasail-2.6.2
```

### Optional: Docker

As an alternative to the steps above, `inductive_artifact/Dockerfile` builds
an image with all of them done: the apt packages, Julia, parasail and the
Halide build. From the repository root:

```shell
# CPU
docker build -f inductive_artifact/Dockerfile -t inductive-artifact .
docker run --rm -it inductive-artifact inductive_artifact/basic_test.sh
docker run --rm -it inductive-artifact inductive_artifact/run_benchmarks.sh
# GPU, with the Python and GPU baselines (needs the NVIDIA Container Toolkit)
docker build -f inductive_artifact/Dockerfile -t inductive-artifact-gpu \
    --build-arg BASE=nvidia/cuda:13.0.2-devel-ubuntu24.04 --build-arg PYTHON_BASELINES=1 .
docker run --rm -it --gpus all inductive-artifact-gpu inductive_artifact/run_benchmarks.sh
```

Results are written inside the container, under `/artifact/inductive_artifact/results`;
mount a host directory there (`-v $PWD/results:/artifact/inductive_artifact/results`)
to keep them.

### Basic test

To check the installation, `inductive_artifact/basic_test.sh` runs Halide's
own tests of inductive functions with Halide's Makefile: the correctness
tests of inductive functions, storage folding and sliding windows, and the
error tests of schedules Halide must reject. It ends with `Basic test passed.`

## Running

```shell
inductive_artifact/run_benchmarks.sh   # -> results/benchmarks.md
inductive_artifact/run_ablation.sh     # -> results/ablation*.{md,txt}
```

Both take `--only app,app` (`cpu_rng`, `cpu_biquads`, `cpu_alignment`,
`viterbi`, `ode`, `prefixsum`, `chebyshev`, `cuda_mamba2`, `flash_attention`;
the ablation covers the six CPU apps). Rows are cached in
`apps/inductive_benchmarks_logs/`, so a partial run updates the table.

The GPU ablation levels come from the GPU apps' own runners:

```shell
make -C apps/cuda_mamba2 SCAN=inductive WMMA=true ABLATE_FUSION=1 test test_bwd
make -C apps/cuda_attention test
```

Optional baselines are used when found:

| Variable | Baseline |
|---|---|
| `BOOST_DIR` (or `libboost-dev`) | Boost.odeint (ODE) |
| `TBB_DIR` (or `libtbb-dev`, or `tbb-devel` in the venv) | oneTBB (prefix mean) |
| `IPP_DIR` (default: the venv) | Intel IPP (biquads) |
| `PARASAIL_DIR` (see Setup) | parasail (alignment) |
| `HB_JULIA` (or `julia` on the `PATH`; see Setup) | Julia (rng) |
| `HB_PYTHON` (default: the venv's python) | scipy, torch, Triton, mamba_ssm |

`HB_NCORES`, `HB_PIN_CORE` and `HB_JEMALLOC` override the core count, the
core single-thread rows are pinned to, and the jemalloc library.

## Expected results

The paper machine's outputs are `apps/inductive_benchmarks_results.md` and
`apps/inductive_ablation_results/`. The peak-heap columns depend only on the
problem sizes, so they should match exactly; the ODE and Viterbi all-cores
rows run one batch element per core, so theirs scale with the core count
(`HB_NCORES=32` gives the paper's sizes). The time ratios depend on the
machine: the all-cores rows are memory-bound, so they shrink on a machine with
more memory bandwidth per core. `appendix/artifact_appendix.tex` lists the claims to
check.
