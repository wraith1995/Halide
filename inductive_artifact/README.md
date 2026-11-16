# Inductive Funcs: artifact

Scripts to build this fork of Halide and rerun the paper's benchmark table
and optimization ablation. The CGO artifact appendix is in `appendix/`
(`latexmk -pdf main.tex` builds a standalone copy; `appendix.tex` is the
part to `\input` into the paper).

## Setup

Linux x86-64, a C++17 compiler, GNU make, pkg-config,
[uv](https://docs.astral.sh/uv/), jemalloc, and the libpng and libjpeg
headers (Halide's Makefile checks for them):

```shell
sudo apt install build-essential pkg-config libpng-dev libjpeg-turbo8-dev zlib1g-dev \
    libjemalloc2 libboost-dev libtbb-dev
inductive_artifact/setup.sh
# With the Python and GPU baselines (scipy, IPP, oneTBB, torch, Triton, mamba_ssm):
WITH_PYTHON_BASELINES=1 inductive_artifact/setup.sh
```

`setup.sh` installs LLVM 22 into `.venv` and builds `distrib/`, the Halide
distribution every app in `apps/` links against.

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
| `PARASAIL_DIR` | parasail (alignment) |
| `HB_JULIA` (or `julia` on the `PATH`) | Julia (rng) |
| `HB_PYTHON` (default: the venv's python) | scipy, torch, Triton, mamba_ssm |

`HB_NCORES`, `HB_PIN_CORE` and `HB_JEMALLOC` override the core count, the
core single-thread rows are pinned to, and the jemalloc library.

## Expected results

The paper machine's outputs are `apps/inductive_benchmarks_results.md` and
`apps/inductive_ablation_results/`. The peak-heap columns are properties of
the programs and should match on any machine. The time ratios depend on the
machine: the all-cores rows are memory-bound, so they shrink on a machine with
more memory bandwidth per core. `appendix/appendix.tex` lists the claims to
check.
