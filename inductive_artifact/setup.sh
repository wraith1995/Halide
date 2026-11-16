#!/usr/bin/env bash
# Builds the Halide distribution (distrib/) that every benchmark app links
# against, with the LLVM that Halide's CI uses, installed by uv into .venv.
#
#   inductive_artifact/setup.sh            # Halide only
#   WITH_PYTHON_BASELINES=1 inductive_artifact/setup.sh
#                                          # also scipy, numpy, IPP, oneTBB,
#                                          # torch, Triton and mamba_ssm
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
JOBS=${JOBS:-$(nproc)}

command -v uv >/dev/null || { echo "uv is needed: https://docs.astral.sh/uv/"; exit 1; }
# A Python 3.12 environment (uv downloads that Python if the system has
# another) holding only the LLVM build Halide's CI uses.
[ -x .venv/bin/python ] || uv venv --python 3.12 .venv
# shellcheck disable=SC1091
source .venv/bin/activate
uv pip install --index https://pypi.halide-lang.org/simple "halide-llvm~=22.1.0"
LLVM_PREFIX=$(halide-llvm --prefix)

make -j"$JOBS" LLVM_CONFIG="$LLVM_PREFIX/bin/llvm-config" distrib

if [ "${WITH_PYTHON_BASELINES:-0}" = 1 ]; then
    uv pip install -r inductive_artifact/requirements.txt \
        --extra-index-url https://download.pytorch.org/whl/cu130
    # mamba_ssm compiles CUDA extensions against the torch just installed.
    uv pip install --no-build-isolation "mamba_ssm==2.3.2.post1"
fi

echo "Halide distribution: $ROOT/distrib"
echo "clang for cpu_biquads: $LLVM_PREFIX/bin/clang++"
