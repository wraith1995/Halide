#!/usr/bin/env bash
# The paper's benchmark table: every app's inductive form, RDom form and
# fastest baseline, with each form's peak heap. Arguments pass through to
# apps/inductive_benchmarks.py (e.g. --only ode,viterbi).
#
# Optional baselines are picked up from the environment: BOOST_DIR (a
# directory holding boost/, for the ODE app), TBB_DIR, IPP_DIR (default:
# the active venv), PARASAIL_DIR, HB_JULIA. The CPU-only apps run without
# a GPU; the GPU rows fail cleanly if there is none.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
# shellcheck disable=SC1091
source .venv/bin/activate
LLVM_PREFIX=$(halide-llvm --prefix)
export HALIDE_DISTRIB_PATH=${HALIDE_DISTRIB_PATH:-$ROOT/distrib}
export CLANGXX=${CLANGXX:-$LLVM_PREFIX/bin/clang++}
export IPP_DIR=${IPP_DIR:-$VIRTUAL_ENV}
[ -n "${TBB_DIR:-}" ] || [ ! -e "$VIRTUAL_ENV/include/oneapi/tbb.h" ] || export TBB_DIR=$VIRTUAL_ENV
OUT=inductive_artifact/results
mkdir -p "$OUT"
python3 apps/inductive_benchmarks.py --out "$OUT/benchmarks.md" "$@" 2>&1 | tee "$OUT/benchmarks.log"
