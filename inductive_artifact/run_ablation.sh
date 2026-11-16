#!/usr/bin/env bash
# The optimization ablation of the CPU experiments: RDom, inductive with
# fusion only, + sliding window, + loop unrolling, + storage folding (the
# paper form). Writes per-level times and memory, then the summary tables
# in the layout of apps/inductive_ablation_results/data/reviewer_*.txt.
# Arguments pass through to apps/inductive_ablation.py (e.g. --only ode).
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
python3 apps/inductive_ablation.py --tag artifact --out "$OUT/ablation.md" "$@" 2>&1 | tee "$OUT/ablation.log"
python3 apps/inductive_ablation_results/reviewer_tables.py "$OUT/ablation" \
    "This machine: $(nproc) logical CPUs, all-cores rows on ${HB_NCORES:-one thread per physical core}." \
    apps/inductive_benchmarks_logs/ablation_artifact.json
