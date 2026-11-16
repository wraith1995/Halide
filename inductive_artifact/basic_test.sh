#!/usr/bin/env bash
# Checks the installation by running Halide's own tests of inductive
# functions, storage folding and sliding windows (test/correctness), and of
# the schedules it must reject (test/error), with Halide's Makefile against
# the build setup.sh made. Prints "Basic test passed." when all pass.
set -euo pipefail
ROOT=$(cd "$(dirname "$0")/.." && pwd)
cd "$ROOT"
JOBS=${JOBS:-$(nproc)}
# shellcheck disable=SC1091
source .venv/bin/activate
LLVM_PREFIX=$(halide-llvm --prefix)

correctness=$(cd test/correctness && ls inductive*.cpp fold_inductive*.cpp fold_invariant_to_split.cpp \
    storage_folding.cpp slide_*.cpp recursive_box_filters.cpp | sed 's/\.cpp$//')
errors=$(cd test/error && ls inductive_*.cpp reorder_inductive_split.cpp | sed 's/\.cpp$//')
echo "Running $(wc -w <<<"$correctness") correctness and $(wc -w <<<"$errors") error tests."
# quiet_correctness_X prints a dot per passing test and the output of any
# failing one; error_X passes when the test stops with a Halide error.
make -j"$JOBS" -k LLVM_CONFIG="$LLVM_PREFIX/bin/llvm-config" \
    $(sed 's/^/quiet_correctness_/' <<<"$correctness") \
    $(sed 's/^/error_/' <<<"$errors") > basic_test.log 2>&1 || {
    grep -E "FAILED TEST|Error [0-9]" basic_test.log || tail -20 basic_test.log
    echo "Basic test failed; the full output is in basic_test.log."
    exit 1
}
echo "Basic test passed."
