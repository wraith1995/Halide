# Inductive-Func ablation results

Measurements on the paper machine (AMD Threadripper 9970X, 32 cores; NVIDIA
RTX 5060 Ti), with the paper's protocol: jemalloc preloaded with
`oversize_threshold:0,dirty_decay_ms:-1,muzzy_decay_ms:-1,thp:always`,
`taskset -c 0-31`, `HL_NUM_THREADS=32`, best of 30 trials.

Each entry is that form's time over the paper form's time, at the paper's
sizes on all cores.

```
Experiment        RDom    Inductive, fusion only*  + Sliding window  + Folding (= paper)
----------------  ------  -----------------------  ----------------  -------------------
rng               5.55x   6038x                    4.75x             1.00x (18.1 ms) †
biquads           6.51x   2787x                    5.69x             1.00x (29.5 ms)
alignment         13.12x  62x                      13.88x            1.00x (26.4 ms)
prefix mean       2.06x   490x                     1.73x             1.00x (23.4 ms) †
viterbi           3.56x   988x                     1.34x             1.00x (12.0 ms)
ODE (worst of 3)  19.52x  282x                     38.57x            1.00x (1.39 ms)
mamba2 fwd (GPU)  1.15x   3.12x                    1.04x             1.00x (1.57 ms)
mamba2 bwd (GPU)  1.03x   4.13x                    1.05x             1.00x (5.88 ms)
flash attn (GPU)  1.10x   n/a                      n/a               1.00x (0.40 ms) †
```

\* The inductive version with fusion alone, no sliding window or folding. It
recomputes the recurrence from its start for every output, O(T^2), so it is
measured at a small size on one thread, against the paper form at that size.

† The paper form also unrolls the loop. Folding without the unroll is 1.10x
(rng), 1.00x (prefix mean) and 1.01x (flash attention).

Every row but ODE is a single run of `apps/inductive_ablation.py`. Across
other runs the RDom ratio varies: biquads 6.3-7.6x, alignment 14.0-14.8x,
viterbi 3.36-3.61x, rng 5.82-5.84x (the last two on the paper-commit source).

## ODE

The ODE row is `ode_paper_close/ode_paperclose.cpp`: the paper commit's
(833d45328) `apps/inductive_suite/ode_observer_sparse_fused_test.cpp` with
one schedule line added to the observer of every form,

```cpp
E.reorder(n, b).vectorize(n, 16).parallel(b);
```

which schedules the output's zero fill. Unscheduled, it runs serially on
one thread with b innermost, striding 32 KB per store across the n-dense
output, and costs about 1.3 ms. The file also builds the fusion-only form
when `HB_FUSION` is set; nothing else differs from the paper commit.

Built from `apps/inductive_suite` of the paper commit against its Halide:

```
g++ -O3 -march=native -ffast-math -std=c++17 -I$BOOST -I$HALIDE/include -I$HALIDE/tools \
    ode_paperclose.cpp -L$HALIDE/lib -lHalide -lpthread -ldl -Wl,-rpath,$HALIDE/lib -o ode_paperclose
HB_BATCH=128 HL_NUM_THREADS=32 taskset -c 0-31 ./ode_paperclose 1024 32 8192    # paper config
HB_FUSION=1 HL_NUM_THREADS=1 taskset -c 0 ./ode_paperclose 1024 1 512           # fusion only
```

`paperclose128.log` has the three paper-config runs at 128 realizes per
sample, the paper driver's setting (19.52-21.92x over RDom; the paper
reports 19.01x). `paperclose.log` has three runs at 16 realizes per sample
(19.79-23.05x) and the two fusion-only runs.

## Files

- `data/ablation_orig_*.json`: per-level times, peak Halide heap and RSS
  growth from `apps/inductive_ablation.py`. `allcores` and `fusion` are the
  CPU suite; `odefix` and `odefix_fusion` are ODE on this branch's source,
  with the same zero-fill schedule.
- `data/gpu_5060.json`: the GPU rows.
- `data/reviewer_*.txt`: the tables `reviewer_tables.py` renders from them,
  with a loop-unrolling column and memory tables.
- `machine.txt`: the machine's CPU.
