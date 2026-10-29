#!/usr/bin/env python3
"""Optimization ablation of the inductive-Func benchmarks, with memory.

For each app and configuration, the Halide forms at cumulative levels of
optimization, and the third-party baselines:

    rdom           the same algorithm as update definitions (materializing,
                   not fused)
    fusion         the inductive Func computed inside its consumer's loop and
                   stored there: nothing slides, so every iteration recomputes
                   the history from the base case. Quadratic in the
                   recurrence length, so measured in a separate, smaller
                   configuration, beside the other levels at that size.
    slide          + sliding window: one new slice per iteration, the whole
                   history kept (storage pinned to the full extent)
    slide+unroll   + the unrolled recurrence step, still unfolded, where
                   the shipped schedule unrolls one (rng, prefixsum)
    slide+fold     + storage folding without the unroll, for the same apps,
                   so the unroll and the fold are each measured alone
    fold           the shipped inductive form: every optimization

Each row has its time, the Halide profiler's peak heap (Halide forms), and
the resident memory it touched while running (rss growth: peak RSS over the
timed runs less the RSS when they started, measured the same way for every
row; see apps/support/bench_harness.h).

    apps/inductive_ablation.py [--only app,app] [--out file.md] [--tag name]

Uses the benchmark driver's environment (apps/inductive_benchmarks.py):
jemalloc preloaded, pinning, and the HB_* overrides for paths and cores.
HALIDE_DISTRIB_PATH, and optionally TBB_DIR, BOOST_DIR, SUNDIALS_DIR,
IPP_DIR, PARASAIL_DIR, are passed through to the app Makefiles.
"""
import argparse
import json
import os
import re
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent))
import inductive_benchmarks as ib  # noqa: E402
from inductive_benchmarks import APPS, LOGS, NCORES, sh, us_row  # noqa: E402

MAKE_VARS = " ".join(f"{k}={os.environ[k]}" for k in
                     ["HALIDE_DISTRIB_PATH", "TBB_DIR", "BOOST_DIR", "SUNDIALS_DIR", "IPP_DIR", "PARASAIL_DIR"]
                     if os.environ.get(k))
# The distrib may not carry the autoscheduler plugins; no app here uses one.
AOT_VARS = MAKE_VARS + " AUTOSCHEDULER="
# The Finding Fast Filters library is written for AVX-512; without it the
# baseline runs through the library's AVX2 shim, and its row says so.
AVX512 = "avx512f" in Path("/proc/cpuinfo").read_text()
FFF_NAME = "Finding Fast Filters strided cascade" + ("" if AVX512 else " (AVX2 port)")


def kb_mb(kb):
    return None if kb is None or kb < 0 else kb / 1024.0


def hb_full(text):
    """bench_harness rows: {variant: dict(ms, heap, growth, peak)} in ms and MB."""
    rows = {}
    pat = re.compile(r"^  (\S.*?)\s{2,}([\d.]+)\s+([\d.]+)\s.*?state_bytes=([\d.eE+]+)"
                     r"(?:\s+rss_growth_kb=(-?\d+)\s+rss_peak_kb=(-?\d+))?", re.M)
    for m in pat.finditer(text):
        rows[m.group(1).strip()] = dict(
            ms=float(m.group(2)), heap=float(m.group(4)) / 1e6,
            growth=kb_mb(int(m.group(5))) if m.group(5) else None,
            peak=kb_mb(int(m.group(6))) if m.group(6) else None)
    return rows


def mem_lines(text):
    """'  mem <label>: rss_growth_kb=G  rss_peak_kb=P' -> {label: (growth MB, peak MB)}."""
    out = {}
    for m in re.finditer(r"^\s+mem (.+?): rss_growth_kb=(-?\d+)\s+rss_peak_kb=(-?\d+)", text, re.M):
        out[m.group(1).strip()] = (kb_mb(int(m.group(2))), kb_mb(int(m.group(3))))
    return out


def row(level, name, ms, heap=None, growth=None, peak=None, kind="halide"):
    return dict(level=level, name=name, ms=ms, heap=heap, growth=growth, peak=peak, kind=kind)


def from_hb(rows, label, level, kind="halide", name=None, prefix=False):
    if prefix:
        label = next((k for k in rows if k.startswith(label)), label)
    r = rows.get(label)
    if r is None:
        return row(level, name or label, None, kind=kind)
    return row(level, name or label, r["ms"], r["heap"] if kind == "halide" else None, r["growth"], r["peak"], kind)


def from_us(text, mems, label, level, kind="halide", name=None, mem_label=None, add_ms=0.0):
    ms = us_row(text, label)
    g, p = mems.get(mem_label or label, (None, None))
    return row(level, name or label, None if ms is None else ms + add_ms, None, g, p, kind)


def aot_peaks(d, kv, names, tag, pin, cores):
    """Profiler peak heap (MB) of named AOT pipelines, from a profiled build."""
    js = ib.SCRATCH / f"abl_peak_{tag}.json"
    js.unlink(missing_ok=True)
    env = {"HL_PROFILER_JSON_OUTPUT": str(js), "HB_TRIALS": "1", "HB_WARMUP": "0"}
    try:
        sh("make -s clean", d)
        sh(f"make -s {kv} test", d, env=env, pin=pin, cores=cores, log=LOGS / f"abl_peak_{tag}.txt")
    except Exception as e:
        print(f"   peak measurement for {tag} failed: {str(e)[:200]}", flush=True)
        return {}
    return {n: ib.mb(b) for n, b in ib.peak_from_json(js, set(names)).items()}


# ---------------------------------------------------------------------------
# JIT apps (apps/inductive_suite)

def suite_bin(name):
    d = APPS / "inductive_suite"
    sh(f"make -s {MAKE_VARS} all", d)
    return d / "bin" / name


def ode(D, B, T, par, fusion=False):
    b = suite_bin("ode_observer_sparse_fused_test")
    env = {"HB_BATCH": os.environ.get("HB_ABL_ODE_BATCH", "16")} if par else {"HL_NUM_THREADS": "1"}
    if fusion:
        env["HB_FUSION"] = "1"
    out = sh(f"{b} {D} {B} {T}", APPS / "inductive_suite", env=env, pin=not par, cores=par,
             log=LOGS / f"abl_ode_{D}_{B}_{T}_f{int(fusion)}.txt")
    r = hb_full(out)
    rows = [from_hb(r, "non-inductive (materialize)", "rdom")]
    if fusion:
        rows.append(from_hb(r, "inductive FUSION ONLY (no slide)", "fusion"))
    rows += [from_hb(r, "inductive UNFOLDED (fold n -> T)", "slide"),
             row("slide+unroll", "-", None),
             from_hb(r, "inductive FOLDED (fold n -> 2)", "fold"),
             from_hb(r, "Boost.odeint (rk4 init + observer)", "", "baseline", "Boost.odeint"),
             from_hb(r, "fused C++ loop", "", "baseline", "fused C++ loop"),
             from_hb(r, "SUNDIALS N_Vector AB2", "", "baseline", "SUNDIALS N_Vector AB2"),
             from_hb(r, "SUNDIALS ARKODE Heun (fixed h)", "", "baseline", "SUNDIALS ARKODE Heun (fixed h)")]
    return dict(config=f"Allen-Cahn D={D}, batch {B}, T={T}, {'all cores' if par else '1 thread'}", rows=rows)


def viterbi(S, M, T, B, par, fusion=False):
    env = {} if par else {"HL_NUM_THREADS": "1"}
    if fusion:
        env["HB_FUSION"] = "1"
    out = sh(f"{suite_bin('viterbi_log')} {S} {M} {T} {B}", APPS / "inductive_suite", env=env,
             pin=not par, cores=par, log=LOGS / f"abl_viterbi_{S}_{M}_{T}_{B}_f{int(fusion)}.txt")
    r = hb_full(out)
    rows = [from_hb(r, "non-inductive (materialize)", "rdom")]
    if fusion:
        rows.append(from_hb(r, "ablation fusion", "fusion"))
    rows += [from_hb(r, "ablation slide", "slide"), row("slide+unroll", "-", None),
             from_hb(r, "inductive FOLDED (fold t -> 2)", "fold")]
    return dict(config=f"{S} states, {M} symbols, T={T}, {B} decode{'s' if B > 1 else ''}, "
                       f"{'all cores' if par else '1 thread'}", rows=rows)


def prefixsum(W, H, threads, fusion=False):
    d = APPS / "inductive_suite"
    env = {"HL_NUM_THREADS": str(threads)}
    if fusion:
        env["HB_FUSION"] = "1"
    pin, cores = threads == 1, threads > 1
    tag = f"{W}_{H}_t{threads}_f{int(fusion)}"
    ind = hb_full(sh(f"{suite_bin('prefixsum_bench')} {W} {H}", d, env=env, pin=pin, cores=cores,
                     log=LOGS / f"abl_prefixsum_{tag}.txt"))
    rd = hb_full(sh(f"{suite_bin('prefixsum_bench_rdom')} {W} {H}", d, env=env, pin=pin, cores=cores,
                    log=LOGS / f"abl_prefixsum_rdom_{tag}.txt"))
    rows = [from_hb(rd, "non-inductive (RDom, materialize row)", "rdom")]
    if fusion:
        rows.append(from_hb(ind, "ablation fusion", "fusion"))
    rows += [from_hb(ind, "ablation slide", "slide"), from_hb(ind, "ablation slide+unroll", "slide+unroll"),
             from_hb(ind, "ablation slide+fold", "slide+fold"),
             from_hb(ind, "inductive FOLDED", "fold", name="inductive FOLDED", prefix=True)]
    tbb = d / "bin/prefixsum_bench_tbb"
    if tbb.exists():
        tb = hb_full(sh(f"{tbb} {W} {H}", d, env=env, pin=pin, cores=cores, log=LOGS / f"abl_prefixsum_tbb_{tag}.txt"))
        rows += [from_hb(tb, "oneTBB parallel_for rows (serial scan)", "", "baseline", "oneTBB parallel_for rows"),
                 from_hb(tb, "oneTBB parallel_scan (fused)", "", "baseline", "oneTBB parallel_scan")]
    return dict(config=f"{W} x {H} rows, running mean, {threads} thread{'s' if threads > 1 else ''}", rows=rows)


# ---------------------------------------------------------------------------
# AOT apps

def cpu_alignment(Q, Tl, batch, par, fusion=False):
    d = APPS / "cpu_alignment"
    knobs = dict(QLEN=Q, TLEN=Tl, BATCH=batch, PAR=str(par).lower(), ABLATE_FUSION=int(fusion))
    kv = " ".join(f"{k}={v}" for k, v in knobs.items()) + f" HL_TARGET={ib.ALIGN_TARGET} {AOT_VARS}"
    sh("make -s clean", d)
    out = sh(f"make -s {kv} test", d, pin=not par, cores=par,
             log=LOGS / f"abl_alignment_{Q}_{batch}_par{par}_f{int(fusion)}.txt")
    mems = mem_lines(out)
    # Every Halide form emits the alignment ops; turning them into a CIGAR
    # (compaction) is a separate pass, added to each so the rows match the
    # baselines' contract (fill + traceback + CIGAR).
    comp = us_row(out, "compaction") or 0.0
    rows = [from_us(out, mems, "int8 rdom", "rdom", add_ms=comp)]
    if fusion:
        rows.append(from_us(out, mems, "ablation fusion", "fusion", add_ms=comp))
    rows += [from_us(out, mems, "ablation slide", "slide", add_ms=comp), row("slide+unroll", "-", None),
             from_us(out, mems, "int8 ind", "fold", add_ms=comp),
             from_us(out, mems, "ksw2 sse", "", "baseline", "ksw2 (minimap2 kernel)"),
             from_us(out, mems, "parasail", "", "baseline", "parasail"),
             from_us(out, mems, "ksw2 gg", "", "baseline", "ksw2 scalar")]
    peaks = aot_peaks(d, kv.replace(f"HL_TARGET={ib.ALIGN_TARGET}", f"HL_TARGET={ib.ALIGN_TARGET} PROF=-profile"),
                      ["align8_rdom", "align8_fus", "align8_sld", "align8_ind"],
                      f"alignment_{Q}_{batch}_par{par}_f{int(fusion)}", not par, par)
    for r, n in zip([x for x in rows if x["kind"] == "halide" and x["name"] != "-"],
                    ["align8_rdom"] + (["align8_fus"] if fusion else []) + ["align8_sld", "align8_ind"]):
        r["heap"] = peaks.get(n)
    return dict(config=f"{Q}x{Tl} x {batch} pairs, {'all cores' if par else '1 thread'}, fill+traceback+cigar, "
                       f"AVX2", rows=rows)


def cpu_rng(lanes, steps, par, fusion=False):
    d = APPS / "cpu_rng"
    knobs = dict(LANES=lanes, STEPS=steps, PAR=str(par).lower(), ABLATE_FUSION=int(fusion))
    kv = " ".join(f"{k}={v}" for k, v in knobs.items()) + f" {AOT_VARS}"
    sh("make -s clean", d)
    out = sh(f"make -s {kv} test", d, pin=not par, cores=par, log=LOGS / f"abl_rng_{lanes}_{steps}_par{par}_f{int(fusion)}.txt")
    mems = mem_lines(out)
    rows = [from_us(out, mems, "rdom", "rdom")]
    if fusion:
        rows.append(from_us(out, mems, "ablation fusion", "fusion"))
    rows += [from_us(out, mems, "ablation slide", "slide"), from_us(out, mems, "ablation slide+unroll", "slide+unroll"),
             from_us(out, mems, "ablation slide+fold", "slide+fold"), from_us(out, mems, "inductive", "fold"),
             from_us(out, mems, "scalar C++", "", "baseline", "scalar C++ loop")]
    if ib.JULIA.exists():
        threads = f"-t {NCORES}" if par else ""
        jo = sh(f"{ib.JULIA} {threads} julia_bench.jl {2 * lanes} {steps}", d, pin=not par, cores=par, check=False,
                log=LOGS / f"abl_rng_julia_{lanes}_{steps}_par{par}.txt")
        rows.append(from_us(jo, mem_lines(jo), "julia xoshiro", "", "baseline", "Julia rand!"))
    peaks = aot_peaks(d, kv + " HL_TARGET=host-profile", ["rng_rdom", "rng_fus", "rng_sld", "rng_slu", "rng_slf", "rng_ind"],
                      f"rng_{lanes}_{steps}_par{par}_f{int(fusion)}", not par, par)
    for r in rows:
        n = {"rdom": "rng_rdom", "fusion": "rng_fus", "slide": "rng_sld", "slide+unroll": "rng_slu",
             "slide+fold": "rng_slf",
             "fold": "rng_ind"}.get(r["level"]) if r["kind"] == "halide" else None
        if n:
            r["heap"] = peaks.get(n)
    return dict(config=f"{lanes} streams x {steps} steps, {'all cores' if par else '1 thread'}", rows=rows)


def cpu_biquads(sections, channels, samples, par, fusion=False):
    d = APPS / "cpu_biquads"
    knobs = dict(SECTIONS=sections, CHANNELS=channels, SAMPLES=samples, PAR=str(par).lower(),
                 ABLATE_FUSION=int(fusion))
    kv = " ".join(f"{k}={v}" for k, v in knobs.items()) + f" {AOT_VARS}"
    sh("make -s clean", d)
    out = sh(f"make -s {kv} test", d, pin=not par, cores=par,
             log=LOGS / f"abl_biquads_{channels}_{samples}_par{par}_f{int(fusion)}.txt")
    mems = mem_lines(out)
    rows = [from_us(out, mems, "rdom", "rdom")]
    if fusion:
        rows.append(from_us(out, mems, "ablation fusion", "fusion"))
    rows += [from_us(out, mems, "ablation slide", "slide"), row("slide+unroll", "-", None),
             from_us(out, mems, "inductive", "fold"),
             from_us(out, mems, "fff", "", "baseline", FFF_NAME),
             from_us(out, mems, "ipp", "", "baseline", "Intel IPP ippsIIR_32f_P")]
    if ib.VENV.exists() and not par:
        so = sh(f"{ib.VENV} sosfilt_bench.py {channels} {samples} {sections}", d, pin=True,
                log=LOGS / f"abl_biquads_scipy_{channels}_{samples}.txt")
        rows.append(from_us(so, mem_lines(so), "scipy.sosfilt", "", "baseline", "scipy.sosfilt"))
    peaks = aot_peaks(d, kv + " HL_TARGET=host-profile", ["biquads_rdom", "biquads_fus", "biquads_sld", "biquads_ind"],
                      f"biquads_{channels}_{samples}_par{par}_f{int(fusion)}", not par, par)
    for r in rows:
        n = {"rdom": "biquads_rdom", "fusion": "biquads_fus", "slide": "biquads_sld",
             "fold": "biquads_ind"}.get(r["level"]) if r["kind"] == "halide" else None
        if n:
            r["heap"] = peaks.get(n)
    return dict(config=f"{sections} sections, {channels} ch x {samples} samples, {'all cores' if par else '1 thread'}",
                rows=rows)


# ---------------------------------------------------------------------------

SUITE = [
    ("cpu_rng", lambda: cpu_rng(32, 1 << 22, False)),
    ("cpu_rng", lambda: cpu_rng(1024, 131072, True)),
    ("cpu_rng", lambda: cpu_rng(32, 4096, False, fusion=True)),
    ("cpu_biquads", lambda: cpu_biquads(8, 32, 8 << 20, False)),
    ("cpu_biquads", lambda: cpu_biquads(8, 1024, 1 << 18, True)),
    ("cpu_biquads", lambda: cpu_biquads(8, 32, 1024, False, fusion=True)),
    ("ode", lambda: ode(1024, 1, 32768, False)),
    ("ode", lambda: ode(1024, NCORES, 8192, True)),
    ("ode", lambda: ode(1024, 1, 512, False, fusion=True)),
    ("viterbi", lambda: viterbi(16, 4, 320000, 1, False)),
    ("viterbi", lambda: viterbi(16, 4, 262144, NCORES, True)),
    ("viterbi", lambda: viterbi(16, 4, 2000, 1, False, fusion=True)),
    ("prefixsum", lambda: prefixsum(1 << 26, 2, 1)),
    ("prefixsum", lambda: prefixsum(1 << 23, 32, NCORES)),
    ("prefixsum", lambda: prefixsum(1 << 16, 2, 1, fusion=True)),
    ("cpu_alignment", lambda: cpu_alignment(1024, 1024, 128, False)),
    ("cpu_alignment", lambda: cpu_alignment(1024, 1024, 4096, True)),
    ("cpu_alignment", lambda: cpu_alignment(128, 128, 128, False, fusion=True)),
]

LEVELS = ["rdom", "fusion", "slide", "slide+unroll", "slide+fold", "fold"]


def fmt(x, nd=3):
    if x is None:
        return "-"
    return f"{x:.{nd}f}" if x < 100 else f"{x:.1f}"


def render(results, tag):
    out = [f"# Inductive-Func optimization ablation ({tag})", "",
           "Times in ms (best of the protocol's trials). Heap: the Halide profiler's peak internal heap. "
           "RSS growth: resident memory touched during the timed runs beyond what was resident when they "
           "started (inputs, outputs, code), the same measurement for Halide forms and baselines. "
           "The fusion level is quadratic in the recurrence length, so it is measured in its own smaller "
           "configuration, beside the other levels at that size.", ""]
    for r in results:
        out += [f"## {r['app']}: {r['config']}", "",
                "| Level / implementation | Time (ms) | vs fold | Heap (MB) | RSS growth (MB) | Peak RSS (MB) |",
                "|---|---:|---:|---:|---:|---:|"]
        fold = next((x["ms"] for x in r["rows"] if x["level"] == "fold"), None)
        for x in r["rows"]:
            if x["kind"] == "halide" and x["level"] == "slide+unroll" and x["name"] == "-":
                out.append("| slide+unroll | n/a | | | | |")
                continue
            label = (x["level"] if x["level"] != "fold" else "fold (shipped inductive form)") \
                if x["kind"] == "halide" else x["name"]
            vs = "-" if x["ms"] is None or not fold else f"{x['ms'] / fold:.2f}x"
            out.append(f"| {label} | {fmt(x['ms'])} | {vs} | {fmt(x['heap'], 2)} | {fmt(x['growth'], 2)} | "
                       f"{fmt(x['peak'], 1)} |")
        out.append("")
    return "\n".join(out)


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("--only", help="comma-separated app names")
    ap.add_argument("--configs", help="comma-separated SUITE indices")
    ap.add_argument("--tag", default=os.uname().nodename)
    ap.add_argument("--out", default=None)
    args = ap.parse_args()
    LOGS.mkdir(exist_ok=True)
    ib.SCRATCH.mkdir(parents=True, exist_ok=True)
    if not Path(ib.JEMALLOC).exists():
        sys.exit(f"{ib.JEMALLOC} is needed (set HB_JEMALLOC)")
    only = set(args.only.split(",")) if args.only else None
    configs = {int(i) for i in args.configs.split(",")} if args.configs else None
    cache_path = LOGS / f"ablation_{args.tag}.json"
    cache = json.loads(cache_path.read_text()) if cache_path.exists() else {}
    for idx, (name, fn) in enumerate(SUITE):
        if (only and name not in only) or (configs is not None and idx not in configs):
            continue
        print(f"== {name} [{idx}] ==", flush=True)
        try:
            r = fn()
        except Exception as e:
            print(f"   FAILED: {str(e)[:600]}", flush=True)
            r = dict(config="(failed)", rows=[])
        r["app"] = name
        cache[f"{idx}:{name}"] = r
        cache_path.write_text(json.dumps(cache, indent=1))
        for x in r["rows"]:
            print(f"   {x['level'] or x['name']:<14} {fmt(x['ms'])} ms  growth {fmt(x['growth'], 2)} MB", flush=True)
    results = [cache[f"{i}:{n}"] for i, (n, _) in enumerate(SUITE) if f"{i}:{n}" in cache]
    text = render(results, args.tag)
    out = Path(args.out) if args.out else APPS / f"inductive_ablation_{args.tag}.md"
    out.write_text(text + "\n")
    print(f"wrote {out}")


if __name__ == "__main__":
    main()
