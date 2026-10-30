"""The reviewer's ablation, as asked: (1) fusion only, (2) + sliding window,
(3) + loop unrolling, (4) + storage folding, each cumulative, with RDom (the
paper's non-inductive baseline) beside it; and the memory of the paper form
against the third-party implementations.

usage: reviewer_tables.py OUT_PREFIX MACHINE_NOTE JSON [JSON ...] [--gpu GPU.json]
"""
import json
import sys

args = sys.argv[1:]
gpu = None
if "--gpu" in args:
    i = args.index("--gpu")
    gpu = json.load(open(args[i + 1]))
    del args[i:i + 2]
prefix, note, srcs = args[0], args[1], args[2:]

cfgs = []
for s in srcs:
    cfgs += json.load(open(s)).values()

NAMES = {"cpu_rng": "rng", "cpu_biquads": "biquads", "cpu_alignment": "alignment",
         "prefixsum": "prefix mean", "viterbi": "viterbi", "ode": "ODE"}
ORDER = ["cpu_rng", "cpu_biquads", "cpu_alignment", "prefixsum", "viterbi", "ode"]
# The paper form unrolls only in these; elsewhere step 3 adds nothing.
UNROLLS = {"cpu_rng", "prefixsum"}

big, small = {}, {}
for c in cfgs:
    by = {r["level"]: r for r in c["rows"] if r["ms"] is not None and r["kind"] == "halide"}
    base = [r for r in c["rows"] if r["ms"] is not None and r["kind"] == "baseline"]
    if "fusion" in by:
        small[c["app"]] = by
    elif "all cores" in c["config"] or ("threads" in c["config"] and "1 thread" not in c["config"]):
        big[c["app"]] = (by, base, c["config"])


def render(rows):
    w = [max(len(r[i]) for r in rows) for i in range(len(rows[0]))]
    out = []
    for k, r in enumerate(rows):
        out.append("  ".join(c.ljust(w[i]) if i == 0 else c.rjust(w[i]) for i, c in enumerate(r)).rstrip())
        if k == 0:
            out.append("  ".join("-" * x for x in w))
    return "\n".join(out) + "\n"


def ms(v):
    return f"{v:.1f} ms" if v >= 10 else (f"{v:.2f} ms" if v >= 0.1 else f"{v * 1000:.1f} us")


def mb(v):
    if v is None:
        return "-"
    return "0" if v < 0.05 else (f"{v:.1f}" if v < 10 else f"{v:.0f}")


HEAD = ["Experiment", "RDom (non-inductive)", "(1) Fusion only*", "(2) + Sliding window",
        "(3) + Loop unrolling", "(4) + Storage folding = paper"]
t_rows, m_rows = [HEAD], [["Experiment", "RDom (non-inductive)", "(1) Fusion only*",
                           "(2) + Sliding window", "(3) + Loop unrolling",
                           "(4) + Storage folding = paper"]]
notes = []
for a in ORDER:
    if a not in big:
        continue
    by, _, cfg = big[a]
    p = by["fold"]["ms"]
    sm = small.get(a)
    fus_t = f"{sm['fusion']['ms'] / sm['fold']['ms']:.0f}x" if sm else "not run"
    fus_m = f"{mb(sm['fusion']['heap'])} (vs {mb(sm['fold']['heap'])})" if sm else "not run"
    s3 = by.get("slide+unroll") if a in UNROLLS else None
    t_rows.append([NAMES[a], f"{by['rdom']['ms'] / p:.2f}x", fus_t, f"{by['slide']['ms'] / p:.2f}x",
                   f"{s3['ms'] / p:.2f}x" if s3 else "= (2), no unroll", f"1.00x ({ms(p)})"])
    m_rows.append([NAMES[a], mb(by["rdom"]["heap"]), fus_m, mb(by["slide"]["heap"]),
                   mb(s3["heap"]) if s3 else "= (2)", mb(by["fold"]["heap"])])
    if "slide+fold" in by:
        notes.append(f"{NAMES[a]}: slide + fold without the unroll is {by['slide+fold']['ms'] / p:.2f}x.")

if gpu:
    for name, g in gpu.items():
        p = g["paper"]
        def r(k):
            v = g.get(k)
            return "n/a" if v is None else (v if isinstance(v, str) else f"{v / p:.2f}x")
        def h(k):
            v = g.get(k + "_heap")
            return "n/a" if v is None else (v if isinstance(v, str) else mb(v))
        t_rows.append([name, r("rdom"), r("fusion"), r("slide"), r("slide_unroll"), f"1.00x ({ms(p / 1000)})"])
        m_rows.append([name, h("rdom"), h("fusion"), h("slide"), h("slide_unroll"), mb(g["paper_heap"])])

# Paper form against third-party implementations: the same resident-memory
# growth measurement for both, and the time ratio for context.
v_rows = [["Experiment", "Paper form: time", "RSS growth (MB)", "Third-party implementation",
           "Time vs paper", "RSS growth (MB)"]]
for a in ORDER:
    if a not in big:
        continue
    by, base, _ = big[a]
    p = by["fold"]
    first = True
    for b in base or [None]:
        row = [NAMES[a] if first else "", ms(p["ms"]) if first else "", mb(p["growth"]) if first else ""]
        if b is None:
            row += ["(none)", "", ""]
        else:
            row += [b["name"], f"{b['ms'] / p['ms']:.2f}x", mb(b["growth"])]
        v_rows.append(row)
        first = False
if gpu:
    for name, g in gpu.items():
        first = True
        for bn, (bt, bm) in g.get("baselines", {}).items():
            v_rows.append([name if first else "", ms(g["paper"] / 1000) if first else "",
                           (mb(g["paper_heap"]) + " (device heap)") if first else "",
                           bn, f"{bt / g['paper']:.2f}x", f"{mb(bm)} (device)"])
            first = False

hdr = (f"{note}\n"
       "Time: each step's time / the paper form's time, at the paper's sizes on all cores.\n"
       "* Fusion only recomputes the recurrence from its start for every output, O(T^2), so it is\n"
       "  measured at a small size on one thread, against the paper form at that same size.\n"
       "= (2): the paper form does not unroll this walk, so step (3) is the step (2) schedule.\n")
open(prefix + "_time.txt", "w").write(hdr + "\n" + render(t_rows) + ("\n" + "\n".join(notes) + "\n" if notes else ""))
open(prefix + "_memory.txt", "w").write(
    f"{note}\nPeak Halide heap (MB) at each step; fusion only at the small size (paper form's heap there in brackets).\n\n"
    + render(m_rows))
open(prefix + "_vs_thirdparty.txt", "w").write(
    f"{note}\nPaper form vs third-party implementations. CPU: resident-memory growth during the timed runs,\n"
    "the same measurement for every implementation. GPU: Halide's peak device heap vs the torch allocator's\n"
    "growth during the call.\n\n" + render(v_rows))
