# Times scipy.signal.sosfilt on the same shape and coefficients the runner
# uses. Usage: sosfilt_bench.py [channels samples sections]
import ctypes
import gc
import os
import sys
import time

import numpy as np
from scipy.signal import sosfilt

C, S, N = (int(a) for a in (sys.argv[1:4] + [32, 8 << 20, 8][len(sys.argv) - 1:]))

fs, q = 48000.0, 0.8
sos = np.zeros((N, 6))
for k in range(N):
    f0 = 120.0 * (8000.0 / 120.0) ** (k / (N - 1) if N > 1 else 0.0)
    gain_db = 3.0 if k % 2 else -3.0
    A = 10.0 ** (gain_db / 40)
    w0 = 2 * np.pi * f0 / fs
    alpha = np.sin(w0) / (2 * q)
    b = np.array([1 + alpha * A, -2 * np.cos(w0), 1 - alpha * A])
    a = np.array([1 + alpha / A, -2 * np.cos(w0), 1 - alpha / A])
    sos[k] = np.concatenate([b / a[0], a / a[0]])

x = np.empty((C, S), dtype=np.float32)
rng = np.random.default_rng(1234)
x[:] = rng.standard_normal((C, S), dtype=np.float32)

sos = sos.astype(np.float32)


def status_kb(field):
    """A field of /proc/self/status in kB, or -1 off Linux."""
    try:
        with open("/proc/self/status") as f:
            for line in f:
                if line.startswith(field + ":"):
                    return int(line.split()[1])
    except OSError:
        pass
    return -1


# Resident memory as bench_harness.h measures it: hand freed memory back to
# the kernel (jemalloc's arenas when it is preloaded), note the RSS, reset
# the peak, and read the peak after the runs.
gc.collect()
try:
    mallctl = ctypes.CDLL(None).mallctl
    mallctl.argtypes = [ctypes.c_char_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_void_p, ctypes.c_size_t]
    mallctl(b"arena.4096.purge", None, None, None, 0)
except (AttributeError, OSError):
    pass
rss_start = status_kb("VmRSS")
try:
    with open("/proc/self/clear_refs", "w") as f:
        f.write("5")
except OSError:
    pass
# The shared protocol (apps/support/bench_harness.h): HB_WARMUP untimed
# runs, HB_TRIALS timed ones, the best reported.
for _ in range(int(os.environ.get("HB_WARMUP", 3))):
    y = sosfilt(sos, x, axis=-1)
best = float("inf")
for _ in range(int(os.environ.get("HB_TRIALS", 30))):
    t0 = time.perf_counter()
    y = sosfilt(sos, x, axis=-1)
    best = min(best, time.perf_counter() - t0)
print("  scipy.sosfilt %10.1f us  (dtype %s)" % (best * 1e6, y.dtype))
rss_peak = status_kb("VmHWM")
growth = -1 if rss_start < 0 or rss_peak < 0 else max(0, rss_peak - rss_start)
print("  mem scipy.sosfilt: rss_growth_kb=%d  rss_peak_kb=%d" % (growth, rss_peak))
