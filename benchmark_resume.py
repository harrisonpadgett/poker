"""
benchmark_resume_fast.py — Quick, targeted benchmarks for resume bullet points.
Designed to run in ~30s alongside the training process without starving it.
"""
import time, sys, os
import numpy as np

def hline(c="─", n=60): print(c * n)
def section(title):
    print(); hline("═"); print(f"  {title}"); hline("═")

# ─── 1. C++ game engine vs pure-Python (game state operations) ───────────────
section("1. C++ Game State vs. Pure-Python Game State")

import poker_cpp
from poker_env import RoyalState as PyState

N = 10_000

# Python: reset + apply a sequence of actions
t0 = time.perf_counter()
for _ in range(N):
    s = PyState().reset()
    s.apply_action(3)  # Raise-M
    s.apply_action(1)  # Call
    s.apply_action(1)  # Check
    s.apply_action(2)  # Raise-S
    s.apply_action(1)  # Call
t_py = (time.perf_counter() - t0) / N * 1e6  # µs

# C++: same operations
t0 = time.perf_counter()
for _ in range(N):
    s = poker_cpp.RoyalState()
    s.reset()
    s.apply_action(3)
    s.apply_action(1)
    s.apply_action(1)
    s.apply_action(2)
    s.apply_action(1)
t_cpp = (time.perf_counter() - t0) / N * 1e6  # µs

speedup_env = t_py / t_cpp
print(f"  Python game state (5 actions): {t_py:.2f} µs / episode")
print(f"  C++ game state   (5 actions): {t_cpp:.2f} µs / episode")
print(f"  Speedup:                       {speedup_env:.1f}×")

# ─── 2. Multi-threaded traversal (1 vs all cores) ────────────────────────────
section("2. Multi-Threaded C++ Traversal")

from deep_cfr import DeepCFRTrainer

trainer = DeepCFRTrainer(adv_buffer_size=3_000, strat_buffer_size=3_000)
trainer._export_adv_nets()

def make_bufs(t):
    adv, strat = [], []
    for p in range(2):
        for r in range(4):
            b = t.adv_buffers[p][r]
            adv.append((b.states.numpy(), b.targets.numpy(),
                        b.weights.numpy(), b.capacity, b.n_inserted))
    for r in range(4):
        b = t.strat_buffers[r]
        strat.append((b.states.numpy(), b.targets.numpy(),
                      b.weights.numpy(), b.capacity, b.n_inserted))
    return adv, strat

K    = 100   # traversals per measurement (small to be fast)
REPS = 3

def measure_threads(n):
    poker_cpp.set_num_threads(n)
    adv, strat = make_bufs(trainer)
    # warmup
    poker_cpp.run_traversals(20, 1, adv, strat)
    times = []
    for _ in range(REPS):
        t0 = time.perf_counter()
        poker_cpp.run_traversals(K, 1, adv, strat)
        times.append(time.perf_counter() - t0)
    return np.median(times)

n_cores = os.cpu_count() or 8
t1  = measure_threads(1)
t_n = measure_threads(n_cores)
speedup_mt = t1 / t_n

print(f"  1 thread:   {t1:.3f}s for {K} traversals")
print(f"  {n_cores} threads:  {t_n:.3f}s for {K} traversals")
print(f"  Speedup:    {speedup_mt:.2f}×")
print(f"  Efficiency: {speedup_mt / n_cores * 100:.0f}%")

# ─── 3. Regret-Based Pruning ─────────────────────────────────────────────────
section("3. Regret-Based Pruning (RBP) — Tree Reduction")

poker_cpp.set_num_threads(1)  # isolate to single thread for fair comparison
K2   = 100
REPS2 = 4

def measure_iter(iteration):
    adv, strat = make_bufs(trainer)
    times = []
    for _ in range(REPS2):
        t0 = time.perf_counter()
        poker_cpp.run_traversals(K2, iteration, adv, strat)
        times.append(time.perf_counter() - t0)
    return np.median(times)

# Warmup
measure_iter(1)

t_no_prune = measure_iter(1)        # iter=1: pruning disabled (< PRUNE_START_ITER=2000)
t_pruned   = measure_iter(10_000)   # iter=10000: pruning enabled

speedup_rbp   = t_no_prune / t_pruned
reduction_pct = (1 - t_pruned / t_no_prune) * 100

print(f"  No pruning (early iter=1):     {t_no_prune:.3f}s for {K2} traversals")
print(f"  With RBP   (late iter=10000):  {t_pruned:.3f}s for {K2} traversals")
print(f"  Speedup:   {speedup_rbp:.2f}×  ({reduction_pct:.0f}% faster)")
print()
print("  RBP skips subtrees where strategy[a]==0 AND advantage[a] < -0.03")
print("  (= actions more than ~3 chips below expected value)")

# ─── 4. BLAS MLP inference (Accelerate cblas_sgemv) ─────────────────────────
section("4. BLAS-Accelerated MLP Inference (cblas_sgemv vs. scalar loop)")

import ctypes
IN, H, OUT = 126, 256, 5
N_FWD = 50_000

rng = np.random.default_rng(42)
W1 = np.ascontiguousarray(rng.standard_normal((H, IN),  dtype=np.float32))
b1 = rng.standard_normal(H,   dtype=np.float32)
W2 = np.ascontiguousarray(rng.standard_normal((H, H),   dtype=np.float32))
b2 = rng.standard_normal(H,   dtype=np.float32)
Wo = np.ascontiguousarray(rng.standard_normal((OUT, H), dtype=np.float32))
bo = rng.standard_normal(OUT, dtype=np.float32)
x  = rng.standard_normal(IN,  dtype=np.float32)

# Numpy/PyTorch forward (equivalent to C++ scalar loop with compiler optimisation)
import torch
xt  = torch.from_numpy(x).float()
W1t = torch.from_numpy(W1).float()
b1t = torch.from_numpy(b1).float()
W2t = torch.from_numpy(W2).float()
b2t = torch.from_numpy(b2).float()
Wot = torch.from_numpy(Wo).float()
bot = torch.from_numpy(bo).float()

with torch.no_grad():
    # warmup
    for _ in range(1000):
        h1 = torch.relu(W1t @ xt + b1t)
        h2 = torch.relu(W2t @ h1 + b2t)
        _  = Wot @ h2 + bot
    t0 = time.perf_counter()
    for _ in range(N_FWD):
        h1 = torch.relu(W1t @ xt + b1t)
        h2 = torch.relu(W2t @ h1 + b2t)
        _  = Wot @ h2 + bot
t_torch = (time.perf_counter() - t0) / N_FWD * 1e6

# BLAS cblas_sgemv (loaded at runtime, same as C++ does it)
blas = None
for lib in ["/System/Library/Frameworks/Accelerate.framework/Accelerate",
            "libBLAS.dylib", "libblas.dylib"]:
    try:
        blas = ctypes.CDLL(lib)
        _ = blas.cblas_sgemv
        break
    except (OSError, AttributeError):
        blas = None

if blas:
    fn = blas.cblas_sgemv
    fn.restype = None
    fn.argtypes = [ctypes.c_int, ctypes.c_int, ctypes.c_int, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_void_p, ctypes.c_int,
                   ctypes.c_float, ctypes.c_void_p, ctypes.c_int]

    class BlasMLP:
        """Mimics the C++ CppMLP::forward — avoids Python closure scoping issues."""
        def __init__(self):
            self.h1b = np.empty(H,   dtype=np.float32)
            self.h2b = np.empty(H,   dtype=np.float32)
            self.ob  = np.empty(OUT, dtype=np.float32)
            self.h1p = self.h1b.ctypes.data_as(ctypes.c_void_p)
            self.h2p = self.h2b.ctypes.data_as(ctypes.c_void_p)
            self.op  = self.ob.ctypes.data_as(ctypes.c_void_p)
            self.W1p = W1.ctypes.data_as(ctypes.c_void_p)
            self.W2p = W2.ctypes.data_as(ctypes.c_void_p)
            self.Wop = Wo.ctypes.data_as(ctypes.c_void_p)
        def forward(self, xp):
            fn(101, 111, H,   IN, 1., self.W1p, IN, xp,       1, 0., self.h1p, 1)
            self.h1b += b1; np.maximum(self.h1b, 0, out=self.h1b)
            fn(101, 111, H,   H,  1., self.W2p, H,  self.h1p, 1, 0., self.h2p, 1)
            self.h2b += b2; np.maximum(self.h2b, 0, out=self.h2b)
            fn(101, 111, OUT, H,  1., self.Wop, H,  self.h2p, 1, 0., self.op,  1)
            self.ob  += bo

    mlp = BlasMLP()
    xp = x.ctypes.data_as(ctypes.c_void_p)
    for _ in range(1000): mlp.forward(xp)  # warmup

    t0 = time.perf_counter()
    for _ in range(N_FWD):
        mlp.forward(xp)
    t_blas = (time.perf_counter() - t0) / N_FWD * 1e6

    speedup_blas = t_torch / t_blas
    print(f"  PyTorch (baseline):   {t_torch:.2f} µs / forward pass")
    print(f"  BLAS cblas_sgemv:     {t_blas:.2f} µs / forward pass")
    print(f"  Speedup:              {speedup_blas:.2f}×")
    print()
    print("  C++ avoids PyTorch overhead (graph, dispatch, Python interop).")
    print("  BLAS uses Apple AMX/NEON SIMD units for GEMV — same backend PyTorch")
    print("  uses, but with zero Python overhead per call.")
else:
    speedup_blas = None
    print("  BLAS not found — skipping.")
    t_torch = t_blas = None

# ─── 5. Async concurrency gain ───────────────────────────────────────────────
section("5. Async Overlap: Traversal + Training Concurrency")

from concurrent.futures import ThreadPoolExecutor

# Pre-fill buffers minimally so train_adv_network has something to do
trainer2 = DeepCFRTrainer(adv_buffer_size=5_000, strat_buffer_size=5_000)
trainer2._export_adv_nets()

# Fill buffers with synthetic data so training steps are real
for p in range(2):
    for r in range(4):
        buf = trainer2.adv_buffers[p][r]
        buf.states[:200]  = 0.0
        buf.targets[:200] = 0.0
        buf.weights[:200] = 1.0
        buf.n_inserted    = 200
for r in range(4):
    buf = trainer2.strat_buffers[r]
    buf.states[:200]  = 0.0
    buf.targets[:200] = 0.1
    buf.weights[:200] = 1.0
    buf.n_inserted    = 200

poker_cpp.set_num_threads(4)
K3  = 100
N3  = 4

# Sequential
trainer2._export_adv_nets()
t0 = time.perf_counter()
for i in range(N3):
    adv, strat = make_bufs(trainer2)
    poker_cpp.run_traversals(K3, i + 1, adv, strat)
    trainer2.train_adv_network()
t_seq = (time.perf_counter() - t0) / N3

# Async (overlap)
exc = ThreadPoolExecutor(max_workers=1)
trainer2._export_adv_nets()
t0 = time.perf_counter()
for i in range(N3):
    trainer2._export_adv_nets()
    fut = exc.submit(poker_cpp.run_traversals, K3, i + 1, *make_bufs(trainer2))
    trainer2.train_adv_network()
    fut.result()
t_async = (time.perf_counter() - t0) / N3
exc.shutdown(wait=False)

speedup_async = t_seq / t_async
saving_pct    = (1 - t_async / t_seq) * 100

print(f"  Sequential (traverse → train): {t_seq:.3f}s / iter")
print(f"  Async (overlap):               {t_async:.3f}s / iter")
print(f"  Speedup:  {speedup_async:.2f}×  ({saving_pct:.0f}% faster per iteration)")

# ─── Summary ─────────────────────────────────────────────────────────────────
print()
hline("═")
print("  SUMMARY — Speedups over baseline")
hline("═")
print(f"  {'Optimization':<48} {'Speedup'}")
print(f"  {'─'*48} {'─'*8}")
print(f"  {'C++ game engine vs. Python (per game state op)':<48} {speedup_env:.1f}×")
print(f"  {f'Multi-threading (1 → {n_cores} cores)':<48} {speedup_mt:.1f}×")
print(f"  {'Regret-Based Pruning (later iterations)':<48} {speedup_rbp:.2f}×")
print(f"  {'Async traversal + training overlap':<48} {speedup_async:.2f}×")
if speedup_blas:
    print(f"  {'BLAS cblas_sgemv MLP inference':<48} {speedup_blas:.2f}×")
hline("═")
combined = speedup_env * speedup_mt * speedup_rbp * speedup_async
print(f"  Combined (product of independent gains): ~{combined:.0f}×")
print()
