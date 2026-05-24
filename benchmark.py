import time, sys
sys.stdout.flush()

from deep_cfr import DeepCFRTrainer

print("Warming up (2 iters)...", flush=True)
trainer = DeepCFRTrainer(adv_buffer_size=10000, strat_buffer_size=10000)
trainer.train(n_iterations=2, k_traversals=20)

print("Benchmarking 10 iterations...", flush=True)
t0 = time.time()
trainer.train(n_iterations=10, k_traversals=20)
elapsed = time.time() - t0

ms = elapsed / 10 * 1000
print(f"\nResult: {ms:.0f}ms per iteration", flush=True)
print(f"Baseline (pure Python): ~1500ms", flush=True)
print(f"Speedup: {1500/ms:.1f}x", flush=True)
