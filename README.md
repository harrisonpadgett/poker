# Royal Hold'em Deep CFR Poker AI

A heads-up (two-player) poker AI for **Royal Hold'em** that learns a near-optimal strategy from scratch using **Deep Counterfactual Regret Minimization (Deep CFR)**. No human poker knowledge is baked in — the AI discovers correct strategy entirely through self-play against itself.

The same algorithmic family powers **Libratus** and **Pluribus** — the AI systems that defeated world-champion poker professionals in 2017 and 2019.

---

## Gameplay

![Web UI screenshot — Flop betting round. The player holds K♦ T♣ against a board of A♠ Q♥ J♦. The AI Intelligence Panel on the right shows the AI called with 67.3% probability; the Neural Network Output section shows Call at +2.1 chips advantage over the alternatives.](docs/game_screenshot.png)

*The web interface mid-hand on the Flop. The player (K♦ T♣) faces a board of A♠ Q♥ J♦ — a Broadway-heavy board where the AI has already called a small raise. The right-hand **AI Intelligence Panel** exposes the AI's full internal reasoning: the strategy network assigned 67.3% probability to Call, and the advantage network (the "Brain") confirms calling is worth roughly **+2.1 chips** more than folding on this board.*

---

## What is Royal Hold'em?

Royal Hold'em is a simplified variant of Texas Hold'em that uses only the **20 highest cards** in a standard deck: Tens, Jacks, Queens, Kings, and Aces in all four suits. The rules are otherwise identical to Hold'em:

- Each player receives **2 private hole cards**
- **5 community cards** are revealed across 4 betting rounds: Preflop → Flop (3 cards) → Turn (4th card) → River (5th card)
- Players can **Fold**, **Call**, or **Raise** at any decision point
- The best 5-card hand at showdown wins the pot

Because the deck is so small, premium hands like flushes, straights, and full houses come up constantly, making the game fast-moving and high-variance. Two players start with **100 chips** each with a 1-chip ante from each side.

### Action Space

This implementation uses **5 actions** per decision point:

| Action | Meaning | Raise Amount |
|--------|---------|-------------|
| `Fold` | Give up the pot | — |
| `Call` | Match the opponent's bet (or check for free) | — |
| `Raise-S` | Small raise | +2 chips (preflop/flop), +4 chips (turn/river) |
| `Raise-M` | Medium raise | +4 chips (preflop/flop), +6 chips (turn/river) |
| `Raise-L` | Large raise | +6 chips (preflop/flop), +8 chips (turn/river) |

A maximum of **2 raises** per round is allowed to bound the game tree size.

---

## How the AI Learns: Deep CFR

### The Core Idea — Regret Minimization

Think of it this way: you fold a hand, your opponent flips over their cards, and you realize you would have won. That sick feeling — *I should have called* — is **regret**. If you kept a precise running tally of every regret from every decision you've ever made, and always played the action you most regretted *not* playing, you would converge toward the mathematically optimal strategy over time.

**Counterfactual Regret Minimization (CFR)** formalizes this for both players simultaneously. It computes regret at *every possible decision point in the entire game tree* — not just the hands you actually played. After enough iterations, both players converge to a **Nash Equilibrium**: a strategy where neither player can improve their results by unilaterally changing their behavior. Against a Nash Equilibrium strategy, no opponent can do better than break even in the long run.

**Deep CFR** replaces the lookup table of classical CFR — which would need a separate entry for every possible game state, an intractable number — with **neural networks** that generalize across similar situations. This is the same core insight that makes AlphaGo work, applied to imperfect-information games like poker.

### The Training Loop (One Iteration)

Each iteration consists of two overlapping phases:

#### Phase 1 — Game Tree Traversal (C++ engine, multi-threaded)

**800 games** are simulated per iteration (configurable). For each game:

1. Cards are dealt randomly.
2. The traversal alternates between two roles:
   - **Traverser** (one player per iteration): at their own decision points, the traverser explores **all 5 possible actions** and records what each would have been worth.
   - **Opponent**: samples a single action proportional to their current strategy.
3. At each traverser decision point, the **advantage** of each action is computed:
   ```
   advantage(action) = value(action) − expected_value_of_current_strategy
   ```
   Positive advantage → "I should have done this more." Negative → "I should have done this less."
4. These `(game_state, advantages, iteration_weight)` samples are stored in **reservoir buffers** — one per player per betting round (8 advantage buffers + 4 strategy buffers).

**Regret-Based Pruning (RBP):** When the AI is confident an action is clearly suboptimal (normalized advantage < −0.03, equivalent to ~3 chips below expected value), it skips exploring that subtree entirely. This avoids spending compute on obviously bad plays like folding pocket aces, and significantly speeds up traversal at later training stages.

**Linear CFR Weighting:** Samples from early iterations are given lower weight than recent ones (`weight = iteration^1.5` for advantages, `iteration^2.0` for strategy). The AI's confused early attempts don't dilute what it has recently learned.

#### Phase 2 — Neural Network Training (PyTorch, concurrent)

While C++ traversal runs in a background thread, **12 neural networks** train on the main thread:

- **8 Advantage Networks** (`adv_nets[player][round]`): One per (player × betting round) combination. Each learns to predict, for any game state, which actions are currently over- or under-played relative to the optimum. These are the "Brain" — the AI's working model of regret.
- **4 Strategy Networks** (`strat_nets[round]`): One per betting round. These accumulate the *average* strategy across all iterations — the final, deployable policy the AI uses during play.

All networks share the same basic architecture:

```
Advantage Nets:  122 inputs → 256 → 256 → 5 outputs
Strategy Nets:   122 inputs → 512 → 512 → 5 outputs
                 (larger because they must memorize the full game tree)
```

The **122 input features** encode everything the AI can observe:

| Slice | Size | Description |
|-------|------|-------------|
| `[0]` | 1 bit | Which player is acting (0 or 1) |
| `[1–20]` | 20 bits | Hole cards, one-hot over the 20-card deck |
| `[21–40]` | 20 bits | Visible community cards, one-hot |
| `[41–120]` | 80 bits | Action history: 16 slots × 5 one-hot bits each |
| `[121]` | 1 bit | `is_suited`: 1 if both hole cards share a suit |

No hand-strength features are provided. The network must discover that A-K suited beats Q-J offsuit entirely from the one-hot card encodings and the payoffs it observes during traversal.

---

## Key Engineering Details

### Warm Restarts Every 1,000 Iterations

Every 1,000 iterations, all 8 advantage networks are **completely destroyed and rebuilt from random weights**. This might sound counterproductive, but it solves a real problem: neural networks can get stuck in **local minima** — states where the network has settled on a strategy that's internally consistent but globally mediocre. A network gradually nudged toward a mediocre strategy resists further improvement because every small gradient step makes things locally worse.

By resetting to random weights and immediately running 500 "burn-in" training steps on the existing buffers (which contain rich data from thousands of prior iterations), the network rapidly re-converges to a fresh, globally-aware perspective. Crucially, **strategy buffers do not reset** — they accumulate wisdom across all iterations unaffected by the advantage network resets.

After a warm restart you'll see `ADV LOSS` spike briefly, then fall back below its pre-restart level within ~500 iterations. This is expected and healthy — it signals the network is escaping a local minimum.

### Reservoir Buffers

The advantage and strategy buffers use **reservoir sampling**: when a buffer is full, each new sample replaces a random old one with a probability designed to keep the overall distribution uniform over time. Buffers are capped so that the insertion probability stays at ~50% permanently, ensuring old data doesn't dominate and new experience is always incorporated.

Each of the 8 advantage buffers and 4 strategy buffers holds up to **2,000,000 samples** (totaling ~8.5 GB of buffer RAM at full capacity). River buffers grow much faster than preflop buffers because every hand that reaches the river generates decision points across the entire river betting tree.

### Async Traversal (C++ + Python Concurrency)

C++ game tree traversal runs on a background thread via Python's `ThreadPoolExecutor` while PyTorch trains the networks on the main thread. This means game simulation and gradient updates happen **concurrently**, so the effective time per iteration is approximately:

```
iter_time ≈ max(traversal_time, training_time)
```

instead of their sum, giving roughly 30–50% throughput improvement at typical buffer sizes.

### Atomic Checkpoint Saving

The training state is saved to `checkpoint.pt` every 100 iterations. To prevent corruption if the process is interrupted mid-save, the file is first written to a `.tmp` file and then atomically renamed over the real checkpoint. This means `checkpoint.pt` is always in a valid, loadable state — even if you press Ctrl+C during a save.

### AdamW Optimizer with Gradient Clipping

Advantage networks use **AdamW** (Adam with weight decay, `wd=1e-4`) rather than plain Adam. Weight decay acts as a regularizer between warm restarts, preventing the network from catastrophically overfitting to the last few hundred samples in small buffers. **Gradient clipping** (`max_norm=1.0`) prevents high-weight river samples from occasionally producing catastrophically large gradient updates that would destabilize training.

Strategy networks use plain **Adam** with a **ReduceLROnPlateau** scheduler (factor 0.5, patience 200 iterations), which halves the learning rate when strategy loss stagnates and unfreezes it if it drops too low.

---

## Project Structure

```
poker/
├── poker_env.py          # Python game engine (RoyalState, hand evaluator)
├── deep_cfr.py           # Deep CFR trainer: networks, buffers, training loop
├── train_terminal.py     # Training script with live terminal display
├── app.py                # Flask web interface to play against the AI
├── server.py             # Game server backend
├── cpp/
│   ├── poker_env.h       # C++ game state (RoyalState)
│   ├── poker_env.cpp     # C++ game engine implementation
│   └── bindings.cpp      # C++ traversal engine + pybind11 bindings
├── build/                # CMake build artifacts
├── docs/
│   └── game_screenshot.png
└── checkpoint.pt         # Saved training state (~12 GB at convergence)
```

### Component Roles

| Component | Language | Role |
|-----------|----------|------|
| `poker_env.py` | Python | Reference game engine; hand evaluator; Python `RoyalState` used during subgame solving rollouts |
| `cpp/bindings.cpp` | C++ / pybind11 | High-speed traversal engine; exports network weights into an in-process MLP cache; all multi-threaded game tree work happens here |
| `deep_cfr.py` | Python / PyTorch | 12 neural networks; reservoir buffers; training loops; subgame solver; checkpoint I/O |
| `train_terminal.py` | Python | Training entry point with live terminal display of losses, strategy, and buffer fill |
| `app.py` | Python / Flask | REST API (`/api/start`, `/api/action`, `/api/state`) that powers the web UI |
| `static/script.js` | JavaScript | Single-page game client; renders cards, action buttons, Intelligence Panel |

---

## Training

```bash
# Build the C++ extension first
cd build && cmake --build . && cp poker_cpp*.so ../ && cd ..

# Start or resume training (auto-loads checkpoint.pt if present)
python3 train_terminal.py
```

Training runs indefinitely up to 100,000 iterations. The terminal display updates every iteration:

```
ITER   |   ADV LOSS | STRAT LOSS
--------------------------------
Elapsed: 3.2s  ETA: 285000s
12500  |   532.1847 |     0.8621
  [BUF] AA: F=0.00 C=0.11 RS=0.29 RM=0.36 RL=0.24
  [ADV] AA: F=-3.1 C=-0.0 RS=+0.2 RM=+0.3 RL=+0.1
  [BUF] TT: F=0.00 C=0.08 RS=0.17 RM=0.33 RL=0.42
  [ADV] TT: F=-1.9 C=-0.1 RS=+0.2 RM=+0.1 RL=+0.2
--------------------------------
Buffers: adv[0][0]=7,200,000 adv[1][0]=6,900,000 strat[0]=18,000,000
         adv[0][3]=22,000,000 adv[1][3]=550,000,000 strat[3]=900,000,000
```

**Reading the display:**

| Field | What it means |
|-------|--------------|
| `ADV LOSS` | Mean squared error of the advantage network predictions (in chip² units). Spikes every 1,000 iterations at warm restarts, then falls back. Lower is better — don't panic at spikes. |
| `STRAT LOSS` | Cross-entropy loss of the strategy network. The real convergence signal. Should trend steadily downward over thousands of iterations. |
| `[BUF] AA` | Accumulated average strategy for **Pocket Aces** at preflop. The AI's actual play probabilities if you faced it right now. F=fold, C=call/check, RS/RM/RL=raise sizes. |
| `[ADV] AA` | Advantage network's current estimates for Pocket Aces (in chips). Negative = the Brain thinks this action loses chips. Positive = gains chips. |
| `[BUF] TT` | Same as above but for **Pocket Tens** — a hand where the correct strategy is subtler. |
| `Buffers` | Insertion counts for preflop and river buffers. River grows much faster because every completed hand generates many river decision points. |

### Effort Toggle

`train_terminal.py` exposes an `EFFORT` variable at the top of the file:

```python
EFFORT = 'medium'   # 'high' | 'medium' | 'low'
```

| Level | Threads | Traversals/iter | Speed | Use when |
|-------|---------|-----------------|-------|----------|
| `high` | All cores | 800 | ~3–4s/iter | Overnight training, plugged in |
| `medium` | 4 | 500 | ~3–4s/iter | Active work session, balanced |
| `low` | 2 | 250 | ~3–5s/iter | Background training, laptop cool |

All three levels produce the same trained model — lower effort simply takes proportionally longer.

### Convergence Milestones

| Iterations | What to expect |
|---|---|
| 0–10k | Rapid learning. AA Fold probability drops toward 0%. Raise sizes begin differentiating. |
| 10k–30k | Strategy stabilizes for premium hands. STRAT LOSS falls below 0.85. |
| 30k–50k | Near-Nash equilibrium for preflop and flop. Visible change slows to <1% per 5,000 iters. |
| 50k–80k | Marginal refinement. River strategies converge. |
| 80k+ | Effectively Nash for all practical purposes. |

At ~3.2 seconds per iteration on an M1 MacBook Pro, reaching 50,000 iterations takes approximately **44 hours** of continuous training.

---

## Playing Against the AI

```bash
python3 app.py
# Opens http://localhost:5001
```

The web interface lets you play heads-up Royal Hold'em against the trained AI. Each game:

- Cards are dealt from a fresh shuffled deck via the C++ engine
- The AI uses its **strategy network** for decisions — the accumulated Nash Equilibrium policy across all training iterations
- After each AI move, the **AI Intelligence Panel** (right column) shows:
  - The exact probability the strategy network assigned to each action
  - The raw advantage network output — how many chips the "Brain" estimates each action gains or loses

The app also hot-reloads `checkpoint.pt` automatically: if training is running in a separate terminal and saves a new checkpoint, the web app will pick it up on the next game start without a restart.

---

## System Requirements

- **Python 3.11+** with PyTorch (CPU build)
- **C++ compiler** with C++17 support (clang on macOS, gcc on Linux)
- **CMake** for building the C++ extension
- **~32 GB RAM** recommended (8.5 GB for training buffers + 12 GB checkpoint + OS overhead)
- Training was developed on Apple M1 (32 GB). CPU-only training; MPS (Apple GPU) is not used due to a pybind11 compatibility issue with shared-memory tensor access across threads.

---

## Algorithm Reference

- **Deep CFR** — Brown, N., Lerer, A., Gross, S., & Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164). ICML 2019.
- **Linear CFR / Discounted CFR** — Brown, N., & Sandholm, T. (2019). [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040). AAAI 2019.
- **Regret-Based Pruning** — Brown, N., & Sandholm, T. (2015). [Regret-Based Pruning in Extensive-Form Games](https://arxiv.org/abs/1509.01888). NeurIPS 2015.
- **Libratus** — Brown, N., & Sandholm, T. (2017). [Libratus: The Superhuman AI for No-Limit Poker](https://www.science.org/doi/10.1126/science.aao1733). Science 2018.
- **Pluribus** — Brown, N., & Sandholm, T. (2019). [Superhuman AI for Multiplayer Poker](https://www.science.org/doi/10.1126/science.aay2400). Science 2019.
