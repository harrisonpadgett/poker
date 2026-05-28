# Royal Hold'em Deep CFR Poker AI

A heads-up (two-player) poker AI for **Royal Hold'em** that learns a near-optimal strategy from scratch using **Deep Counterfactual Regret Minimization (Deep CFR)**. No human poker knowledge is baked in — the AI discovers the correct strategy entirely through self-play.

---

## What is Royal Hold'em?

Royal Hold'em is a simplified variant of Texas Hold'em that uses only the **20 highest cards** in a standard deck: Tens, Jacks, Queens, Kings, and Aces in all four suits. The rules are otherwise identical to Hold'em:

- Each player receives **2 private hole cards**
- **5 community cards** are revealed across 4 betting rounds: Preflop, Flop (3 cards), Turn (4th card), River (5th card)
- Players can **Fold**, **Call**, or **Raise** at any point
- The best 5-card hand at showdown wins the pot

Because the deck is so small, strong hands like flushes and full houses come up frequently. Two players start with **100 chips** each, with a 1-chip ante from each side.

### Action Space

This implementation uses **5 actions** per decision point:

| Action | Meaning | Raise Amount |
|--------|---------|-------------|
| `Fold` | Give up the pot | — |
| `Call` | Match the opponent's bet | — |
| `Raise-S` | Small raise | +2 chips (preflop/flop), +4 chips (turn/river) |
| `Raise-M` | Medium raise | +4 chips (preflop/flop), +6 chips (turn/river) |
| `Raise-L` | Large raise | +6 chips (preflop/flop), +8 chips (turn/river) |

A maximum of **2 raises** per round is allowed to prevent runaway betting.

---

## How the AI Learns: Deep CFR

### The Core Idea — Regret Minimization

Imagine you're playing poker and you folded a hand, then your opponent shows you their cards and you realize you would have won. That missed opportunity is your **regret** for folding. If you always track this regret and play the action you *wish* you had played more often, you gradually converge toward the mathematically optimal strategy — one where no single action would have consistently been better.

**Counterfactual Regret Minimization (CFR)** formalizes this for multi-player games. Instead of one player's perspective, it computes the regret for *every possible decision point in the game tree* simultaneously, across both players. After enough iterations, both players converge to a **Nash Equilibrium** — a strategy where neither player can improve by changing their behavior unilaterally. Against a Nash Equilibrium player, no opponent strategy can do better than break even in the long run.

**Deep CFR** replaces the lookup table used in classical CFR (which would require storing a separate value for every possible game state) with **neural networks** that generalize across similar situations. This makes it tractable for games with millions of distinct states.

### The Training Loop (One Iteration)

Each iteration consists of two parallel phases:

#### Phase 1 — Game Tree Traversal (C++ engine, multi-threaded)

**800 games** are simulated per iteration. For each game:

1. Cards are dealt randomly.
2. The traversal alternates between two roles:
   - **Traverser** (one player at a time): at their own decision points, the traverser explores **all 5 possible actions** and records what each one would have been worth.
   - **Opponent**: samples a single action according to their current strategy.
3. At each traverser decision point, the **advantage** of each action is computed: `advantage(action) = value(action) - expected_value_of_current_strategy`. A positive advantage means "I should have done this more"; a negative advantage means "I should have done this less."
4. These (state, advantage, iteration_weight) samples are stored in **reservoir buffers** — one per player per betting round (8 total advantage buffers, 4 strategy buffers).

**Regret-Based Pruning:** When the AI is confident an action is clearly suboptimal (normalized advantage < -0.03, equivalent to ~3 chips below expected value), it skips exploring that subtree entirely. This avoids computing outcomes for actions like "fold pocket aces" that are obviously bad, significantly speeding up traversal at later training stages.

**Linear CFR Weighting:** Samples from earlier iterations are given lower weight than recent ones. This means the strategy reflects *what the AI has learned recently*, not its confused early attempts at the game.

#### Phase 2 — Neural Network Training (PyTorch, runs concurrently)

While the C++ traversal runs in the background, **12 neural networks** are trained simultaneously:

- **8 Advantage Networks** (`adv_nets[player][round]`): One for each combination of player (0 or 1) × betting round (Preflop, Flop, Turn, River). Each network learns to predict, for any game state it's shown, which actions are currently over- or under-played relative to optimal. These are the "Brain" — the AI's working model of regret.
- **4 Strategy Networks** (`strat_nets[round]`): One per betting round. These accumulate the *average* strategy across all iterations — the final strategy the AI will actually use when playing.

All networks share the same architecture: **122 input features → 256 hidden → 256 hidden → 5 outputs**.

The 122 input features encode everything the AI can see:
- Which player is acting (1 bit)
- Hole cards, one-hot encoded over the 20-card deck (20 bits)
- Visible community cards, one-hot encoded (20 bits)
- Up to 16 previous actions in the hand, one-hot encoded (80 bits)
- Whether both hole cards are the same suit (`is_suited`, 1 bit)

---

## Key Engineering Details

### Warm Restarts Every 1,000 Iterations

Every 1,000 iterations, all 8 advantage networks are **completely destroyed and rebuilt from random weights**. This might sound counterproductive, but it solves a real problem: neural networks can get stuck in **local minima** — states where the network has learned a strategy that's internally consistent but not globally optimal. A network that has been gradually nudged toward a mediocre strategy will resist further improvement because every small change makes things locally worse.

By resetting to random weights and immediately running 500 "burn-in" training steps on the existing buffers (which contain rich data from thousands of previous iterations), the network rapidly converges to a fresh, global perspective. The **strategy buffers do not reset** — they accumulate wisdom across all iterations and are unaffected by the network resets.

After a warm restart you'll see `ADV LOSS` spike briefly, then fall back below its previous level within ~500 iterations. This is expected and healthy.

### Reservoir Buffers

The advantage and strategy buffers use **reservoir sampling**: when a buffer is full, new samples replace old ones with a probability designed to keep the distribution uniform over time. The buffers are capped so that the insertion probability stays at ~50% at all times, ensuring old data doesn't dominate and new experience is always being incorporated.

Each of the 8 advantage buffers and 4 strategy buffers holds up to **2,000,000 samples** (totaling ~8.5 GB of buffer RAM). River buffers fill much faster than preflop buffers because every hand that reaches the river generates many decision points.

### Atomic Checkpoint Saving

The training state is saved to `checkpoint.pt` every 100 iterations. To prevent corruption if the process is interrupted mid-save, the file is first written to a `.tmp` file and then atomically renamed over the real checkpoint. This means `checkpoint.pt` is always in a valid state, even if you press Ctrl+C during a save.

### Async Traversal

C++ traversals run on a background thread while PyTorch trains the networks on the main thread. This means game simulation and network training happen **concurrently**, reducing the effective iteration time to roughly `max(traversal_time, training_time)` instead of their sum.

### AdamW Optimizer with Gradient Clipping

Advantage networks use **AdamW** (Adam with weight decay) rather than plain Adam. Weight decay acts as a regularizer between warm restarts, preventing the networks from overfitting to the most recent few hundred samples. **Gradient clipping** (max norm 1.0) prevents the high-weight river samples from occasionally producing catastrophically large gradient updates that destabilize training.

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
└── checkpoint.pt         # Saved training state (~12 GB at convergence)
```

---

## Training

```bash
# Build the C++ extension first
cd build && cmake --build . && cp poker_cpp*.so ../ && cd ..

# Start or resume training
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
- `ADV LOSS` — how wrong the Brain's advantage predictions currently are (in chip² units). Spikes every 1,000 iterations during warm restarts, then falls back. Lower is better, but don't panic at spikes.
- `STRAT LOSS` — cross-entropy loss on the strategy network. This is the real convergence signal. Should trend steadily downward over thousands of iterations.
- `[BUF]` rows — the accumulated average strategy for Pocket Aces and Pocket Tens at preflop, expressed as action probabilities. This is what the AI would actually do if you played against it right now.
- `[ADV]` rows — the Brain's current advantage estimates for each action (in chips). Negative = the Brain thinks this action loses chips. Positive = the Brain thinks this action gains chips.
- `Buffers` — insertion counts for the preflop and river advantage/strategy buffers. The river buffers grow much faster because every hand that reaches showdown generates many river samples.

### Convergence Milestones

| Iterations | What to expect |
|---|---|
| 0–10k | Rapid learning. AA Fold probability drops toward 0%. Raise sizes begin differentiating. |
| 10k–30k | Strategy stabilizes for premium hands. STRAT LOSS falls below 0.85. |
| 30k–50k | Near-Nash equilibrium for preflop and flop. Visible change slows to <1% per 5,000 iters. |
| 50k–80k | Marginal refinement. River strategies converge. |
| 80k+ | Effectively Nash for all practical purposes. |

At ~3.2 seconds per iteration, reaching 50,000 iterations takes approximately **44 hours** of continuous training.

---

## Playing Against the AI

```bash
python3 app.py
```

Opens a web interface at `http://localhost:5000` where you can play heads-up Royal Hold'em against the trained AI. The AI uses the strategy network for decisions during normal play, and can optionally run **real-time subgame solving** at each decision point — a Monte Carlo technique that refines the global strategy network's answer for the specific board and bet history you're currently facing.

---

## System Requirements

- **Python 3.11+** with PyTorch
- **C++ compiler** with C++17 support (clang on macOS, gcc on Linux)
- **CMake** for building the C++ extension
- **~32 GB RAM** recommended (8.5 GB for training buffers + 12 GB checkpoint + OS overhead)
- Training was developed on Apple M1 (32 GB). CPU-only training; MPS (Apple GPU) is not used due to a pybind11 compatibility issue.

---

## Algorithm Reference

- **Deep CFR** — Brown, N., Lerer, A., Gross, S., & Sandholm, T. (2019). [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164). ICML 2019.
- **Linear CFR** — Brown, N., & Sandholm, T. (2019). [Solving Imperfect-Information Games via Discounted Regret Minimization](https://arxiv.org/abs/1809.04040). AAAI 2019.
- **Regret-Based Pruning** — Brown, N., & Sandholm, T. (2015). [Regret-Based Pruning in Extensive-Form Games](https://arxiv.org/abs/1509.01888). NeurIPS 2015.
