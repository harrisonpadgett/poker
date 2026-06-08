# Royal Hold'em Deep CFR Poker AI

A heads-up (two-player) poker AI for **Royal Hold'em** that learns a near-optimal strategy from scratch using **Deep Counterfactual Regret Minimization (Deep CFR)**. No human poker knowledge is baked in,the AI discovers correct strategy entirely through self-play against itself.

---

## Gameplay

![Gameplay during the preflop betting round. The player holds K♣ A♠. The AI Intelligence Panel on the right explains the AI made a small bet; the Nash Strategy Distribution shows Call and Raise-S as the most probable actions, supported by the Neural Network Output predicting +0.0 chip advantage for those moves.](docs/game_screenshot.png)

*Gameplay during the preflop betting round. The player holds K♣ A♠ facing a small bet from the AI. The right-hand **AI Intelligence Panel** exposes the AI's full internal reasoning: the strategy network assigned nearly equal probability to calling or raising small, and the advantage network (the "Brain") confirms both actions yield a neutral chip advantage over the alternatives.*

---

## What is Royal Hold'em?

Royal Hold'em is a simplified Texas Hold'em variant using only the **20 highest cards** (Tens through Aces). The rules are identical to standard Hold'em (2 hole cards, 5 community cards, 4 betting rounds), but because the deck is stripped, premium hands like straights and full houses are extremely common.

Both players start with **100 chips**. To keep the game tree manageable, the AI is restricted to **5 discrete actions** per decision: Fold, Call, and three Raise sizes (Small, Medium, Large), with a maximum of 2 raises per round.

---

## How the AI Learns: Deep CFR

**Counterfactual Regret Minimization (CFR)** is a reinforcement learning algorithm where an agent plays against itself, tracking the "regret" of not taking alternative actions at every decision point. Over millions of hands, always choosing the action with the highest regret naturally converges toward a **Nash Equilibrium**—an unexploitable poker strategy.

**Deep CFR** replaces the massive lookup tables of classical CFR with neural networks, allowing the AI to generalize across similar poker situations rather than memorizing them.

### The Training Loop

1. **Game Tree Traversal (C++)**: The engine simulates 800 hands per iteration across multiple threads. At each decision point, the AI calculates the *advantage* (regret) of all 5 possible actions and stores these experiences in massive reservoir buffers. 
2. **Neural Network Training (PyTorch)**: Concurrently on the main thread, 12 neural networks learn from the buffers:
   - **8 Advantage Networks ("The Brain")**: Predict which actions are currently over- or under-played.
   - **4 Strategy Networks**: Accumulate the average policy across all iterations into the final, deployable Nash Equilibrium strategy.

The networks only see 122 raw bits of input (one-hot encoded cards and action history). No human poker knowledge or hand strength heuristics are provided.

---

## Engineering Highlights

- **Concurrent Execution**: C++ game tree traversal and PyTorch gradient updates run asynchronously.
- **Warm Restarts**: Every 1,000 iterations, the advantage networks are destroyed and rebuilt from random weights. This intentionally kicks the networks out of local minima, ensuring global convergence.
- **Regret-Based Pruning (RBP)**: The traversal engine skips exploring subtrees for clearly suboptimal actions (like folding pocket Aces), drastically speeding up later training.
- **Reservoir Sampling**: 12 memory buffers hold up to 2 million samples each (~8.5 GB RAM), maintaining a uniform distribution of experience over time.

---

## Project Structure

- `poker_env.py` / `cpp/` — Game engine and high-speed C++ tree traversal (pybind11).
- `deep_cfr.py` — Deep CFR trainer, neural nets, and buffers.
- `train_terminal.py` — Multi-threaded training loop with terminal UI.
- `app.py` / `static/` — Flask backend and web UI for playing the trained AI.

---

## Training the AI

```bash
# 1. Build the C++ extension
cd build && cmake --build . && cp poker_cpp*.so ../ && cd ..

# 2. Start training
python3 train_terminal.py
```

Training runs up to 100,000 iterations, hot-saving to `checkpoint.pt` every 100 iterations.
- At ~3.2 seconds per iteration (M1 Mac), reaching a solid strategy (30k+ iters) takes about a day.
- **Performance Toggle**: Edit `EFFORT = 'medium'` in `train_terminal.py` to change thread allocation and background CPU usage.

---

## Playing Against the AI

You can play against the trained AI live at **[hpadgett.com/poker](https://hpadgett.com/poker)**.

Alternatively, you can host it locally. The local web UI hot-reloads `checkpoint.pt`, so you can play against the AI while it actively trains in a separate terminal:

```bash
python3 app.py
# Opens http://localhost:5001
```

The Intelligence Panel exposes the exact probabilities and chip advantages the neural networks are calculating in real-time.

---

## System Requirements

- Python 3.11+, PyTorch (CPU)
- C++17 compiler and CMake
- ~32 GB RAM (due to massive experience buffers)

## References

- Brown, N. et al. [Deep Counterfactual Regret Minimization](https://arxiv.org/abs/1811.00164) (ICML 2019)
- Brown, N. & Sandholm, T. [Libratus: The Superhuman AI for No-Limit Poker](https://www.science.org/doi/10.1126/science.aao1733) (Science 2018)
