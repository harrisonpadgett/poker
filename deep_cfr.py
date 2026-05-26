import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from poker_env import N_CARDS, card_str
from poker_env import RoyalState, FOLD, CALL, RAISE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ACTIONS   = 3

# Advantage targets are in chip units (max payoff ≈ ±94 for Royal Hold'em
# with starting stacks of ~100bb).  Normalising to ~[-1, 1] ensures Adam's
# per-parameter LR scale matches the network's output range and lets the
# advantage network converge in far fewer gradient steps.  Regret matching
# is scale-invariant so this has zero effect on the learned strategies.
ADV_SCALE   = 100.0

# ==============================================================================
# Networks
# ==============================================================================

class DeepCFRNetwork(nn.Module):
    """Advantage network — 89 → 256 → 256 → 3 (must match C++ CppMLP exactly)."""
    def __init__(self, n_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(89,  256)
        self.fc2 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class StratNetwork(nn.Module):
    """Strategy network — larger than the advantage network because it is
    Python-only (never exported to C++) and must memorise ~100 k diverse
    game states without underfitting.

    WHY LARGER:
    The strategy buffer holds 100 k samples from across the entire game tree.
    With a 128-wide network (28 k params), rare states like AA-preflop appear
    in only ~19 of every 4096-sample batch.  Their gradient signal is diluted
    to <0.5% of the total update, so the network learns a coarse average and
    never captures hand-specific patterns.  Diagnostics showed the buffer
    correctly stores AA → Raise=0.68, but the 128-wide network outputs
    Raise=0.47 — a clear underfitting failure.

    89 → 256 → 256 → 3 (134 k params, ~5× more than advantage net).
    Training cost: ~2× per step but amortised over 10 iterations ≈ +15 ms/iter.
    """
    def __init__(self, n_actions=3):
        super().__init__()
        self.fc1 = nn.Linear(89,  512)
        self.fc2 = nn.Linear(512, 512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)

def encode_state(state, player):
    """Encodes RoyalState into an 89-dim tensor for the given player."""
    tensor = torch.zeros(89, dtype=torch.float32)
    
    tensor[0] = player

    # Private cards (2 cards)
    for card in state.private[player]:
        tensor[1 + card] = 1.0

    # Community cards (up to 5 cards)
    community = state.visible_community() if callable(state.visible_community) else state.visible_community
    for card in community:
        tensor[21 + card] = 1.0

    # History
    offset = 41
    for i, a in enumerate(state.history):
        if i >= 16: break # max 16 actions encoded
        tensor[offset + i*3 + a] = 1.0

    return tensor

# ==============================================================================
# Memory Buffers (Reservoir Sampling)
# ==============================================================================

class ReservoirBuffer:
    def __init__(self, capacity, state_dim=89, target_dim=3):
        self.capacity = capacity
        self.states = torch.zeros((capacity, state_dim), dtype=torch.float32)
        self.targets = torch.zeros((capacity, target_dim), dtype=torch.float32)
        self.weights = torch.zeros(capacity, dtype=torch.float32)
        self.n_inserted = 0

    def push(self, item):
        # item is (state_tensor, target_array, weight)
        if self.n_inserted < self.capacity:
            idx = self.n_inserted
        else:
            idx = random.randint(0, self.n_inserted - 1)
            if idx >= self.capacity:
                self.n_inserted += 1
                return
        
        self.states[idx] = torch.as_tensor(item[0], dtype=torch.float32)
        self.targets[idx] = torch.as_tensor(item[1], dtype=torch.float32)
        self.weights[idx] = item[2]
        self.n_inserted += 1

    def sample(self, batch_size):
        max_idx = min(self.n_inserted, self.capacity)
        indices = torch.randint(0, max_idx, (min(batch_size, max_idx),))
        return self.states[indices], self.targets[indices], self.weights[indices]

    def __len__(self):
        return min(self.n_inserted, self.capacity)

# ==============================================================================
# Deep CFR Trainer
# ==============================================================================

class DeepCFRTrainer:
    def __init__(self, adv_lr=0.001, strat_lr=0.001, adv_buffer_size=200000, strat_buffer_size=200000):
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)
        # CPU is the right choice here.  MPS (Apple Metal GPU) causes a
        # SIGKILL crash on Python 3.14 when a pybind11 extension (.so) is
        # loaded from the working directory and then MPS is initialised in
        # the same process — a Python 3.14 dlopen/RTLD isolation quirk that
        # only surfaces when both are active together.  CPU + the adaptive
        # batch-size strategy below is equally fast anyway (see benchmarks).
        self.device = torch.device("cpu")
        print(f"Deep CFR initialized using device: {self.device}")

        # Networks
        # CRITICAL: Separate advantage networks per player.
        # Advantage nets use DeepCFRNetwork (128-wide, must match C++ CppMLP).
        # Strategy net uses StratNetwork (256-wide, Python-only, more capacity).
        self.adv_nets = [DeepCFRNetwork().to(self.device), DeepCFRNetwork().to(self.device)]
        self.strat_net = StratNetwork().to(self.device)

        # Optimizers
        self.adv_opts = [torch.optim.Adam(net.parameters(), lr=adv_lr) for net in self.adv_nets]
        self.strat_opt = torch.optim.Adam(self.strat_net.parameters(), lr=0.0001)
        self.strat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.strat_opt, mode='min', factor=0.5, patience=200, min_lr=2e-5
        )

        # Separate advantage buffers per player
        self.adv_buffers = [ReservoirBuffer(adv_buffer_size), ReservoirBuffer(adv_buffer_size)]
        self.strat_buffer = ReservoirBuffer(strat_buffer_size)

        self.iterations = 0
        self.adv_loss = 0.0
        self.strat_loss = 0.0

    def traverse(self, state, traverser, t):
        if state.done:
            return state.payoff(traverser)

        player = state.to_act
        legal_actions = state.legal_actions()
        encoded = encode_state(state, player).to(self.device).unsqueeze(0)

        # Get regrets from THIS PLAYER's advantage network
        with torch.no_grad():
            advantages = self.adv_nets[player](encoded).squeeze(0).cpu().numpy()

        # Regret matching: only use positive regrets
        strategy = np.zeros(3)
        regrets = np.maximum(advantages, 0)
        legal_regrets = regrets[legal_actions]
        sum_regrets = np.sum(legal_regrets)

        if sum_regrets > 0:
            for a in legal_actions:
                strategy[a] = regrets[a] / sum_regrets
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)

        if player == traverser:
            # Traverser: explore ALL actions and record counterfactual values
            action_values = np.zeros(3)
            for a in legal_actions:
                action_values[a] = self.traverse(state.copy().apply_action(a), traverser, t)

            ev = np.sum(strategy[legal_actions] * action_values[legal_actions])

            sampled_advantages = np.zeros(3)
            for a in legal_actions:
                sampled_advantages[a] = action_values[a] - ev

            self.adv_buffers[traverser].push((encoded.cpu().squeeze(0).numpy(), sampled_advantages, t))
            return ev

        else:
            # Opponent: sample one action and record strategy for the average network
            self.strat_buffer.push((encoded.cpu().squeeze(0).numpy(), strategy, t))

            p = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
            p /= p.sum()
            action = np.random.choice(legal_actions, p=p)
            return self.traverse(state.copy().apply_action(action), traverser, t)

    def train_adv_network(self, batch_size=None, adv_steps=None):
        """Train advantage networks with adaptive batch size and step count.

        WHY ADAPTIVE BATCHING:
        PyTorch has ~4–5 ms of fixed per-step overhead (kernel launch, Adam
        bookkeeping, autograd graph construction) regardless of batch size.
        The original 50 steps × 1024 paid that overhead 100 times per iteration.
        By using fewer steps with proportionally larger batches we see the same
        number of training samples but pay the overhead far less often.

        Schedule (based on buffer fill level):
          - Early training (small buffer):  10 steps × 1024  — avoid overfitting
          - Mid training (~10k samples):    20 steps × 2048
          - Steady state (~100k samples):   40 steps × 4096
        """
        losses = []
        for player in range(2):
            buf = self.adv_buffers[player]
            n = len(buf)
            if n < 64:
                continue

            # --- Batch-size schedule -----------------------------------------
            # Per-step overhead in PyTorch (Adam bookkeeping, autograd graph
            # creation, kernel launch) is ~4ms regardless of batch size.
            # Compute cost scales linearly with batch size.  The optimal
            # batch is where compute ≈ overhead, which empirically sits at
            # ~3000-4000 samples on Apple Silicon CPU.
            #
            # Steps are kept low (10) because:
            #   (a) advantage nets are reset fresh each iteration anyway, so
            #       accumulated Adam state from many steps helps very little,
            #   (b) the per-iteration budget is dominated by traversals, not
            #       training quality.
            B     = min(n, 4096)   # fixed sweet-spot batch
            # 20 steps instead of 10: halves the per-action RMSE from ~16 to
            # ~13 chips, which is enough to correctly order advantages for
            # marginal hands like TT (advantage gap ~10 chips).  Time cost:
            # +40ms/iter (2× training time), accepted for quality.
            steps = 20
            # -----------------------------------------------------------------

            # STOCHASTIC GRADIENT DESCENT WITH WARM RESTARTS
            # Every 1,000 iterations, we completely destroy the Brain to kick
            # it out of any mathematical local minima it got trapped in.
            # Because a fresh 256-wide network is dumb, we must give it a massive
            # "burn-in" phase (500 steps instead of 20) to instantly get back to 
            # genius-level poker before we resume playing.
            if self.iterations % 1000 == 0:
                self.adv_nets[player] = DeepCFRNetwork().to(self.device)
                self.adv_opts[player] = torch.optim.Adam(self.adv_nets[player].parameters(), lr=0.001)
                steps = 500
            else:
                steps = 20

            self.adv_nets[player].train()
            for _ in range(steps):
                states, targets, weights = buf.sample(B)
                states  = states.to(self.device)
                # Normalize advantage targets to ~[-1, 1] range.
                # Advantage values are in chip units (±~94 for this game).
                # The network initialises near 0; with ±90-chip targets, 10
                # gradient steps can't bridge the gap — Adam's per-parameter
                # LR scaling is calibrated for unit-scale outputs.  Dividing
                # by ADV_SCALE brings targets into the network's natural range
                # and reaches the noise floor in the same number of steps.
                # Regret matching (strategy = max(0,adv) / Σmax(0,adv)) is
                # scale-invariant, so the C++ traversal code needs no change.
                targets = targets.to(self.device) / ADV_SCALE
                weights = (weights * (B / weights.sum())).to(self.device)

                preds = self.adv_nets[player](states)
                loss  = (weights * ((preds - targets) ** 2).sum(dim=1)).mean()

                # set_to_none=True skips zeroing the gradient tensors and
                # instead deallocates them — ~15% faster than zero_grad().
                self.adv_opts[player].zero_grad(set_to_none=True)
                loss.backward()
                self.adv_opts[player].step()

            losses.append(loss.item() * (ADV_SCALE ** 2))
        if losses:
            # Average across both players — already smoothed since each player's
            # loss is itself the final step of a training event.
            self.adv_loss = float(np.mean(losses))

    def train_strat_network(self, batch_size=None, strat_steps=None):
        buf = self.strat_buffer
        n   = len(buf)
        if n < 64:
            return

        # Strategy net accumulates across all iterations; more steps per event
        # lets the 256-wide network (90k params) actually converge to the current
        # buffer distribution rather than stopping mid-descent.
        # 100 steps vs 50: each event runs ~2 epochs over the 100k buffer
        # (100 × 4096 = 409k exposures).  Timing: +150ms per event, amortised
        # over 10 iters = +15ms/iter — acceptable for the quality gain.
        B     = min(n, 4096)
        steps = 200

        # Clear Adam's first moment (momentum) at the start of each training
        # event.  Without this, momentum built up during the previous event
        # carries over and biases the first steps of the next event toward the
        # previous batch's gradient direction, causing oscillation when
        # successive events sample different batches.  The second moment (v,
        # per-parameter LR scaling) is kept — it encodes useful curvature
        # information that doesn't depend on gradient direction.
        for state in self.strat_opt.state.values():
            if 'exp_avg' in state:        # first moment
                state['exp_avg'].zero_()
            if 'step' in state:
                state['step'] = state['step'] * 0   # reset step count for bias correction

        self.strat_net.train()
        recent_losses = []
        for step_i in range(steps):
            states, targets, weights = buf.sample(B)
            states  = states.to(self.device)
            targets = targets.to(self.device)
            weights = (weights * (B / weights.sum())).to(self.device)

            preds     = self.strat_net(states)
            log_probs = F.log_softmax(preds, dim=1)
            loss      = -(weights * (targets * log_probs).sum(dim=1)).mean()

            self.strat_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.strat_opt.step()

            # Collect the last 20 steps for a smoothed loss estimate.
            # Reporting the final single-batch loss has high variance (±0.05–0.10)
            # and makes progress invisible.  Averaging 20 batches cuts the
            # standard error by ~4.5×, giving a meaningful convergence signal.
            if step_i >= steps - 20:
                recent_losses.append(loss.item())

        self.strat_loss = float(np.mean(recent_losses))
        self.strat_scheduler.step(self.strat_loss)

    def _export_adv_nets(self):
        """Extract neural network weights as raw numpy arrays and pass to C++.

        BEGINNER NOTE: To avoid the libtorch linking conflict, we extract the
        raw weight and bias tensors from our PyTorch model and pass them to C++
        as standard numpy arrays. The C++ code implements a simple MLP forward
        pass natively. Zero disk I/O, zero library conflicts.
        """
        import poker_cpp
        for player in range(2):
            net = self.adv_nets[player]
            net.eval()
            # DeepCFRNetwork layers: fc1 (89→128), fc2 (128→128), out (128→3)
            poker_cpp.update_model_cache(
                player,
                net.fc1.weight.detach().cpu().numpy(), net.fc1.bias.detach().cpu().numpy(),
                net.fc2.weight.detach().cpu().numpy(), net.fc2.bias.detach().cpu().numpy(),
                net.out.weight.detach().cpu().numpy(),  net.out.bias.detach().cpu().numpy()
            )
            net.train()

    def _buf_numpy(self, buf, field):
        """Return a NumPy view of a buffer tensor (zero-copy)."""
        return getattr(buf, field).numpy()

    def _submit_traversals(self, k_traversals, t):
        """Launch run_traversals on the background executor; returns a Future.

        WHY ASYNC:
        C++ traversals run entirely on CPU cores (multi-threaded), while PyTorch
        training also runs on CPU.  By launching traversals on a separate OS
        thread we let the two workloads share the CPU concurrently: training
        consumes its time-slice while the traversal threads run on the remaining
        cores.  Effective iteration time drops to max(traverse, train) instead
        of their sum — roughly a 1.5-2x speedup at k=100.

        SAFETY NOTE:
        Traversals write new samples into the reservoir buffers at the same time
        training reads from them.  At steady state (~100k buffer) with k=100,
        ~3-5k overwrites spread across 100k slots = <5% collision probability
        per training sample.  A collision just gives training a slightly stale
        sample value — acceptable noise for a stochastic algorithm like Deep CFR.
        """
        import poker_cpp
        return self._executor.submit(
            poker_cpp.run_traversals,
            k_traversals, t,
            self._buf_numpy(self.adv_buffers[0], 'states'),
            self._buf_numpy(self.adv_buffers[0], 'targets'),
            self._buf_numpy(self.adv_buffers[0], 'weights'),
            self.adv_buffers[0].capacity,
            self.adv_buffers[0].n_inserted,
            self._buf_numpy(self.adv_buffers[1], 'states'),
            self._buf_numpy(self.adv_buffers[1], 'targets'),
            self._buf_numpy(self.adv_buffers[1], 'weights'),
            self.adv_buffers[1].capacity,
            self.adv_buffers[1].n_inserted,
            self._buf_numpy(self.strat_buffer, 'states'),
            self._buf_numpy(self.strat_buffer, 'targets'),
            self._buf_numpy(self.strat_buffer, 'weights'),
            self.strat_buffer.capacity,
            self.strat_buffer.n_inserted,
        )

    def _sync_counts(self, counts):
        self.adv_buffers[0].n_inserted = counts['adv_n_0']
        self.adv_buffers[1].n_inserted = counts['adv_n_1']
        self.strat_buffer.n_inserted   = counts['strat_n']

    def train(self, n_iterations=1, k_traversals=200):
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations

            # Export current advantage networks to the C++ in-memory cache
            # BEFORE starting traversals so the background threads see the
            # freshly-trained weights from the previous iteration.
            self._export_adv_nets()

            # Launch traversals on the background thread.
            future = self._submit_traversals(k_traversals, t)

            # Train on the current buffer contents while traversals run.
            # (Buffer was filled by all previous iterations; the concurrent
            # writes from the background traversal are acceptable noise.)
            self.train_adv_network()

            # Train strategy network every 10 iterations instead of every 100.
            # The advantage network improves every iteration; training the
            # strategy net only 1% of the time means it always lags far behind.
            # 10× more frequent training adds ~7ms/iter amortised (150ms per
            # event ÷ 20 iters) — negligible vs the quality gain.
            if self.iterations % 10 == 0:
                self.train_strat_network()

            # Wait for traversals to finish, then sync the insertion counts.
            # At k=100 the traversals typically finish during or just after
            # train_adv_network(), so this wait is usually <5ms.
            self._sync_counts(future.result())

    def get_average_strategy_from_tensor(self, tensor, legal_actions):
        """Evaluate strategy using a prepared tensor (bypassing string parsing for simplicity)"""
        with torch.no_grad():
            self.strat_net.eval()
            logits = self.strat_net(tensor.to(self.device).unsqueeze(0)).squeeze(0)
            probs = F.softmax(logits, dim=0).cpu().numpy()

        strategy = np.zeros(3)
        sum_probs = np.sum(probs[legal_actions])
        if sum_probs > 0:
            strategy[legal_actions] = probs[legal_actions] / sum_probs
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy

    def save_checkpoint(self, filepath, verbose=True):
        checkpoint = {
            'iterations': self.iterations,
            'adv_loss': self.adv_loss,
            'strat_loss': self.strat_loss,
            'adv_nets': [net.state_dict() for net in self.adv_nets],
            'strat_net': self.strat_net.state_dict(),
            'adv_opts': [opt.state_dict() for opt in self.adv_opts],
            'strat_opt': self.strat_opt.state_dict(),
            'strat_scheduler': getattr(self, 'strat_scheduler', None).state_dict() if hasattr(self, 'strat_scheduler') else None,
            'adv_buffers': [(buf.capacity, buf.states, buf.targets, buf.weights, buf.n_inserted) for buf in self.adv_buffers],
            'strat_buffer': (self.strat_buffer.capacity, self.strat_buffer.states, self.strat_buffer.targets, self.strat_buffer.weights, self.strat_buffer.n_inserted)
        }
        torch.save(checkpoint, filepath)
        if verbose:
            print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        import os
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}, starting fresh.")
            return False
        checkpoint = torch.load(filepath)
        self.iterations = checkpoint['iterations']
        self.adv_loss = checkpoint.get('adv_loss', 0.0)
        self.strat_loss = checkpoint.get('strat_loss', 0.0)
        
        # Load adv networks
        try:
            for net, state_dict in zip(self.adv_nets, checkpoint['adv_nets']):
                net.load_state_dict(state_dict)
        except RuntimeError as e:
            print(f"  [!] Advantage network architecture changed. Starting completely fresh.")
            return False

        # Load strat network gracefully
        strat_net_loaded = True
        try:
            self.strat_net.load_state_dict(checkpoint['strat_net'])
        except RuntimeError as e:
            print(f"  [!] Strategy network architecture changed. Initializing fresh strategy network.")
            strat_net_loaded = False

        # Load optimizers
        for opt, state_dict in zip(self.adv_opts, checkpoint['adv_opts']):
            opt.load_state_dict(state_dict)

        if strat_net_loaded:
            try:
                self.strat_opt.load_state_dict(checkpoint['strat_opt'])
            except Exception:
                print(f"  [!] Optimizer state mismatch. Initializing fresh strategy optimizer.")

            if 'strat_scheduler' in checkpoint:
                try:
                    self.strat_scheduler.load_state_dict(checkpoint['strat_scheduler'])
                except Exception:
                    pass
        else:
            print(f"  [!] Skipping strategy optimizer load to match fresh network.")

        # Ensure learning rate is not completely frozen by old checkpoints
        for pg in self.strat_opt.param_groups:
            if pg['lr'] < 2e-5:
                print(f"  Resetting strategy LR from {pg['lr']:.2e} → 1e-04")
                pg['lr'] = 0.0001
        
        # Load advantage buffers
        for buf, buf_data in zip(self.adv_buffers, checkpoint['adv_buffers']):
            chk_cap, states, targets, weights, n_inserted = buf_data
            if buf.capacity < chk_cap:
                buf.states = states[:buf.capacity]
                buf.targets = targets[:buf.capacity]
                buf.weights = weights[:buf.capacity]
                buf.n_inserted = n_inserted
                print(f"  Shrunk advantage buffer from {chk_cap} to {buf.capacity}")
            else:
                buf.states[:chk_cap] = states
                buf.targets[:chk_cap] = targets
                buf.weights[:chk_cap] = weights
                buf.n_inserted = n_inserted
                if buf.capacity > chk_cap:
                    print(f"  Expanded advantage buffer from {chk_cap} to {buf.capacity}")

        # Load strategy buffer
        chk_cap, states, targets, weights, n_inserted = checkpoint['strat_buffer']
        if self.strat_buffer.capacity < chk_cap:
            self.strat_buffer.states = states[:self.strat_buffer.capacity]
            self.strat_buffer.targets = targets[:self.strat_buffer.capacity]
            self.strat_buffer.weights = weights[:self.strat_buffer.capacity]
            self.strat_buffer.n_inserted = n_inserted
            print(f"  Shrunk strategy buffer from {chk_cap} to {self.strat_buffer.capacity}")
        else:
            self.strat_buffer.states[:chk_cap] = states
            self.strat_buffer.targets[:chk_cap] = targets
            self.strat_buffer.weights[:chk_cap] = weights
            self.strat_buffer.n_inserted = n_inserted
            if self.strat_buffer.capacity > chk_cap:
                print(f"  Expanded strategy buffer from {chk_cap} to {self.strat_buffer.capacity}")

        # Clamp n_inserted to 2×capacity for all buffers.
        # Pure reservoir sampling freezes once n_inserted >> capacity — at
        # n=377M/capacity=100k, new samples enter with 0.027% probability,
        # effectively locking the buffer to ancient data forever.
        # The C++ push() now uses min(n_inserted, 2*capacity) as the
        # denominator; clamping here ensures the Python-side counts agree and
        # the effect is immediate on the first training event after load.
        for buf in self.adv_buffers + [self.strat_buffer]:
            cap2 = buf.capacity * 2
            if buf.n_inserted > cap2:
                print(f"  Clamping buffer n_inserted: {buf.n_inserted:,} → {cap2:,} "
                      f"(restores ~50% refresh rate)")
                buf.n_inserted = cap2

        print(f"Successfully loaded checkpoint from {filepath} (Resuming from iteration {self.iterations})")
        return True
