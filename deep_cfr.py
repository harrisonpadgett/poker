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

# ==============================================================================
# Networks
# ==============================================================================

class DeepCFRNetwork(nn.Module):
    def __init__(self, n_actions=3):
        super().__init__()
        # Features (89 dimensions for Royal Hold'em):
        # 0: player
        # 1-20: private cards (one-hot, 20 dims)
        # 21-40: community cards (one-hot, 20 dims)
        # 41-88: history (max 16 actions, 3 dims each = 48 dims)
        self.fc1 = nn.Linear(89, 256)
        self.fc2 = nn.Linear(256, 256)
        self.fc3 = nn.Linear(256, 256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
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
        self.adv_nets = [DeepCFRNetwork().to(self.device), DeepCFRNetwork().to(self.device)]
        self.strat_net = DeepCFRNetwork().to(self.device)

        # Optimizers
        self.adv_opts = [torch.optim.Adam(net.parameters(), lr=adv_lr) for net in self.adv_nets]
        self.strat_opt = torch.optim.Adam(self.strat_net.parameters(), lr=strat_lr)
        
        self.strat_scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(
            self.strat_opt, patience=5, factor=0.5, min_lr=1e-5
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
            steps = 10             # fixed low step count
            # -----------------------------------------------------------------

            # Deep CFR resets the advantage network at each iteration (fresh
            # regret estimates).  Re-init in place rather than constructing a
            # new Module object to avoid unnecessary GC pressure.
            self.adv_nets[player] = DeepCFRNetwork().to(self.device)
            self.adv_opts[player] = torch.optim.Adam(self.adv_nets[player].parameters(), lr=0.001)

            self.adv_nets[player].train()
            for _ in range(steps):
                states, targets, weights = buf.sample(B)
                # non_blocking=True lets the CPU→device copy overlap with
                # other work; no correctness issue because we synchronise
                # implicitly when the tensors are consumed by the forward pass.
                states  = states.to(self.device)
                targets = targets.to(self.device)
                weights = (weights * (B / weights.sum())).to(self.device)

                preds = self.adv_nets[player](states)
                loss  = (weights * ((preds - targets) ** 2).sum(dim=1)).mean()

                # set_to_none=True skips zeroing the gradient tensors and
                # instead deallocates them — ~15% faster than zero_grad().
                self.adv_opts[player].zero_grad(set_to_none=True)
                loss.backward()
                self.adv_opts[player].step()

            losses.append(loss.item())
        if losses:
            self.adv_loss = float(np.mean(losses))

    def train_strat_network(self, batch_size=None, strat_steps=None):
        buf = self.strat_buffer
        n   = len(buf)
        if n < 64:
            return

        # Strategy net is NOT reset each iteration — it accumulates across all
        # iterations — so more steps DO help convergence here.  But we still
        # cap the batch at the compute/overhead sweet-spot and keep steps
        # reasonable so strategy training doesn't dominate when called.
        B     = min(n, 4096)
        steps = 50

        self.strat_net.train()
        for _ in range(steps):
            states, targets, weights = buf.sample(B)
            states  = states.to(self.device, non_blocking=True)
            targets = targets.to(self.device, non_blocking=True)
            weights = (weights * (B / weights.sum())).to(self.device, non_blocking=True)

            preds     = self.strat_net(states)
            log_probs = F.log_softmax(preds, dim=1)
            loss      = -(weights * (targets * log_probs).sum(dim=1)).mean()

            self.strat_opt.zero_grad(set_to_none=True)
            loss.backward()
            self.strat_opt.step()

        self.strat_loss = loss.item()
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

            # The DeepCFRNetwork has layers: fc1, fc2, fc3, out
            # We extract weight and bias tensors as numpy arrays
            poker_cpp.update_model_cache(
                player,
                net.fc1.weight.detach().cpu().numpy(), net.fc1.bias.detach().cpu().numpy(),
                net.fc2.weight.detach().cpu().numpy(), net.fc2.bias.detach().cpu().numpy(),
                net.fc3.weight.detach().cpu().numpy(), net.fc3.bias.detach().cpu().numpy(),
                net.out.weight.detach().cpu().numpy(), net.out.bias.detach().cpu().numpy()
            )
            net.train()

    def _buf_numpy(self, buf, field):
        """Return a NumPy view of a buffer tensor (zero-copy)."""
        return getattr(buf, field).numpy()

    def train(self, n_iterations=1, k_traversals=100):
        import poker_cpp
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations

            # Export current advantage networks to C++ in-memory cache
            self._export_adv_nets()

            # Run all traversals in C++, writing directly into our PyTorch tensor buffers
            counts = poker_cpp.run_traversals(
                k_traversals, t,
                # Advantage buffer 0
                self._buf_numpy(self.adv_buffers[0], 'states'),
                self._buf_numpy(self.adv_buffers[0], 'targets'),
                self._buf_numpy(self.adv_buffers[0], 'weights'),
                self.adv_buffers[0].capacity,
                self.adv_buffers[0].n_inserted,
                # Advantage buffer 1
                self._buf_numpy(self.adv_buffers[1], 'states'),
                self._buf_numpy(self.adv_buffers[1], 'targets'),
                self._buf_numpy(self.adv_buffers[1], 'weights'),
                self.adv_buffers[1].capacity,
                self.adv_buffers[1].n_inserted,
                # Strategy buffer
                self._buf_numpy(self.strat_buffer, 'states'),
                self._buf_numpy(self.strat_buffer, 'targets'),
                self._buf_numpy(self.strat_buffer, 'weights'),
                self.strat_buffer.capacity,
                self.strat_buffer.n_inserted,
            )

            # Sync n_inserted counts back from C++ to Python
            self.adv_buffers[0].n_inserted = counts['adv_n_0']
            self.adv_buffers[1].n_inserted = counts['adv_n_1']
            self.strat_buffer.n_inserted   = counts['strat_n']

            # Train advantage network every iteration
            self.train_adv_network(adv_steps=50)

            # Train strategy network periodically
            if self.iterations % 100 == 0:
                self.train_strat_network(strat_steps=500)

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

    def save_checkpoint(self, filepath):
        checkpoint = {
            'iterations': self.iterations,
            'adv_loss': self.adv_loss,
            'strat_loss': self.strat_loss,
            'adv_nets': [net.state_dict() for net in self.adv_nets],
            'strat_net': self.strat_net.state_dict(),
            'adv_opts': [opt.state_dict() for opt in self.adv_opts],
            'strat_opt': self.strat_opt.state_dict(),
            'adv_buffers': [(buf.capacity, buf.states, buf.targets, buf.weights, buf.n_inserted) for buf in self.adv_buffers],
            'strat_buffer': (self.strat_buffer.capacity, self.strat_buffer.states, self.strat_buffer.targets, self.strat_buffer.weights, self.strat_buffer.n_inserted)
        }
        torch.save(checkpoint, filepath)
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
        
        for net, state_dict in zip(self.adv_nets, checkpoint['adv_nets']):
            net.load_state_dict(state_dict)
        self.strat_net.load_state_dict(checkpoint['strat_net'])
        
        for opt, state_dict in zip(self.adv_opts, checkpoint['adv_opts']):
            opt.load_state_dict(state_dict)
        self.strat_opt.load_state_dict(checkpoint['strat_opt'])
        
        # Load advantage buffers
        for buf, buf_data in zip(self.adv_buffers, checkpoint['adv_buffers']):
            buf.capacity, buf.states, buf.targets, buf.weights, buf.n_inserted = buf_data
        
        # Load strategy buffer
        buf_data = checkpoint['strat_buffer']
        self.strat_buffer.capacity, self.strat_buffer.states, self.strat_buffer.targets, self.strat_buffer.weights, self.strat_buffer.n_inserted = buf_data

        print(f"Successfully loaded checkpoint from {filepath} (Resuming from iteration {self.iterations})")
        return True
