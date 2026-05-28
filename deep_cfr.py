import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from poker_env import (N_CARDS, N_ACTIONS, RAISE_AMOUNTS, card_str,
                       RoyalState, FOLD, CALL, RAISE_S, RAISE_M, RAISE_L)

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ROUNDS  = 4   # Preflop, Flop, Turn, River

# State encoding: 1 (player) + 20 (hole) + 20 (community) + 16*N_ACTIONS (history) + 1 (is_suited)
STATE_DIM = 1 + N_CARDS + N_CARDS + 16 * N_ACTIONS + 1   # = 122

# Advantage targets are in chip units (max payoff ≈ ±99 for Royal Hold'em).
# Normalising to ~[-1,1] keeps Adam's per-param LR scale healthy.
# Regret matching is scale-invariant, so this doesn't affect learned strategies.
ADV_SCALE = 100.0


# ==============================================================================
# Networks
# ==============================================================================

class DeepCFRNetwork(nn.Module):
    """Advantage network — 122 → 256 → 256 → 5.

    One instance per (player, round) combination (8 total).
    Must match C++ CppMLP exactly.
    """
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 256)
        self.fc2 = nn.Linear(256,       256)
        self.out = nn.Linear(256, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


class StratNetwork(nn.Module):
    """Average strategy network — 122 → 512 → 512 → 5.

    One instance per round (4 total). Larger than each advantage network
    because it must memorise the full game tree for that round across
    all hand combinations without underfitting.
    """
    def __init__(self, n_actions=N_ACTIONS):
        super().__init__()
        self.fc1 = nn.Linear(STATE_DIM, 512)
        self.fc2 = nn.Linear(512,       512)
        self.out = nn.Linear(512, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.out(x)


def encode_state(state, player):
    """Encode RoyalState into a 122-dim float32 tensor for the given player.

    Layout  (must match C++ encode_state in bindings.cpp):
      [0]       player index (0 or 1)
      [1-20]    hole cards one-hot (20-card Royal deck)
      [21-40]   visible community cards one-hot
      [41-120]  action history: 16 slots × N_ACTIONS=5 one-hot bits
      [121]     is_suited: 1.0 if both hole cards share the same suit

    Accepts both Python RoyalState (visible_community is a property)
    and C++ poker_cpp.RoyalState (visible_community() is a method).
    """
    tensor = torch.zeros(STATE_DIM, dtype=torch.float32)
    tensor[0] = player

    for card in state.private[player]:
        tensor[1 + card] = 1.0

    community = (state.visible_community()
                 if callable(state.visible_community)
                 else state.visible_community)
    for card in community:
        tensor[21 + card] = 1.0

    for i, a in enumerate(state.history):
        if i >= 16:
            break
        tensor[41 + i * N_ACTIONS + a] = 1.0

    cards = state.private[player]
    if cards[0] // 5 == cards[1] // 5:
        tensor[121] = 1.0

    return tensor


# ==============================================================================
# Memory Buffers (Reservoir Sampling)
# ==============================================================================

class ReservoirBuffer:
    def __init__(self, capacity, state_dim=STATE_DIM, target_dim=N_ACTIONS):
        self.capacity = capacity
        self.states  = torch.zeros((capacity, state_dim),  dtype=torch.float32)
        self.targets = torch.zeros((capacity, target_dim), dtype=torch.float32)
        self.weights = torch.zeros(capacity,               dtype=torch.float32)
        self.n_inserted = 0

    def push(self, item):
        if self.n_inserted < self.capacity:
            idx = self.n_inserted
        else:
            idx = random.randint(0, self.n_inserted - 1)
            if idx >= self.capacity:
                self.n_inserted += 1
                return
        self.states[idx]  = torch.as_tensor(item[0], dtype=torch.float32)
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
# Subgame Solver (used at inference time by app.py)
# ==============================================================================

def _regret_match(cumulative_regrets, legal):
    """Convert accumulated regrets into a mixed strategy via regret matching."""
    strategy = np.zeros(N_ACTIONS)
    pos = np.maximum(cumulative_regrets[legal], 0.0)
    total = pos.sum()
    if total > 0:
        strategy[legal] = pos / total
    else:
        strategy[legal] = 1.0 / len(legal)
    return strategy


def _blueprint_rollout(state, player_of_interest, trainer):
    """Roll out to terminal using the blueprint average strategy (per-round strat nets).

    `state` must be a Python RoyalState with full private and community info.
    Returns the chip payoff for player_of_interest.
    """
    max_depth = 24
    for _ in range(max_depth):
        if state.done:
            return state.payoff(player_of_interest)
        legal = state.legal_actions()
        if not legal:
            return state.payoff(player_of_interest)

        t = encode_state(state, state.to_act)
        probs = trainer.get_average_strategy_from_tensor(t, legal, state.round)
        p = np.array([probs[a] for a in legal], dtype=np.float64)
        p /= p.sum()

        action = int(np.random.choice(legal, p=p))
        state = state.copy()
        state.apply_action(action)

    return state.payoff(player_of_interest) if state.done else 0.0


def _make_rollout_state(cpp_state, ai_player, opp_hand):
    """Build a Python RoyalState from a C++ game state with a substituted opponent hand.

    Future community cards are taken from the pre-dealt deck (exposed via
    community_cards property), so rollouts are deterministic for that deal.

    Returns None if opp_hand conflicts with known cards.
    """
    ai_cards    = list(cpp_state.private[ai_player])
    vis_comm    = list(cpp_state.visible_community())
    full_comm   = list(cpp_state.community_cards)   # all 5, pre-dealt

    # Sanity check: opponent hand doesn't collide with known cards
    known = set(ai_cards) | set(full_comm)
    if any(c in known for c in opp_hand):
        return None

    s = RoyalState.__new__(RoyalState)
    s.private   = [None, None]
    s.private[ai_player]       = ai_cards
    s.private[1 - ai_player]   = list(opp_hand)
    s.community = full_comm
    s.round     = cpp_state.round
    s.pot       = cpp_state.pot
    s.stacks    = list(cpp_state.stacks)
    s.bets      = list(cpp_state.bets)
    s.to_act    = cpp_state.to_act
    s.raises    = cpp_state.raises
    s.done      = cpp_state.done
    s.winner    = None
    s.history   = list(cpp_state.history)
    s.n_acted   = cpp_state.n_acted
    return s


def subgame_solve(cpp_state, ai_player, trainer,
                  n_opponent_samples=20, n_rollouts=5):
    """Real-time subgame CFR solve at the current decision point.

    Algorithm (Monte Carlo Subgame Solving):
      1. Sample N plausible opponent hands from the remaining deck.
      2. For each hand, estimate action values via blueprint rollouts.
      3. Accumulate CFR counterfactual regrets across all samples.
      4. Return the regret-matched strategy.

    Why this outperforms the raw blueprint:
      The blueprint approximates Nash equilibrium globally — it has to cover
      every possible game state with finite network capacity. Subgame solving
      concentrates all inference compute on THIS specific board/pot/history,
      deriving a strategy that is closer to the exact Nash solution for this
      subgame rather than a global approximation.

    Args:
        cpp_state:           Current poker_cpp.RoyalState (C++ object).
        ai_player:           The AI's player index (0 or 1).
        trainer:             DeepCFRTrainer holding the blueprint networks.
        n_opponent_samples:  How many opponent hand combos to consider.
        n_rollouts:          Rollouts per action per opponent hand.

    Returns:
        np.ndarray of shape (N_ACTIONS,) — action probabilities for ai_player.
    """
    legal = cpp_state.legal_actions()
    if len(legal) <= 1:
        probs = np.zeros(N_ACTIONS)
        for a in legal: probs[a] = 1.0
        return probs

    # Enumerate cards not held by the AI or visible on the board
    ai_cards  = list(cpp_state.private[ai_player])
    full_comm = list(cpp_state.community_cards)
    known     = set(ai_cards) | set(full_comm)
    remaining = [c for c in range(N_CARDS) if c not in known]

    if len(remaining) < 2:
        # Fallback: not enough cards left to sample an opponent hand
        t = encode_state(cpp_state, ai_player)
        return trainer.get_average_strategy_from_tensor(t, legal, cpp_state.round)

    cumulative_regrets = np.zeros(N_ACTIONS)
    n_valid = 0

    for _ in range(n_opponent_samples):
        opp_hand = random.sample(remaining, 2)
        py_state = _make_rollout_state(cpp_state, ai_player, opp_hand)
        if py_state is None:
            continue
        n_valid += 1

        # Estimate value of each action via Monte Carlo rollouts
        action_vals = np.zeros(N_ACTIONS)
        for a in legal:
            total = 0.0
            for _ in range(n_rollouts):
                s = py_state.copy()
                s.apply_action(a)
                total += _blueprint_rollout(s, ai_player, trainer)
            action_vals[a] = total / n_rollouts

        # CFR regret update: accumulate regret(a) = value(a) - EV
        strategy = _regret_match(cumulative_regrets, legal)
        ev = float(np.sum(strategy[legal] * action_vals[legal]))
        for a in legal:
            cumulative_regrets[a] += action_vals[a] - ev

    if n_valid == 0:
        # All samples failed — fall back to blueprint
        t = encode_state(cpp_state, ai_player)
        return trainer.get_average_strategy_from_tensor(t, legal, cpp_state.round)

    return _regret_match(cumulative_regrets, legal)


# ==============================================================================
# Deep CFR Trainer
# ==============================================================================

class DeepCFRTrainer:
    def __init__(self, adv_lr=0.001, strat_lr=0.001,
                 adv_buffer_size=200_000, strat_buffer_size=200_000):
        from concurrent.futures import ThreadPoolExecutor
        self._executor = ThreadPoolExecutor(max_workers=1)

        self.device = torch.device("cpu")
        print(f"Deep CFR initialized using device: {self.device}")

        # 8 advantage networks: adv_nets[player][round]
        self.adv_nets = [
            [DeepCFRNetwork().to(self.device) for _ in range(N_ROUNDS)]
            for _ in range(2)
        ]
        # 4 per-round strategy networks: strat_nets[round]
        self.strat_nets = [StratNetwork().to(self.device) for _ in range(N_ROUNDS)]

        # Advantage optimizers: adv_opts[player][round]
        # AdamW adds proper L2 weight decay, which regularises between warm restarts
        # and reduces overfitting on buffers with imbalanced hand distributions.
        self.adv_opts = [
            [torch.optim.AdamW(self.adv_nets[p][r].parameters(),
                               lr=adv_lr, weight_decay=1e-4)
             for r in range(N_ROUNDS)]
            for p in range(2)
        ]
        # Per-round strategy optimizers & schedulers
        self.strat_opts = [
            torch.optim.Adam(self.strat_nets[r].parameters(), lr=strat_lr)
            for r in range(N_ROUNDS)
        ]
        self.strat_schedulers = [
            torch.optim.lr_scheduler.ReduceLROnPlateau(
                self.strat_opts[r], mode='min', factor=0.5, patience=200, min_lr=2e-5
            )
            for r in range(N_ROUNDS)
        ]

        # 8 advantage buffers: adv_buffers[player][round]
        self.adv_buffers = [
            [ReservoirBuffer(adv_buffer_size) for _ in range(N_ROUNDS)]
            for _ in range(2)
        ]
        # 4 per-round strategy buffers: strat_buffers[round]
        self.strat_buffers = [ReservoirBuffer(strat_buffer_size) for _ in range(N_ROUNDS)]

        self.iterations = 0
        self.adv_loss   = 0.0
        self.strat_loss = 0.0

    # -------------------------------------------------------------------------
    # Python-side traversal (fallback — C++ run_traversals is the fast path)
    # -------------------------------------------------------------------------
    def traverse(self, state, traverser, t):
        if state.done:
            return state.payoff(traverser)

        player = state.to_act
        rnd    = state.round
        legal_actions = state.legal_actions()
        encoded = encode_state(state, player).to(self.device).unsqueeze(0)

        with torch.no_grad():
            advantages = self.adv_nets[player][rnd](encoded).squeeze(0).cpu().numpy()

        strategy = np.zeros(N_ACTIONS)
        regrets  = np.maximum(advantages, 0)
        sum_reg  = np.sum(regrets[legal_actions])
        if sum_reg > 0:
            for a in legal_actions: strategy[a] = regrets[a] / sum_reg
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)

        if player == traverser:
            action_values = np.zeros(N_ACTIONS)
            for a in legal_actions:
                action_values[a] = self.traverse(state.copy().apply_action(a), traverser, t)

            ev = np.sum(strategy[legal_actions] * action_values[legal_actions])
            sampled_adv = np.zeros(N_ACTIONS)
            for a in legal_actions:
                sampled_adv[a] = action_values[a] - ev

            w = max(t, 100)   # floor so early iterations aren't near-zero weight
            self.adv_buffers[traverser][rnd].push(
                (encoded.cpu().squeeze(0).numpy(), sampled_adv, w)
            )
            return ev
        else:
            w = max(t, 100)
            self.strat_buffers[rnd].push(
                (encoded.cpu().squeeze(0).numpy(), strategy, w)
            )
            p = np.array([strategy[a] for a in legal_actions], dtype=np.float64)
            p /= p.sum()
            action = np.random.choice(legal_actions, p=p)
            return self.traverse(state.copy().apply_action(action), traverser, t)

    # -------------------------------------------------------------------------
    # Training
    # -------------------------------------------------------------------------
    def train_adv_network(self):
        """Train all 8 advantage networks (player × round)."""
        losses = []
        for player in range(2):
            for rnd in range(N_ROUNDS):
                buf = self.adv_buffers[player][rnd]
                n   = len(buf)
                if n < 64:
                    continue

                B = min(n, 4096)

                # Warm restart every 1,000 iterations to escape local minima.
                if self.iterations % 1000 == 0:
                    self.adv_nets[player][rnd] = DeepCFRNetwork().to(self.device)
                    self.adv_opts[player][rnd] = torch.optim.AdamW(
                        self.adv_nets[player][rnd].parameters(),
                        lr=0.001, weight_decay=1e-4
                    )
                    steps = 500
                    # LR warmup: first 10 steps at low lr so AdamW's moment
                    # estimates stabilise before the full rate kicks in.
                    WARMUP_STEPS = 10
                    WARMUP_LR    = 0.0001
                else:
                    # Adaptive steps: scale with buffer richness.
                    # Small buffer → fewer steps (avoid overfitting on repeated
                    # samples). Large buffer → more steps (exploit the rich data).
                    # Max is 60, not 100 — keeps training time ~equal to the old
                    # 75-step baseline so RBP traversal savings actually show up.
                    if n < 10_000:
                        steps = 20
                    elif n < 100_000:
                        steps = 40
                    else:
                        steps = 60
                    WARMUP_STEPS = 0
                    WARMUP_LR    = 0.001  # unused

                self.adv_nets[player][rnd].train()
                opt = self.adv_opts[player][rnd]
                for step_i in range(steps):
                    # Apply warmup LR for the first WARMUP_STEPS steps.
                    if step_i == 0 and WARMUP_STEPS > 0:
                        for pg in opt.param_groups: pg['lr'] = WARMUP_LR
                    elif step_i == WARMUP_STEPS and WARMUP_STEPS > 0:
                        for pg in opt.param_groups: pg['lr'] = 0.001
                    states, targets, weights = buf.sample(B)
                    states  = states.to(self.device)
                    targets = targets.to(self.device) / ADV_SCALE
                    weights = (weights * (B / weights.sum())).to(self.device)

                    preds = self.adv_nets[player][rnd](states)
                    loss  = (weights * ((preds - targets) ** 2).sum(dim=1)).mean()

                    opt.zero_grad(set_to_none=True)
                    loss.backward()
                    torch.nn.utils.clip_grad_norm_(
                        self.adv_nets[player][rnd].parameters(), max_norm=1.0
                    )
                    opt.step()

                losses.append(loss.item() * (ADV_SCALE ** 2))

        if losses:
            self.adv_loss = float(np.mean(losses))

    def train_strat_network(self):
        """Train all 4 per-round strategy networks."""
        losses = []
        for rnd in range(N_ROUNDS):
            buf = self.strat_buffers[rnd]
            n   = len(buf)
            if n < 64:
                continue

            B     = min(n, 4096)
            steps = 100

            # Reset first-moment momentum to avoid carryover from previous event.
            for state in self.strat_opts[rnd].state.values():
                if 'exp_avg' in state:
                    state['exp_avg'].zero_()
                if 'step' in state:
                    state['step'] = state['step'] * 0

            self.strat_nets[rnd].train()
            recent_losses = []
            for step_i in range(steps):
                states, targets, weights = buf.sample(B)
                states  = states.to(self.device)
                targets = targets.to(self.device)
                weights = (weights * (B / weights.sum())).to(self.device)

                preds     = self.strat_nets[rnd](states)
                log_probs = F.log_softmax(preds, dim=1)
                loss      = -(weights * (targets * log_probs).sum(dim=1)).mean()

                self.strat_opts[rnd].zero_grad(set_to_none=True)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(
                    self.strat_nets[rnd].parameters(), max_norm=1.0
                )
                self.strat_opts[rnd].step()

                if step_i >= steps - 20:
                    recent_losses.append(loss.item())

            if recent_losses:
                rnd_loss = float(np.mean(recent_losses))
                losses.append(rnd_loss)
                self.strat_schedulers[rnd].step(rnd_loss)

        if losses:
            self.strat_loss = float(np.mean(losses))

    # -------------------------------------------------------------------------
    # C++ interface helpers
    # -------------------------------------------------------------------------
    def _export_adv_nets(self):
        """Push all 8 advantage network weights into the C++ model cache.

        Batching eval()/train() mode switches outside the per-network loop
        avoids redundant Python attribute scans on every network object.
        torch.no_grad() eliminates autograd bookkeeping during the export.
        """
        import poker_cpp
        for p in range(2):
            for r in range(N_ROUNDS):
                self.adv_nets[p][r].eval()
        with torch.no_grad():
            for player in range(2):
                for rnd in range(N_ROUNDS):
                    net = self.adv_nets[player][rnd]
                    poker_cpp.update_model_cache(
                        player, rnd,
                        net.fc1.weight.cpu().numpy(),
                        net.fc1.bias.cpu().numpy(),
                        net.fc2.weight.cpu().numpy(),
                        net.fc2.bias.cpu().numpy(),
                        net.out.weight.cpu().numpy(),
                        net.out.bias.cpu().numpy(),
                    )
        for p in range(2):
            for r in range(N_ROUNDS):
                self.adv_nets[p][r].train()

    def _buf_numpy(self, buf, field):
        return getattr(buf, field).numpy()

    def _submit_traversals(self, k_traversals, t):
        """Launch C++ run_traversals asynchronously; returns a Future."""
        import poker_cpp

        # 8 advantage buffer tuples in player*4+round order
        adv_buf_list = []
        for player in range(2):
            for rnd in range(N_ROUNDS):
                buf = self.adv_buffers[player][rnd]
                adv_buf_list.append((
                    self._buf_numpy(buf, 'states'),
                    self._buf_numpy(buf, 'targets'),
                    self._buf_numpy(buf, 'weights'),
                    buf.capacity,
                    buf.n_inserted,
                ))

        # 4 strategy buffer tuples in round order
        strat_buf_list = []
        for rnd in range(N_ROUNDS):
            buf = self.strat_buffers[rnd]
            strat_buf_list.append((
                self._buf_numpy(buf, 'states'),
                self._buf_numpy(buf, 'targets'),
                self._buf_numpy(buf, 'weights'),
                buf.capacity,
                buf.n_inserted,
            ))

        return self._executor.submit(
            poker_cpp.run_traversals,
            k_traversals, t,
            adv_buf_list,
            strat_buf_list,
        )

    def _sync_counts(self, counts):
        """Sync C++ insertion counts back into Python buffer objects."""
        for player in range(2):
            for rnd in range(N_ROUNDS):
                key = f'adv_n_{player * N_ROUNDS + rnd}'
                self.adv_buffers[player][rnd].n_inserted = counts[key]
        for rnd in range(N_ROUNDS):
            self.strat_buffers[rnd].n_inserted = counts[f'strat_n_{rnd}']

    # -------------------------------------------------------------------------
    # Main training loop
    # -------------------------------------------------------------------------
    def train(self, n_iterations=1, k_traversals=200):
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations

            self._export_adv_nets()
            future = self._submit_traversals(k_traversals, t)
            self.train_adv_network()

            if self.iterations % 25 == 0:
                self.train_strat_network()

            self._sync_counts(future.result())

    # -------------------------------------------------------------------------
    # Strategy evaluation
    # -------------------------------------------------------------------------
    def get_average_strategy_from_tensor(self, tensor, legal_actions, round=0):
        """Evaluate average strategy using the per-round strategy network.

        Args:
            tensor:        122-dim state tensor.
            legal_actions: list of legal action indices.
            round:         betting round index 0-3 (selects strat net specialist).

        Returns:
            np.ndarray of shape (N_ACTIONS,) with probabilities summing to 1.
        """
        with torch.no_grad():
            self.strat_nets[round].eval()
            logits = self.strat_nets[round](
                tensor.to(self.device).unsqueeze(0)
            ).squeeze(0)
            probs = F.softmax(logits, dim=0).cpu().numpy()

        strategy = np.zeros(N_ACTIONS)
        sum_probs = np.sum(probs[legal_actions])
        if sum_probs > 0:
            strategy[legal_actions] = probs[legal_actions] / sum_probs
        else:
            strategy[legal_actions] = 1.0 / len(legal_actions)
        return strategy

    # -------------------------------------------------------------------------
    # Checkpoint save / load
    # -------------------------------------------------------------------------
    def save_checkpoint(self, filepath, verbose=True):
        import os
        checkpoint = {
            'iterations': self.iterations,
            'adv_loss':   self.adv_loss,
            'strat_loss': self.strat_loss,
            # 2D list [player][round]
            'adv_nets': [[self.adv_nets[p][r].state_dict()
                          for r in range(N_ROUNDS)] for p in range(2)],
            # 1D list [round]
            'strat_nets': [self.strat_nets[r].state_dict() for r in range(N_ROUNDS)],
            'adv_opts': [[self.adv_opts[p][r].state_dict()
                          for r in range(N_ROUNDS)] for p in range(2)],
            'strat_opts': [self.strat_opts[r].state_dict() for r in range(N_ROUNDS)],
            'strat_schedulers': [self.strat_schedulers[r].state_dict()
                                 for r in range(N_ROUNDS)],
            # Advantage buffers: flat list in player*4+round order
            'adv_buffers': [
                (self.adv_buffers[p][r].capacity,
                 self.adv_buffers[p][r].states,
                 self.adv_buffers[p][r].targets,
                 self.adv_buffers[p][r].weights,
                 self.adv_buffers[p][r].n_inserted)
                for p in range(2) for r in range(N_ROUNDS)
            ],
            # Strategy buffers: list in round order
            'strat_buffers': [
                (self.strat_buffers[r].capacity,
                 self.strat_buffers[r].states,
                 self.strat_buffers[r].targets,
                 self.strat_buffers[r].weights,
                 self.strat_buffers[r].n_inserted)
                for r in range(N_ROUNDS)
            ],
        }
        tmp_path = filepath + '.tmp'
        torch.save(checkpoint, tmp_path)
        os.replace(tmp_path, filepath)
        if verbose:
            print(f"Checkpoint saved to {filepath}")

    def load_checkpoint(self, filepath):
        import os
        if not os.path.exists(filepath):
            print(f"No checkpoint found at {filepath}, starting fresh.")
            return False

        checkpoint = torch.load(filepath, weights_only=False)
        self.iterations = checkpoint['iterations']
        self.adv_loss   = checkpoint.get('adv_loss', 0.0)
        self.strat_loss = checkpoint.get('strat_loss', 0.0)

        # ---- Advantage networks ----
        adv_nets_data = checkpoint.get('adv_nets')
        if adv_nets_data is not None:
            try:
                if isinstance(adv_nets_data[0], list):
                    for p in range(2):
                        for r in range(N_ROUNDS):
                            self.adv_nets[p][r].load_state_dict(adv_nets_data[p][r])
                else:
                    print("  [!] Old flat-list adv_nets checkpoint. Starting fresh.")
                    return False
            except RuntimeError:
                print("  [!] Advantage net architecture mismatch. Starting fresh.")
                return False

        # ---- Strategy networks ----
        strat_nets_data = checkpoint.get('strat_nets')
        if strat_nets_data is not None:
            if isinstance(strat_nets_data, list):
                # New per-round format
                for r in range(N_ROUNDS):
                    try:
                        self.strat_nets[r].load_state_dict(strat_nets_data[r])
                    except RuntimeError:
                        print(f"  [!] Strat net round {r} architecture mismatch. Initializing fresh.")
            else:
                # Old single-network format — skip (architecture changed)
                print("  [!] Old single strat_net checkpoint. Starting fresh strat nets.")
        elif checkpoint.get('strat_net'):
            print("  [!] Old single strat_net checkpoint. Starting fresh strat nets.")

        # ---- Advantage optimizers ----
        adv_opts_data = checkpoint.get('adv_opts')
        if adv_opts_data is not None and isinstance(adv_opts_data[0], list):
            for p in range(2):
                for r in range(N_ROUNDS):
                    try:
                        self.adv_opts[p][r].load_state_dict(adv_opts_data[p][r])
                    except Exception:
                        pass

        # ---- Strategy optimizers & schedulers ----
        strat_opts_data = checkpoint.get('strat_opts')
        if strat_opts_data is not None and isinstance(strat_opts_data, list):
            for r in range(N_ROUNDS):
                try:
                    self.strat_opts[r].load_state_dict(strat_opts_data[r])
                except Exception:
                    pass
        strat_sched_data = checkpoint.get('strat_schedulers')
        if strat_sched_data is not None and isinstance(strat_sched_data, list):
            for r in range(N_ROUNDS):
                try:
                    self.strat_schedulers[r].load_state_dict(strat_sched_data[r])
                except Exception:
                    pass

        # Unfreeze LR if old checkpoint decayed it to minimum
        for r in range(N_ROUNDS):
            for pg in self.strat_opts[r].param_groups:
                if pg['lr'] < 2e-5:
                    print(f"  Resetting strat[{r}] LR: {pg['lr']:.2e} → 1e-04")
                    pg['lr'] = 0.0001

        # ---- Advantage buffers ----
        adv_buffers_data = checkpoint.get('adv_buffers')
        if adv_buffers_data is not None:
            for idx, buf_data in enumerate(adv_buffers_data):
                player = idx // N_ROUNDS
                rnd    = idx %  N_ROUNDS
                buf    = self.adv_buffers[player][rnd]
                chk_cap, states, targets, weights, n_inserted = buf_data
                n_copy = min(chk_cap, buf.capacity)
                buf.states[:n_copy]  = states[:n_copy]
                buf.targets[:n_copy] = targets[:n_copy]
                buf.weights[:n_copy] = weights[:n_copy]
                buf.n_inserted = n_inserted
                if buf.capacity != chk_cap:
                    print(f"  Resized adv buffer [{player}][{rnd}]: {chk_cap} → {buf.capacity}")

        # ---- Strategy buffers ----
        strat_buffers_data = checkpoint.get('strat_buffers')
        if strat_buffers_data is not None and isinstance(strat_buffers_data, list):
            for r, buf_data in enumerate(strat_buffers_data):
                buf = self.strat_buffers[r]
                chk_cap, states, targets, weights, n_inserted = buf_data
                n_copy = min(chk_cap, buf.capacity)
                buf.states[:n_copy]  = states[:n_copy]
                buf.targets[:n_copy] = targets[:n_copy]
                buf.weights[:n_copy] = weights[:n_copy]
                buf.n_inserted = n_inserted
                if buf.capacity != chk_cap:
                    print(f"  Resized strat buffer [{r}]: {chk_cap} → {buf.capacity}")
        elif checkpoint.get('strat_buffer'):
            # Old single-buffer format: distribute across round 0 only as a seed
            buf_data = checkpoint['strat_buffer']
            chk_cap, states, targets, weights, n_inserted = buf_data
            n_copy = min(chk_cap, self.strat_buffers[0].capacity)
            self.strat_buffers[0].states[:n_copy]  = states[:n_copy]
            self.strat_buffers[0].targets[:n_copy] = targets[:n_copy]
            self.strat_buffers[0].weights[:n_copy] = weights[:n_copy]
            self.strat_buffers[0].n_inserted = n_copy
            print("  Migrated old single strat_buffer into strat_buffers[0].")

        # Clamp n_inserted to 2×capacity to restore ~50% refresh rate
        all_bufs = ([self.adv_buffers[p][r] for p in range(2) for r in range(N_ROUNDS)]
                    + self.strat_buffers)
        for buf in all_bufs:
            cap2 = buf.capacity * 2
            if buf.n_inserted > cap2:
                print(f"  Clamping buffer n_inserted: {buf.n_inserted:,} → {cap2:,}")
                buf.n_inserted = cap2

        print(f"Loaded checkpoint from {filepath} (iteration {self.iterations})")
        return True
