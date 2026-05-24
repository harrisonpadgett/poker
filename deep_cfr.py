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
    for card in state.visible_community:
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
        # Force CPU. Apple Silicon CPUs are significantly faster than MPS for batch-size=1 inference
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

    def train_adv_network(self, batch_size=1024, adv_steps=300):
        losses = []
        for player in range(2):
            buf = self.adv_buffers[player]
            if len(buf) > batch_size:
                self.adv_nets[player] = DeepCFRNetwork().to(self.device)
                self.adv_opts[player] = torch.optim.Adam(self.adv_nets[player].parameters(), lr=0.001)

                self.adv_nets[player].train()
                for _ in range(adv_steps):
                    states, targets, weights = buf.sample(batch_size)
                    states = states.to(self.device)
                    targets = targets.to(self.device)
                    weights = weights.to(self.device)
                    weights = weights / weights.sum() * len(weights)

                    preds = self.adv_nets[player](states)
                    loss  = (weights * ((preds - targets) ** 2).sum(dim=1)).mean()

                    self.adv_opts[player].zero_grad()
                    loss.backward()
                    self.adv_opts[player].step()
                losses.append(loss.item())
        if losses:
            self.adv_loss = float(np.mean(losses))

    def train_strat_network(self, batch_size=1024, strat_steps=2000):
        if len(self.strat_buffer) > batch_size:
            self.strat_net.train()
            for _ in range(strat_steps):
                states, targets, weights = self.strat_buffer.sample(batch_size)
                states = states.to(self.device)
                targets = targets.to(self.device)
                weights = weights.to(self.device)
                weights = weights / weights.sum() * len(weights)

                preds = self.strat_net(states)

                log_probs = F.log_softmax(preds, dim=1)
                loss = -(weights * (targets * log_probs).sum(dim=1)).mean()

                self.strat_opt.zero_grad()
                loss.backward()
                self.strat_opt.step()
            self.strat_loss = loss.item()
            self.strat_scheduler.step(self.strat_loss)

    def train(self, n_iterations=1, k_traversals=100):
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations

            # External Sampling MCCFR: each traversal samples a fresh deal
            for _ in range(k_traversals):
                state = RoyalState().reset()
                self.traverse(state.copy(), 0, t)
                self.traverse(state.copy(), 1, t)

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
