import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np
import random
from numba import njit
from poker_env import N_CARDS, card_str
from poker_env import LeducState, FOLD, CALL, RAISE

# ---------------------------------------------------------------------------
# Constants
# ---------------------------------------------------------------------------
N_ACTIONS   = 3
_CARD_STR   = ['Jc', 'Qc', 'Kc', 'Js', 'Qs', 'Ks']
_HIST_CHARS = ['f', 'c', 'r']

_DEALS = [
    (p0, p1, comm)
    for p0   in range(N_CARDS)
    for p1   in range(N_CARDS) if p1 != p0
    for comm in range(N_CARDS) if comm != p0 and comm != p1
]
N_DEALS = len(_DEALS)   # 120


# ---------------------------------------------------------------------------
# Game-tree builder  (runs once at init)
# ---------------------------------------------------------------------------

def _make_state(p0, p1, comm):
    s = LeducState.__new__(LeducState)
    s.private   = [p0, p1]
    s.community = comm
    s.round     = 0
    s.pot       = 2
    s.stacks    = [9, 9]
    s.bets      = [1, 1]
    s.to_act    = 0
    s.raises    = 0
    s.done      = False
    s.winner    = None
    s.history   = []
    s.n_acted   = 0
    return s


def _info_key(state, player):

    priv = _CARD_STR[state.private[player]]

    comm = (
        _CARD_STR[state.community]
        if state.round == 1
        else 'none'
    )

    hist = ''.join(
        _HIST_CHARS[a]
        for a in state.history
    )

    return f'P{player}|{priv}|{comm}|{hist}'


def _build_tree_arrays():
    """
    Build flat numpy arrays representing the full game tree.
    Critically, also builds per-deal topological orderings so that
    each deal's CFR pass uses only its own nodes.
    """
    info_ids      = {}
    info_legal_l  = []
    node_info_l   = []
    node_player_l = []
    node_term_l   = []
    node_payoff_l = []
    node_child_l  = []
    deal_roots    = []

    def build(state):
        nid = len(node_info_l)
        if state.done:
            node_info_l.append(-1)
            node_player_l.append(-1)
            node_term_l.append(True)
            node_payoff_l.append(float(state.payoff(0)))
            node_child_l.append([-1, -1, -1])
            return nid

        player = state.to_act
        legal  = state.legal_actions()
        key    = _info_key(state, player)

        if key not in info_ids:
            info_ids[key] = len(info_ids)
            mask = [False, False, False]
            for a in legal: mask[a] = True
            info_legal_l.append(mask)

        node_info_l.append(info_ids[key])
        node_player_l.append(player)
        node_term_l.append(False)
        node_payoff_l.append(0.0)
        node_child_l.append([-1, -1, -1])

        children = [-1, -1, -1]
        for a in legal:
            ns = state.copy()
            ns.apply_action(a)
            children[a] = build(ns)
        node_child_l[nid] = children
        return nid

    for p0, p1, comm in _DEALS:
        deal_roots.append(build(_make_state(p0, p1, comm)))

    # Per-deal post-order topological sort
    # Each deal's topo contains only that deal's nodes — no cross-contamination
    deal_topos = []
    for root in deal_roots:
        visited = []
        def postorder(nid):
            for cid in node_child_l[nid]:
                if cid >= 0:
                    postorder(cid)
            visited.append(nid)
        postorder(root)
        deal_topos.append(np.array(visited, dtype=np.int32))

    # Flatten for Numba (single contiguous array + sizes)
    flat_topo  = np.concatenate(deal_topos).astype(np.int32)
    topo_sizes = np.array([len(dt) for dt in deal_topos], dtype=np.int32)

    n_info       = len(info_ids)
    info_str_map = {v: k for k, v in info_ids.items()}

    return (
        flat_topo,
        topo_sizes,
        np.array(node_term_l,   dtype=np.bool_),
        np.array(node_payoff_l, dtype=np.float64),
        np.array(node_player_l, dtype=np.int8),
        np.array(node_info_l,   dtype=np.int32),
        np.array(node_child_l,  dtype=np.int32),
        deal_roots,
        n_info,
        info_str_map,
        info_ids,
        np.array(info_legal_l, dtype=np.bool_),
    )


# ---------------------------------------------------------------------------
# Numba JIT CFR kernel
# ---------------------------------------------------------------------------

@njit(cache=True)
def _cfr_iteration(flat_topo, topo_sizes,
                   node_terminal, node_payoff, node_player,
                   node_info, node_children,
                   info_legal, regret, strat_sum,
                   fixed_player=-1, fixed_strat=np.zeros((1,3)), fixed_strat2=np.zeros((1,3))):

    n_info = regret.shape[0]
    n_nodes = node_terminal.shape[0]
    N_DEALS = len(topo_sizes)

    # Current strategy from cumulative regrets
    strat = np.zeros((n_info, 3))
    
    # Pre-fill strat for fixed player
    if fixed_player == -2:
        # Both players are fixed
        for iid in range(n_info):
            s1 = 0.0
            s2 = 0.0
            for a in range(3):
                s1 += fixed_strat[iid, a]
                s2 += fixed_strat2[iid, a]
            if s1 > 0:
                for a in range(3):
                    strat[iid, a] = fixed_strat[iid, a]
            elif s2 > 0:
                for a in range(3):
                    strat[iid, a] = fixed_strat2[iid, a]
    elif fixed_player != -1:
        for iid in range(n_info):
            for a in range(3):
                strat[iid, a] = fixed_strat[iid, a]

    for iid in range(n_info):
        # Skip regret matching if this info set is for the fixed player
        # We don't strictly know if it is the fixed player's turn here, 
        # but we can just do regret matching for all and override it during backward pass?
        # No, wait. We populated strat above. Let's just do regret matching for ALL info sets,
        # but then explicitly OVERWRITE the strat with fixed_strat.
        ps = 0.0
        n_legal = 0
        for a in range(3):
            if info_legal[iid, a]:
                n_legal += 1
                if regret[iid, a] > 0:
                    ps += regret[iid, a]

        if ps > 0:
            for a in range(3):
                if info_legal[iid, a] and regret[iid, a] > 0:
                    strat[iid, a] = regret[iid, a] / ps
        else:
            for a in range(3):
                if info_legal[iid, a]:
                    strat[iid, a] = 1.0 / n_legal

    # Now override with fixed strategy
    if fixed_player == -2:
        for iid in range(n_info):
            s1 = 0.0
            s2 = 0.0
            for a in range(3):
                s1 += fixed_strat[iid, a]
                s2 += fixed_strat2[iid, a]
            if s1 > 0:
                for a in range(3):
                    strat[iid, a] = fixed_strat[iid, a]
            elif s2 > 0:
                for a in range(3):
                    strat[iid, a] = fixed_strat2[iid, a]
    elif fixed_player != -1:
        for iid in range(n_info):
            s = 0.0
            for a in range(3):
                s += fixed_strat[iid, a]
            if s > 0.0:
                for a in range(3):
                    strat[iid, a] = fixed_strat[iid, a]

    total_utility = 0.0
    offset = 0

    for d in range(N_DEALS):

        dsize = topo_sizes[d]
        dtopo = flat_topo[offset:offset+dsize]
        offset += dsize

        rp0 = np.zeros(n_nodes)
        rp1 = np.zeros(n_nodes)
        value = np.zeros(n_nodes)

        root = dtopo[dsize - 1]

        chance_prob = 1.0 / N_DEALS

        rp0[root] = chance_prob
        rp1[root] = chance_prob

        # Forward reach pass
        for idx in range(dsize-1, -1, -1):

            nid = dtopo[idx]

            if node_terminal[nid]:
                continue

            p = node_player[nid]
            iid = node_info[nid]

            for a in range(3):

                cid = node_children[nid, a]

                if cid < 0:
                    continue

                if p == 0:
                    rp0[cid] += rp0[nid] * strat[iid, a]
                    rp1[cid] += rp1[nid]
                else:
                    rp0[cid] += rp0[nid]
                    rp1[cid] += rp1[nid] * strat[iid, a]

        # Terminal values
        for idx in range(dsize):

            nid = dtopo[idx]

            if node_terminal[nid]:
                value[nid] = node_payoff[nid]

        # Backward CFR pass
        for idx in range(dsize):

            nid = dtopo[idx]

            if node_terminal[nid]:
                continue

            p = node_player[nid]
            iid = node_info[nid]

            c0 = node_children[nid,0]
            c1 = node_children[nid,1]
            c2 = node_children[nid,2]

            av0 = value[c0] if c0 >= 0 else 0
            av1 = value[c1] if c1 >= 0 else 0
            av2 = value[c2] if c2 >= 0 else 0

            s0 = strat[iid,0]
            s1 = strat[iid,1]
            s2 = strat[iid,2]

            nv = s0*av0 + s1*av1 + s2*av2
            value[nid] = nv

            # ---- average strategy accumulation ----
            player_reach = rp0[nid] if p == 0 else rp1[nid]

            if fixed_player != -2 and p != fixed_player:
                strat_sum[iid,0] += player_reach * s0
                strat_sum[iid,1] += player_reach * s1
                strat_sum[iid,2] += player_reach * s2

            # ---- regret update ----
            if fixed_player != -2 and p != fixed_player:
                if p == 0:
                    if c0 >= 0: regret[iid,0] += rp1[nid]*(av0 - nv)
                    if c1 >= 0: regret[iid,1] += rp1[nid]*(av1 - nv)
                    if c2 >= 0: regret[iid,2] += rp1[nid]*(av2 - nv)
                else:
                    if c0 >= 0: regret[iid,0] += rp0[nid]*(nv - av0)
                    if c1 >= 0: regret[iid,1] += rp0[nid]*(nv - av1)
                    if c2 >= 0: regret[iid,2] += rp0[nid]*(nv - av2)

        total_utility += value[root]

    return total_utility / N_DEALS


# ==============================================================================
# Networks
# ==============================================================================

class DeepCFRNetwork(nn.Module):
    def __init__(self, n_actions=3):
        super().__init__()
        # Features (38 dimensions):
        # 0: player
        # 1-3: private card (J, Q, K)
        # 4-8: community card (None, J, Q, K)
        # 9-38: history
        self.fc1 = nn.Linear(38, 128)
        self.fc2 = nn.Linear(128, 128)
        self.fc3 = nn.Linear(128, 128)
        self.out = nn.Linear(128, n_actions)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        x = F.relu(self.fc3(x))
        return self.out(x)

def encode_state(state, player):
    """Encodes LeducState into a 38-dim tensor for the given player."""
    tensor = torch.zeros(38, dtype=torch.float32)
    
    tensor[0] = player

    # Private card
    priv = state.private[player] # 0-5
    rank = priv % 3 # 0=J, 1=Q, 2=K
    tensor[1 + rank] = 1.0

    # Community card
    if state.round == 0:
        tensor[4] = 1.0 # None
    else:
        comm_rank = state.community % 3
        tensor[5 + comm_rank] = 1.0

    # History
    offset = 9
    for i, a in enumerate(state.history):
        if i >= 9: break
        tensor[offset + i*3 + a] = 1.0

    return tensor

# ==============================================================================
# Memory Buffers (Reservoir Sampling)
# ==============================================================================

class ReservoirBuffer:
    def __init__(self, capacity):
        self.capacity = capacity
        self.memory = []
        self.n_inserted = 0

    def push(self, item):
        if len(self.memory) < self.capacity:
            self.memory.append(item)
        else:
            idx = random.randint(0, self.n_inserted)
            if idx < self.capacity:
                self.memory[idx] = item
        self.n_inserted += 1

    def sample(self, batch_size):
        return random.sample(self.memory, min(batch_size, len(self.memory)))

    def __len__(self):
        return len(self.memory)

# ==============================================================================
# Deep CFR Trainer
# ==============================================================================

class DeepCFRTrainer:
    def __init__(self, adv_lr=0.001, strat_lr=0.001, adv_buffer_size=100000, strat_buffer_size=100000):
        # Force CPU. Apple Silicon CPUs are significantly faster than MPS for batch-size=1 inference
        self.device = torch.device("cpu")
        print(f"Deep CFR initialized using device: {self.device}")

        # Networks
        self.adv_net = DeepCFRNetwork().to(self.device)
        self.strat_net = DeepCFRNetwork().to(self.device)

        # Optimizers
        self.adv_opt = torch.optim.Adam(self.adv_net.parameters(), lr=adv_lr)
        self.strat_opt = torch.optim.Adam(self.strat_net.parameters(), lr=strat_lr)

        # Buffers
        self.adv_buffer = ReservoirBuffer(adv_buffer_size)
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

        # Get regrets from advantage network
        with torch.no_grad():
            advantages = self.adv_net(encoded).squeeze(0).cpu().numpy()

        # Regret matching
        strategy = np.zeros(3)
        regrets = np.maximum(advantages, 0)
        legal_regrets = regrets[legal_actions]
        sum_regrets = np.sum(legal_regrets)

        if sum_regrets > 0:
            for a in legal_actions:
                strategy[a] = regrets[a] / sum_regrets
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        # If we are the traverser, explore all actions
        if player == traverser:
            action_values = np.zeros(3)
            for a in legal_actions:
                action_values[a] = self.traverse(state.copy().apply_action(a), traverser, t)
            
            ev = np.sum(strategy[legal_actions] * action_values[legal_actions])

            # Calculate actual sampled advantages
            sampled_advantages = np.zeros(3)
            for a in legal_actions:
                sampled_advantages[a] = action_values[a] - ev

            # Push to advantage buffer
            # (state_tensor, advantages, iteration_t)
            self.adv_buffer.push((encoded.cpu().squeeze(0), sampled_advantages, t))

            return ev

        # If opponent, sample an action according to strategy
        else:
            # Push to strategy buffer
            self.strat_buffer.push((encoded.cpu().squeeze(0), strategy, t))

            p = [strategy[a] for a in legal_actions]
            p = np.array(p, dtype=np.float64)
            p /= np.sum(p)
            action = np.random.choice(legal_actions, p=p)
            return self.traverse(state.copy().apply_action(action), traverser, t)

    def train_networks(self, batch_size=1024, adv_steps=200, strat_steps=200):
        # Train Advantage Network
        if len(self.adv_buffer) > batch_size:
            self.adv_net.train()
            for _ in range(adv_steps):
                batch = self.adv_buffer.sample(batch_size)
                states = torch.stack([x[0] for x in batch]).to(self.device)
                targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)

                preds = self.adv_net(states)
                loss = ((preds - targets) ** 2).mean()

                self.adv_opt.zero_grad()
                loss.backward()
                self.adv_opt.step()
            self.adv_loss = loss.item()

        # Train Strategy Network
        if len(self.strat_buffer) > batch_size:
            self.strat_net.train()
            for _ in range(strat_steps):
                batch = self.strat_buffer.sample(batch_size)
                states = torch.stack([x[0] for x in batch]).to(self.device)
                targets = torch.tensor(np.array([x[1] for x in batch]), dtype=torch.float32).to(self.device)
                t_weights = torch.tensor(np.array([x[2] for x in batch]), dtype=torch.float32).to(self.device)
                
                # Normalize t_weights to prevent exploding gradients
                t_weights = t_weights / t_weights.mean()

                preds = self.strat_net(states)
                # Cross Entropy: targets are probability distributions
                log_probs = F.log_softmax(preds, dim=1)
                loss = - (t_weights.unsqueeze(1) * targets * log_probs).sum(dim=1).mean()

                self.strat_opt.zero_grad()
                loss.backward()
                self.strat_opt.step()
            self.strat_loss = loss.item()

    def train(self, n_iterations=1, k_traversals=1):
        from poker_env import LeducState
        for _ in range(n_iterations):
            self.iterations += 1
            t = self.iterations

            # Generate data via MCCFR
            for _ in range(k_traversals):
                # Sample a single random deal (External Sampling MCCFR)
                p0, p1, comm = random.choice(_DEALS)
                
                # Traversal for Player 0
                s0 = LeducState()
                s0.private = [p0, p1]
                s0.community = comm
                self.traverse(s0, 0, t)

                # Traversal for Player 1
                s1 = LeducState()
                s1.private = [p0, p1]
                s1.community = comm
                self.traverse(s1, 1, t)

            # Train networks periodically (batching)
            if self.iterations % 100 == 0:
                self.train_networks()

    def get_average_strategy(self, info_set_str, legal_actions):
        """Used by server.py to display the UI"""
        # We need to construct a dummy state just to run encode_state
        # Or parse the string. Let's parse the string like before.
        parts = info_set_str.split('|')
        player = int(parts[0][1])
        priv = parts[1]
        comm = parts[2]
        hist = parts[3]
        
        tensor = torch.zeros(38, dtype=torch.float32)
        tensor[0] = player
        rank_to_idx = {'J': 0, 'Q': 1, 'K': 2}
        if priv[0] in rank_to_idx: tensor[1 + rank_to_idx[priv[0]]] = 1.0
        if comm == 'none': tensor[4] = 1.0
        elif comm[0] in rank_to_idx: tensor[5 + rank_to_idx[comm[0]]] = 1.0
        action_to_idx = {'f': 0, 'c': 1, 'r': 2}
        offset = 9
        for i, a in enumerate(hist):
            if i >= 9: break
            idx = action_to_idx.get(a)
            if idx is not None: tensor[offset + i*3 + idx] = 1.0

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

    def compute_exploitability(self):
        # FUTURE-PROOFING GUARD: Tabular exact Best Response requires building the full game tree.
        # This is extremely fast for Leduc (5,000 nodes) but physically impossible for Texas Hold'em (10^161 nodes).
        # For full-scale games, we must implement Local Best Response (LBR) or train an Approximate Best Response neural network.
        if len(_DEALS) > 120:
            raise NotImplementedError("Exact tabular exploitability does not scale beyond Leduc. Implement LBR for full-scale Poker.")

        # 1. Build the tabular game tree once if not already cached
        if not hasattr(self, '_tabular_cache'):
            self._tabular_cache = _build_tree_arrays()
        
        (flat_topo, topo_sizes, node_terminal, node_payoff, node_player,
         node_info, node_children, deal_roots, n_info, info_str_map,
         info_ids, info_legal) = self._tabular_cache

        # 2. Extract full average strategy from Neural Network
        nn_strat = np.zeros((n_info, 3), dtype=np.float64)
        for iid in range(n_info):
            info_str = info_str_map[iid]
            # Find legal actions
            legal = []
            for a in range(3):
                if info_legal[iid, a]:
                    legal.append(a)
            
            if legal:
                strat = self.get_average_strategy(info_str, legal)
                for a in legal:
                    nn_strat[iid, a] = strat[a]

        # 3. Compute Best Response for Player 0 (P1 is fixed to nn_strat)
        regret0 = np.zeros((n_info, 3), dtype=np.float64)
        strat_sum0 = np.zeros((n_info, 3), dtype=np.float64)
        
        # We need to construct fixed_strat such that P1's info sets have sum > 0
        fixed_strat1 = np.zeros((n_info, 3), dtype=np.float64)
        for iid in range(n_info):
            if info_str_map[iid].startswith('P1'):
                fixed_strat1[iid] = nn_strat[iid]

        # Train P0 Best Response
        for _ in range(500):
            _cfr_iteration(
                flat_topo, topo_sizes, node_terminal, node_payoff, node_player,
                node_info, node_children, info_legal, regret0, strat_sum0,
                fixed_player=1, fixed_strat=fixed_strat1
            )
            
        # Normalize P0's best response strategy from strat_sum0
        br_strat0 = np.zeros((n_info, 3), dtype=np.float64)
        for iid in range(n_info):
            if info_str_map[iid].startswith('P0'):
                s = np.sum(strat_sum0[iid])
                if s > 0:
                    br_strat0[iid] = strat_sum0[iid] / s
                else:
                    br_strat0[iid] = 1.0 / 3.0
                
        # Evaluate Exact EV for BR0 vs P1
        # If both players are fixed, _cfr_iteration returns the exact expected value for P0!
        br0_val = _cfr_iteration(
            flat_topo, topo_sizes, node_terminal, node_payoff, node_player,
            node_info, node_children, info_legal, regret0, strat_sum0,
            fixed_player=-2, fixed_strat=br_strat0, fixed_strat2=fixed_strat1
        )

        # 4. Compute Best Response for Player 1 (P0 is fixed to nn_strat)
        regret1 = np.zeros((n_info, 3), dtype=np.float64)
        strat_sum1 = np.zeros((n_info, 3), dtype=np.float64)
        
        fixed_strat0 = np.zeros((n_info, 3), dtype=np.float64)
        for iid in range(n_info):
            if info_str_map[iid].startswith('P0'):
                fixed_strat0[iid] = nn_strat[iid]

        # Train P1 Best Response
        for _ in range(500):
            _cfr_iteration(
                flat_topo, topo_sizes, node_terminal, node_payoff, node_player,
                node_info, node_children, info_legal, regret1, strat_sum1,
                fixed_player=0, fixed_strat=fixed_strat0
            )

        # Normalize P1's best response strategy from strat_sum1
        br_strat1 = np.zeros((n_info, 3), dtype=np.float64)
        for iid in range(n_info):
            if info_str_map[iid].startswith('P1'):
                s = np.sum(strat_sum1[iid])
                if s > 0:
                    br_strat1[iid] = strat_sum1[iid] / s
                else:
                    br_strat1[iid] = 1.0 / 3.0
                
        # Evaluate Exact EV for P0 vs BR1
        br1_val = _cfr_iteration(
            flat_topo, topo_sizes, node_terminal, node_payoff, node_player,
            node_info, node_children, info_legal, regret1, strat_sum1,
            fixed_player=-2, fixed_strat=fixed_strat0, fixed_strat2=br_strat1
        )


        # The game value is theoretically ~0.0, but P0 has a slight disadvantage (-0.08)
        # So br0_val is P0's expected value when P0 plays Best Response.
        # br1_val is P0's expected value when P1 plays Best Response. 
        # Therefore, P1's expected value is -br1_val.
        # True exploitability is the average of what each player can exploit:
        # Exploitability = (EV_BR0 + EV_BR1) / 2 = (br0_val - br1_val) / 2.0
        return (br0_val - br1_val) / 2.0
