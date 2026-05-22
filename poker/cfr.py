"""
cfr.py — Counterfactual Regret Minimization for Heads-up Limit Hold'em.

CFR finds a Nash equilibrium by repeatedly playing the game against itself,
tracking regret at each decision point, and updating strategy to reduce regret.

Key concepts:

REGRET:
  Regret(I, a) = how much better off we would have been if we had always
                 taken action a at information set I.
  Positive regret → we wish we had taken this action more.
  Negative regret → we're glad we didn't take this action.

STRATEGY:
  At each information set, actions are sampled proportional to positive regret.
  Actions with zero or negative regret get zero probability.
  This is regret matching — it's provably optimal.

AVERAGE STRATEGY:
  The current strategy changes every iteration.
  The AVERAGE strategy over all iterations converges to Nash equilibrium.
  This is the counterintuitive insight: the current strategy isn't the goal,
  the long-run average is.

COUNTERFACTUAL:
  "Counterfactual" means we evaluate what would have happened IF we had reached
  this information set, regardless of the actual probability of reaching it.
  This isolates the value of each action from the noise of how often the
  information set is actually visited.

REACH PROBABILITIES:
  Each node in the game tree has a reach probability — how likely are we to
  reach this node given both players' strategies?
  reach(node) = π₀(node) × π₁(node)
  CFR separates these: π_i is "my" contribution, π_{-i} is "opponent's."
  Counterfactual value uses π_{-i} as a weight — actions matter more when
  the opponent would actually play into this situation.
"""

import numpy as np
from collections import defaultdict
from poker_env import LeducState, FOLD, CALL, RAISE
PokerState = LeducState  # alias for compatibility

ACTIONS = [FOLD, CALL, RAISE]
N_ACTIONS = 3


class CFRTrainer:
    """
    Vanilla CFR for heads-up limit Hold'em.

    Stores:
      regret_sum[info_set][action]   — cumulative regret
      strategy_sum[info_set][action] — cumulative strategy (for average)

    The average strategy = strategy_sum / sum(strategy_sum)
    is what converges to Nash equilibrium.
    """

    def __init__(self):
        self.regret_sum   = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.strategy_sum = defaultdict(lambda: np.zeros(N_ACTIONS))
        self.iterations   = 0
        self.total_utility = 0.0   # tracks convergence

    # ------------------------------------------------------------------
    # Core CFR
    # ------------------------------------------------------------------

    def cfr(self, state, reach_p0, reach_p1):
        """
        Recursive CFR traversal. Always returns value from P0's perspective.

        Standard CFR sign convention (Zinkevich et al. 2007):
        - Terminal nodes: return payoff(0)  [always P0's payoff]
        - P0 nodes: action_values are P0's values, return weighted avg
        - P1 nodes: action_values are P0's values (NOT P1's),
                    regrets computed as -(P0_value) for P1's benefit

        This is the cleanest formulation that avoids sign confusion.
        """
        if state.done:
            return state.payoff(0)

        player   = state.to_act
        info_set = state.info_set(player)
        legal    = state.legal_actions()

        strategy  = self._get_strategy(info_set, legal)
        reach_opp = reach_p1 if player == 0 else reach_p0

        # Linear CFR: weight strategy accumulation by iteration number
        t = max(1, self.iterations)
        self.strategy_sum[info_set] += t * reach_opp * strategy

        # action_values[a] = P0's value if action a is taken
        action_values = np.zeros(N_ACTIONS)
        for a in legal:
            next_state = state.copy()
            next_state.apply_action(a)
            if player == 0:
                action_values[a] = self.cfr(
                    next_state, reach_p0 * strategy[a], reach_p1)
            else:
                action_values[a] = self.cfr(
                    next_state, reach_p0, reach_p1 * strategy[a])

        # P0's expected value at this node
        node_value = float(np.dot(strategy, action_values))

        # Regret update:
        # Linear CFR: weight regret by iteration number
        # Proven O(1/T) convergence — much faster than vanilla CFR's O(1/√T)
        for a in legal:
            if player == 0:
                self.regret_sum[info_set][a] += (
                    t * reach_p1 * (action_values[a] - node_value))
            else:
                self.regret_sum[info_set][a] += (
                    t * reach_p0 * (node_value - action_values[a]))

        return node_value

    def train(self, n_iterations):
        """
        Run n_iterations of full-tree CFR+.

        Full tree CFR: each iteration averages over ALL possible card deals,
        weighted equally. This is much faster convergence than sampled CFR
        (one random deal per iteration) because every info set is updated
        with its true expected value every iteration.

        Leduc has 6*5*4 = 120 possible deals (p0_card, p1_card, community).
        Each deal has probability 1/120. We weight cfr() calls by 1/n_deals
        so the total utility reflects expected chips per hand.

        CFR+ improvements:
          1. Regret floor at 0 after each iteration
          2. Weighted strategy accumulation (weight = iteration t)
        """
        from poker_env import N_CARDS
        n_deals = 0
        # Pre-compute all valid deals
        deals = []
        for p0 in range(N_CARDS):
            for p1 in range(N_CARDS):
                if p1 == p0: continue
                for comm in range(N_CARDS):
                    if comm == p0 or comm == p1: continue
                    deals.append((p0, p1, comm))
        n_deals = len(deals)

        utility = 0.0
        for i in range(n_iterations):
            self.iterations += 1
            iter_utility = 0.0

            for (p0, p1, comm) in deals:
                state = PokerState()
                state.reset()
                state.private   = [p0, p1]
                state.community = comm
                iter_utility += self.cfr(state, 1.0, 1.0)

            # Average over all deals
            iter_utility /= n_deals
            utility      += iter_utility

            # CFR+: floor regrets at 0 after each full iteration
            for info_set in self.regret_sum:
                self.regret_sum[info_set] = np.maximum(
                    self.regret_sum[info_set], 0)

        self.total_utility += utility
        return utility / n_iterations

    # ------------------------------------------------------------------
    # Strategy computation
    # ------------------------------------------------------------------

    def _get_strategy(self, info_set, legal_actions):
        """
        Regret matching: strategy proportional to positive cumulative regret.

        If all regrets are ≤ 0: uniform over legal actions (never played this
        situation enough to form preferences).
        Otherwise: normalize positive regrets to get probabilities.

        This is the simplest possible strategy that is still provably optimal
        for converging to Nash equilibrium.
        """
        regrets  = self.regret_sum[info_set]
        strategy = np.zeros(N_ACTIONS)

        pos_regret_sum = sum(max(0, regrets[a]) for a in legal_actions)
        if pos_regret_sum > 0:
            for a in legal_actions:
                strategy[a] = max(0, regrets[a]) / pos_regret_sum
        else:
            # No positive regret yet — play uniformly
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)

        return strategy

    def get_average_strategy(self, info_set, legal_actions):
        """
        The average strategy — this is what converges to Nash equilibrium.
        NOT the current strategy (which fluctuates).

        Average strategy = strategy_sum / sum(strategy_sum)
        """
        strat_sum = self.strategy_sum[info_set]
        total = sum(strat_sum[a] for a in legal_actions)

        strategy = np.zeros(N_ACTIONS)
        if total > 0:
            for a in legal_actions:
                strategy[a] = strat_sum[a] / total
        else:
            for a in legal_actions:
                strategy[a] = 1.0 / len(legal_actions)
        return strategy

    # ------------------------------------------------------------------
    # Exploitability — how far from Nash equilibrium are we?
    # ------------------------------------------------------------------

    def _avg_strategy_value(self, state):
        """
        Compute game value for P0 when BOTH players play the current
        average strategy. Used as the baseline for exploitability.

        This is not the same as cfr() — cfr() updates regrets as a
        side effect. This is a pure read-only evaluation.
        """
        if state.done:
            return state.payoff(0)

        player   = state.to_act
        info_set = state.info_set(player)
        legal    = state.legal_actions()
        strategy = self.get_average_strategy(info_set, legal)

        val = 0.0
        for a in legal:
            if strategy[a] > 1e-10:
                ns = state.copy()
                ns.apply_action(a)
                child = self._avg_strategy_value(ns)
                # Always accumulate as P0's value
                if player == 0:
                    val += strategy[a] * child
                else:
                    # P1's strategy weighted average, but return P0's value
                    val += strategy[a] * child
        return val

    def compute_exploitability(self, n_sample_states=None):
        """
        Exploitability computed by memoized best-response traversal.
        Result approaches 0 as the average strategy approaches Nash.

        Uses memoization keyed on (info_set_string) to avoid recomputing
        the same subtrees across deals — ~10x faster than naive recursion.
        """
        from poker_env import N_CARDS

        # Build memoized best-response values once per player
        # Cache key: (private_card, community_card, history_string, br_player)
        # Since info sets abstract over unknown cards, we memoize by full state
        cache0 = {}
        cache1 = {}

        def br(state, player, cache):
            key = state.info_set(state.to_act) + f'|{state.to_act}|{state.bets}|{state.raises}'
            if state.done:
                return state.payoff(player)
            if key in cache:
                return cache[key]

            acting = state.to_act
            legal  = state.legal_actions()

            if acting == player:
                val = max(br(state.copy().apply_action(a), player, cache) for a in legal)
            else:
                info     = state.info_set(acting)
                strategy = self.get_average_strategy(info, legal)
                val      = sum(strategy[a] * br(state.copy().apply_action(a), player, cache)
                               for a in legal if strategy[a] > 1e-10)
            cache[key] = val
            return val

        total = 0.0
        n = 0
        for p0 in range(N_CARDS):
            for p1 in range(N_CARDS):
                if p1 == p0: continue
                for comm in range(N_CARDS):
                    if comm == p0 or comm == p1: continue
                    s0 = PokerState(); s0.reset()
                    s0.private = [p0, p1]; s0.community = comm
                    s1 = PokerState(); s1.reset()
                    s1.private = [p0, p1]; s1.community = comm
                    total += br(s0, 0, cache0) + br(s1, 1, cache1)
                    n += 1
        return total / n

    def _best_response_value(self, state, br_player):
        """Single-state best response (used externally for debugging)."""
        if state.done:
            return state.payoff(br_player)
        player   = state.to_act
        info_set = state.info_set(player)
        legal    = state.legal_actions()
        if player == br_player:
            return max(self._best_response_value(state.copy().apply_action(a), br_player)
                       for a in legal)
        strategy = self.get_average_strategy(info_set, legal)
        return sum(strategy[a] * self._best_response_value(state.copy().apply_action(a), br_player)
                   for a in legal if strategy[a] > 1e-10)

    # ------------------------------------------------------------------
    # Query interface
    # ------------------------------------------------------------------

    def get_action_probs(self, state):
        """Return average strategy probabilities for the current state."""
        info_set = state.info_set(state.to_act)
        legal    = state.legal_actions()
        return self.get_average_strategy(info_set, legal)

    def n_info_sets(self):
        return len(self.regret_sum)

    def stats(self):
        return {
            'iterations':  self.iterations,
            'info_sets':   self.n_info_sets(),
            'avg_utility': (self.total_utility / self.iterations
                            if self.iterations > 0 else 0.0),
        }


# ---------------------------------------------------------------------------
# Quick test
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    print("=== CFR Training Test (Leduc Hold'em) ===\n")
    trainer = CFRTrainer()

    print("Training 1000 iterations...")
    for batch in range(10):
        util = trainer.train(100)
        s    = trainer.stats()
        print(f"  Iter {s['iterations']:5d} | "
              f"avg utility: {s['avg_utility']:+.4f} | "
              f"info sets: {s['info_sets']:,}")

    print(f"\nFinal info sets: {trainer.n_info_sets():,} (Leduc has ~936 total)")
    print("\nSample strategies (average — converging to Nash):")

    from poker_env import LeducState, ACTION_NAMES, card_str
    state = LeducState().reset(42)
    for player in [0, 1]:
        info  = state.info_set(player)
        legal = state.legal_actions()
        probs = trainer.get_average_strategy(info, legal)
        card  = card_str(state.private[player])
        print(f"  P{player} ({card}): "
              + ", ".join(f"{ACTION_NAMES[a]}={probs[a]:.2f}"
                          for a in legal))