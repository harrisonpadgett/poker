"""
poker_env.py — Leduc Hold'em poker environment.

Leduc Hold'em is a simplified poker game designed for CFR research.
It has a small enough state space for CFR to find near-Nash strategies
in thousands of iterations while capturing the essential strategic
concepts of real poker: bluffing, semi-bluffing, value betting,
exploiting position.

Rules:
  - Deck: 6 cards — J, Q, K in two suits (clubs, spades) = 6 cards total
  - 2 players, heads-up
  - 2 betting rounds: preflop (private card) + flop (one community card)
  - Fixed bet sizes: round 1 = 2 chips, round 2 = 4 chips
  - Max 2 raises per round
  - Ante: each player puts in 1 chip before the hand

Hand ranking:
  - Pair (private card matches community card) > High card
  - Within same category, higher rank wins
  - J < Q < K

Why Leduc is the right starting point:
  - State space: ~936 information sets — manageable for exact CFR
  - Full Hold'em: ~10^18 information sets — requires abstraction
  - Leduc captures all strategic concepts: position, bluffing, semi-bluffing
  - Nash equilibrium is reachable in ~10,000 iterations on a laptop
  - Well-studied: published Nash equilibrium to compare against
"""

import numpy as np
from itertools import product

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------
# Cards: 0=Jc, 1=Qc, 2=Kc, 3=Js, 4=Qs, 5=Ks
JACK  = 0   # rank
QUEEN = 1
KING  = 2
RANKS = ['J', 'Q', 'K']
SUITS = ['c', 's']
N_CARDS = 6   # 2 suits × 3 ranks

def card_rank(card):
    return card % 3

def card_suit(card):
    return card // 3

def card_str(card):
    return RANKS[card_rank(card)] + SUITS[card_suit(card)]

def rank_str(rank):
    return RANKS[rank]

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
FOLD  = 0
CALL  = 1   # also = check when no bet facing
RAISE = 2
ACTION_NAMES = ['Fold', 'Call', 'Raise']
N_ACTIONS = 3

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
ANTE       = 1
BET_SIZES  = [2, 4]    # round 1, round 2
MAX_RAISES = 2

# ---------------------------------------------------------------------------
# Leduc Hold'em state
# ---------------------------------------------------------------------------
class LeducState:
    """
    Complete game state for Leduc Hold'em.

    Information set for player p:
      private_card | community_card (or 'none') | betting_history

    Example: "Qc|Kc|crrc" means:
      - Private card: Queen of clubs
      - Community: King of clubs
      - History: check, raise, raise, call
    """

    def __init__(self):
        self.reset()

    def reset(self, seed=None):
        rng = np.random.RandomState(seed)
        deck = list(range(N_CARDS))
        rng.shuffle(deck)

        self.private    = [deck[0], deck[1]]   # private cards
        self.community  = deck[2]              # community (revealed round 2)

        self.round      = 0       # 0 = preflop, 1 = flop
        self.pot        = ANTE * 2
        self.stacks     = [10 - ANTE, 10 - ANTE]
        self.bets       = [ANTE, ANTE]         # preflop both put in ante
        self.to_act     = 0
        self.raises     = 0
        self.done       = False
        self.winner     = None
        self.history    = []      # list of action chars for info set string
        self.n_acted    = 0       # actions taken this round
        return self

    @property
    def visible_community(self):
        return self.community if self.round == 1 else None

    def info_set(self, player):
        """
        What player p can observe — used as CFR strategy key.
        Format: private_card | community_or_none | action_history
        """
        priv   = card_str(self.private[player])
        comm   = card_str(self.community) if self.round == 1 else 'none'
        hist   = ''.join('f' if a==FOLD else 'c' if a==CALL else 'r'
                         for a in self.history)
        return f'P{player}|{priv}|{comm}|{hist}'

    def legal_actions(self):
        if self.done or self.round > 1:
            return []
        actions = [FOLD, CALL]
        if (self.raises < MAX_RAISES and
                self.stacks[self.to_act] >= BET_SIZES[self.round]):
            actions.append(RAISE)
        return actions

    def apply_action(self, action):
        assert not self.done
        assert action in self.legal_actions()

        player   = self.to_act
        opponent = 1 - player
        bet      = BET_SIZES[self.round]

        self.history.append(action)
        self.n_acted += 1

        if action == FOLD:
            self.done   = True
            self.winner = opponent
            self.stacks[opponent] += self.pot
            self.pot = 0
            return self

        elif action == CALL:
            # Match opponent's bet
            call_amt = self.bets[opponent] - self.bets[player]
            call_amt = min(call_amt, self.stacks[player])
            self.stacks[player] -= call_amt
            self.bets[player]   += call_amt
            self.pot            += call_amt

            if self._round_over():
                self._advance_round()
                return self   # _advance_round sets to_act — don't overwrite

        elif action == RAISE:
            call_amt  = self.bets[opponent] - self.bets[player]
            total     = call_amt + bet
            total     = min(total, self.stacks[player])
            self.stacks[player] -= total
            self.bets[player]   += total
            self.pot            += total
            self.raises         += 1

        self.to_act = opponent
        return self

    def _round_over(self):
        return self.bets[0] == self.bets[1] and self.n_acted >= 2

    def _advance_round(self):
        self.round  += 1
        self.bets    = [0, 0]
        self.raises  = 0
        self.n_acted = 0

        if self.round > 1:
            # Showdown
            self.done   = True
            self.winner = self._showdown_winner()
            if self.winner == -1:
                # Tie — split pot
                half = self.pot // 2
                self.stacks[0] += half
                self.stacks[1] += self.pot - half
            else:
                self.stacks[self.winner] += self.pot
            self.pot = 0   # pot distributed
        else:
            self.to_act = 1   # P1 acts first post-flop (position)

    def _showdown_winner(self):
        """
        Pair > high card. Higher rank wins within category.
        Returns 0, 1, or -1 (tie).
        """
        def hand_val(player):
            priv = card_rank(self.private[player])
            comm = card_rank(self.community)
            if priv == comm:
                return (1, priv)   # pair
            return (0, priv)       # high card

        v0, v1 = hand_val(0), hand_val(1)
        if v0 > v1: return 0
        if v1 > v0: return 1
        return -1

    def payoff(self, player):
        assert self.done
        # Net chips: current stack minus starting stack (10 - ante already paid)
        # Total chips in game = 20 always. Winner gets pot, loser loses investment.
        # Simpler: track as (chips_won - chips_invested)
        # chips_invested = starting_stack(9) - current_stack + pot_share
        # But easiest: just use zero-sum property
        # P0_payoff = -P1_payoff, and we know total pot distributed
        # Use: payoff = final_stack - starting_stack_before_ante
        # starting stack before ante = 10
        return self.stacks[player] - 10

    def copy(self):
        """Fast copy — manually copy only mutable fields."""
        s = LeducState.__new__(LeducState)
        s.rows      = None  # not used in Leduc
        s.cols      = None
        s.private   = list(self.private)
        s.community = self.community
        s.round     = self.round
        s.pot       = self.pot
        s.stacks    = list(self.stacks)
        s.bets      = list(self.bets)
        s.to_act    = self.to_act
        s.raises    = self.raises
        s.done      = self.done
        s.winner    = self.winner
        s.history   = list(self.history)
        s.n_acted   = self.n_acted
        return s

    def __str__(self):
        comm = card_str(self.community) if self.round==1 else '?'
        return (f"Round {self.round} | Pot {self.pot} | "
                f"P{self.to_act} to act\n"
                f"  P0: {card_str(self.private[0])}  "
                f"stack={self.stacks[0]}  bet={self.bets[0]}\n"
                f"  P1: {card_str(self.private[1])}  "
                f"stack={self.stacks[1]}  bet={self.bets[1]}\n"
                f"  Community: {comm}  "
                f"History: {''.join(str(a) for a in self.history)}")


if __name__ == '__main__':
    print("=== Leduc Hold'em Environment ===\n")
    print("Cards:", [card_str(i) for i in range(N_CARDS)])

    s = LeducState().reset(42)
    print(f"\nDealt hand:\n{s}")
    print(f"\nP0 info set: {s.info_set(0)}")
    print(f"P1 info set: {s.info_set(1)}")

    # Play a hand
    print("\nPlaying: P0 raises, P1 calls → flop → P1 bets, P0 calls")
    s.apply_action(RAISE)
    s.apply_action(CALL)
    print(f"After preflop:\n{s}")

    s.apply_action(RAISE)
    s.apply_action(CALL)
    print(f"Result: winner={s.winner}")
    print(f"P0: {s.payoff(0):+d}  P1: {s.payoff(1):+d}")
    assert s.payoff(0) + s.payoff(1) == 0, "Zero-sum check failed"
    print("\n✓ Environment OK")