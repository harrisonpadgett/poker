"""
poker_env.py — Royal Hold'em environment.

Action space (N_ACTIONS = 5):
  FOLD=0, CALL=1, RAISE_S=2 (small), RAISE_M=3 (medium), RAISE_L=4 (large)

Raise chip amounts per round:
  Preflop / Flop : 2 / 4 / 6
  Turn   / River : 4 / 6 / 8
"""

import numpy as np

# ---------------------------------------------------------------------------
# Card constants
# ---------------------------------------------------------------------------
RANKS   = ['T', 'J', 'Q', 'K', 'A']
SUITS   = ['c', 'd', 'h', 's']
N_CARDS = 20

def card_rank(card): return card % 5
def card_suit(card): return card // 5
def card_str(card):  return RANKS[card_rank(card)] + SUITS[card_suit(card)]

# ---------------------------------------------------------------------------
# Actions
# ---------------------------------------------------------------------------
FOLD   = 0
CALL   = 1
RAISE_S = 2   # small raise
RAISE_M = 3   # medium raise
RAISE_L = 4   # large raise

N_ACTIONS    = 5
ACTION_NAMES = ['Fold', 'Call', 'Raise-S', 'Raise-M', 'Raise-L']

# ---------------------------------------------------------------------------
# Game constants
# ---------------------------------------------------------------------------
ANTE       = 1
MAX_RAISES = 2

# Raise chip amounts: RAISE_AMOUNTS[round][size_index]
RAISE_AMOUNTS = [
    [2, 4, 6],   # Preflop: small=2, medium=4, large=6
    [2, 4, 6],   # Flop
    [4, 6, 8],   # Turn:   small=4, medium=6, large=8
    [4, 6, 8],   # River
]

# ---------------------------------------------------------------------------
# Hand Evaluation
# ---------------------------------------------------------------------------
def evaluate_hand(cards):
    """Evaluate best 5-card hand from any number of cards (Royal Hold'em)."""
    ranks = [c % 5 for c in cards]
    suits = [c // 5 for c in cards]

    rank_counts = [0] * 5
    suit_counts = [0] * 4
    for r, s in zip(ranks, suits):
        rank_counts[r] += 1
        suit_counts[s] += 1

    # 1. Royal Flush
    for s in range(4):
        if suit_counts[s] >= 5:
            return (8,)

    # 2. Four of a Kind
    for r in range(4, -1, -1):
        if rank_counts[r] >= 4:
            kicker = max([x for x in ranks if x != r], default=-1)
            return (7, r, kicker)

    # 3. Full House
    trips = [r for r in range(4, -1, -1) if rank_counts[r] >= 3]
    pairs = [r for r in range(4, -1, -1) if rank_counts[r] >= 2]
    if len(trips) >= 2:
        return (6, trips[0], trips[1])
    elif len(trips) == 1:
        valid_pairs = [p for p in pairs if p != trips[0]]
        if valid_pairs:
            return (6, trips[0], valid_pairs[0])

    # 4. Straight (all 5 ranks in the Royal deck)
    if all(c > 0 for c in rank_counts):
        return (5,)

    # 5. Three of a Kind
    if len(trips) == 1:
        kickers = sorted([x for x in ranks if x != trips[0]], reverse=True)[:2]
        return (4, trips[0], kickers[0], kickers[1])

    # 6. Two Pair
    if len(pairs) >= 2:
        kickers = sorted([x for x in ranks if x not in (pairs[0], pairs[1])], reverse=True)[:1]
        return (3, pairs[0], pairs[1], kickers[0] if kickers else -1)

    # 7. One Pair
    if len(pairs) == 1:
        kickers = sorted([x for x in ranks if x != pairs[0]], reverse=True)[:3]
        while len(kickers) < 3: kickers.append(-1)
        return (2, pairs[0], kickers[0], kickers[1], kickers[2])

    # 8. High Card
    kickers = sorted(ranks, reverse=True)[:5]
    while len(kickers) < 5: kickers.append(-1)
    return (1, kickers[0], kickers[1], kickers[2], kickers[3], kickers[4])


# ---------------------------------------------------------------------------
# Royal Hold'em state
# ---------------------------------------------------------------------------
class RoyalState:
    def __init__(self):
        self.reset()

    def reset(self, seed=None):
        rng = np.random.RandomState(seed)
        deck = list(range(N_CARDS))
        rng.shuffle(deck)

        self.private   = [deck[0:2], deck[2:4]]
        self.community = deck[4:9]   # all 5 community cards (revealed progressively)

        self.round   = 0
        self.pot     = ANTE * 2
        self.stacks  = [100 - ANTE, 100 - ANTE]
        self.bets    = [ANTE, ANTE]
        self.to_act  = 0
        self.raises  = 0
        self.done    = False
        self.winner  = None
        self.history = []
        self.n_acted = 0
        return self

    @property
    def visible_community(self):
        if self.round == 0: return []
        if self.round == 1: return self.community[0:3]
        if self.round == 2: return self.community[0:4]
        return self.community[0:5]

    def legal_actions(self):
        if self.done:
            return []
        actions = [FOLD, CALL]
        if self.raises < MAX_RAISES:
            call_amt = self.bets[1 - self.to_act] - self.bets[self.to_act]
            for i in range(3):
                needed = call_amt + RAISE_AMOUNTS[self.round][i]
                if self.stacks[self.to_act] >= needed:
                    actions.append(RAISE_S + i)
        return actions

    def apply_action(self, action):
        assert not self.done
        assert action in self.legal_actions(), f"Illegal action {action}, legal={self.legal_actions()}"

        player   = self.to_act
        opponent = 1 - player

        self.history.append(action)
        self.n_acted += 1

        if action == FOLD:
            self.done   = True
            self.winner = opponent
            self.stacks[opponent] += self.pot
            self.pot = 0
            return self

        elif action == CALL:
            call_amt = self.bets[opponent] - self.bets[player]
            call_amt = min(call_amt, self.stacks[player])
            self.stacks[player] -= call_amt
            self.bets[player]   += call_amt
            self.pot            += call_amt
            if self._round_over():
                self._advance_round()
                return self

        else:
            # RAISE_S / RAISE_M / RAISE_L
            raise_idx = action - RAISE_S
            bet       = RAISE_AMOUNTS[self.round][raise_idx]
            call_amt  = self.bets[opponent] - self.bets[player]
            total     = min(call_amt + bet, self.stacks[player])
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

        if self.round > 3:
            self.done   = True
            self.winner = self._showdown_winner()
            if self.winner == -1:
                half = self.pot // 2
                self.stacks[0] += half
                self.stacks[1] += self.pot - half
            else:
                self.stacks[self.winner] += self.pot
            self.pot = 0
        else:
            self.to_act = 1

    def _showdown_winner(self):
        v0 = evaluate_hand(self.private[0] + self.community)
        v1 = evaluate_hand(self.private[1] + self.community)
        if v0 > v1: return 0
        if v1 > v0: return 1
        return -1

    def payoff(self, player):
        assert self.done
        return self.stacks[player] - 100

    def copy(self):
        s = RoyalState.__new__(RoyalState)
        s.private   = [list(self.private[0]), list(self.private[1])]
        s.community = list(self.community)
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
        comm = ' '.join(card_str(c) for c in self.visible_community) if self.visible_community else 'None'
        return (f"Round {self.round} | Pot {self.pot} | P{self.to_act} to act\n"
                f"  P0: {' '.join(card_str(c) for c in self.private[0])}  "
                f"stack={self.stacks[0]}  bet={self.bets[0]}\n"
                f"  P1: {' '.join(card_str(c) for c in self.private[1])}  "
                f"stack={self.stacks[1]}  bet={self.bets[1]}\n"
                f"  Community: {comm}\n"
                f"  History: {''.join(str(a) for a in self.history)}")


if __name__ == '__main__':
    print("=== Royal Hold'em Environment (5-action) ===")
    s = RoyalState().reset(42)
    print(f"\nDealt hand:\n{s}")
    s.apply_action(RAISE_M)   # medium preflop raise
    s.apply_action(CALL)
    print(f"\nFlop:\n{s}")
    s.apply_action(CALL)
    s.apply_action(RAISE_S)
    s.apply_action(CALL)
    print(f"\nTurn:\n{s}")
    s.apply_action(RAISE_L)
    s.apply_action(CALL)
    print(f"\nRiver:\n{s}")
    s.apply_action(RAISE_S)
    s.apply_action(CALL)
    print(f"\nShowdown:\n{s}")
    print(f"Result: winner={s.winner}")
    print(f"P0 Payoff: {s.payoff(0):+d}  P1 Payoff: {s.payoff(1):+d}")
