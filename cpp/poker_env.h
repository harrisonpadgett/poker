/*
 * poker_env.h — Royal Hold'em game state for C++.
 *
 * Action space (N_ACTIONS = 5):
 *   FOLD=0, CALL=1, RAISE_S=2 (small), RAISE_M=3 (medium), RAISE_L=4 (large)
 *
 * Raise amounts are pot-relative (computed dynamically):
 *   RAISE_S : 33% of pot-after-call  (min BB)
 *   RAISE_M : 66% of pot-after-call  (min BB)
 *   RAISE_L : 100% of pot-after-call (min BB)
 *
 * Blind structure (heads-up):
 *   Player 0 = Small Blind (SB) — posts 1 chip, acts first preflop
 *   Player 1 = Big Blind  (BB) — posts 2 chips, acts first postflop
 */

#pragma once

#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <string>
#include <tuple>

// ---------------------------------------------------------------------------
// Constants
// ---------------------------------------------------------------------------
static constexpr int N_CARDS    = 20;
static constexpr int N_ACTIONS  = 5;
static constexpr int FOLD       = 0;
static constexpr int CALL       = 1;
static constexpr int RAISE_S    = 2;   // small raise  (~33% pot)
static constexpr int RAISE_M    = 3;   // medium raise (~66% pot)
static constexpr int RAISE_L    = 4;   // large raise  (~100% pot)
static constexpr int SB         = 1;   // small blind
static constexpr int BB         = 2;   // big blind
static constexpr int MAX_RAISES = 2;
static constexpr int STARTING_STACK = 100;

// Compute pot-relative raise sizes given current pot and call amount.
// out[0] = 33% pot, out[1] = 66% pot, out[2] = 100% pot, all >= BB.
inline void compute_raise_sizes(int pot, int call_amt, int out[3]) {
    int pac = pot + call_amt;   // pot after call
    out[0] = std::max(BB, pac / 3);
    out[1] = std::max(BB, 2 * pac / 3);
    out[2] = std::max(BB, pac);
}

// ---------------------------------------------------------------------------
// Hand evaluation result
// ---------------------------------------------------------------------------
struct HandScore {
    long long value;
    bool operator>(const HandScore& o) const { return value > o.value; }
    bool operator<(const HandScore& o) const { return value < o.value; }
    bool operator==(const HandScore& o) const { return value == o.value; }
};

HandScore evaluate_hand(const std::vector<int>& cards);
std::string card_str(int card);

// ---------------------------------------------------------------------------
// RoyalState
// ---------------------------------------------------------------------------
struct RoyalState {
    int  private_cards[2][2];
    int  community_cards[5];

    int  round;
    int  pot;
    int  stacks[2];
    int  bets[2];
    int  to_act;
    int  raises;
    bool done;
    int  winner;

    int  history[32];
    int  n_history;
    int  n_acted;

    std::mt19937 rng;

    RoyalState();
    RoyalState& reset(int seed = -1);
    RoyalState  copy() const;

    std::vector<int> legal_actions() const;
    RoyalState& apply_action(int action);

    float payoff(int player) const;
    std::vector<int> visible_community() const;
    std::string to_string() const;

private:
    bool _round_over() const;
    void _advance_round();
    int  _showdown_winner() const;
};
