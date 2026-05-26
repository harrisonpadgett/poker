/*
 * poker_env.h — Royal Hold'em game state for C++.
 *
 * Action space (N_ACTIONS = 5):
 *   FOLD=0, CALL=1, RAISE_S=2 (small), RAISE_M=3 (medium), RAISE_L=4 (large)
 *
 * Raise amounts per round (chips):
 *   Preflop / Flop : 2 / 4 / 6
 *   Turn   / River : 4 / 6 / 8
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
static constexpr int RAISE_S    = 2;   // small raise
static constexpr int RAISE_M    = 3;   // medium raise
static constexpr int RAISE_L    = 4;   // large raise
static constexpr int ANTE       = 1;
static constexpr int MAX_RAISES = 2;
static constexpr int STARTING_STACK = 100;

// Raise chip amounts per round: [preflop, flop, turn, river] × [small, medium, large]
static constexpr int RAISE_AMOUNTS[4][3] = {
    {2, 4, 6},   // Preflop
    {2, 4, 6},   // Flop
    {4, 6, 8},   // Turn
    {4, 6, 8},   // River
};

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
