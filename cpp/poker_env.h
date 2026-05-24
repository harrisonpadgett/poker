/*
 * poker_env.h — Royal Hold'em game state for C++.
 *
 * BEGINNER NOTE: A "header file" (.h) is like a contract or table of contents.
 * It declares *what* exists (class names, function signatures, constants)
 * without defining *how* they work. The actual logic lives in poker_env.cpp.
 *
 * This pattern lets multiple .cpp files share the same declarations
 * without recompiling everything from scratch each time.
 */

#pragma once   // Tells the compiler: "only include this file once per build"

#include <array>
#include <vector>
#include <random>
#include <algorithm>
#include <cassert>
#include <string>
#include <tuple>

// ---------------------------------------------------------------------------
// Constants — same values as Python poker_env.py
// ---------------------------------------------------------------------------
static constexpr int N_CARDS    = 20;
static constexpr int N_ACTIONS  = 3;
static constexpr int FOLD       = 0;
static constexpr int CALL       = 1;
static constexpr int RAISE_A    = 2;   // named RAISE_A to avoid conflict with std headers
static constexpr int ANTE       = 1;
static constexpr int MAX_RAISES = 4;
static constexpr int STARTING_STACK = 100;

// Fixed-limit bet sizes per round: Preflop=2, Flop=2, Turn=4, River=4
static constexpr int BET_SIZES[4] = {2, 2, 4, 4};

// ---------------------------------------------------------------------------
// Hand evaluation result — a simple comparable struct
// (mimics Python's tuple comparison)
// ---------------------------------------------------------------------------
struct HandScore {
    // Up to 6 values: category + up to 5 tiebreakers
    // We pack them into a single int64 for fast comparison
    long long value;

    bool operator>(const HandScore& o) const { return value > o.value; }
    bool operator<(const HandScore& o) const { return value < o.value; }
    bool operator==(const HandScore& o) const { return value == o.value; }
};

// ---------------------------------------------------------------------------
// Forward declarations
// ---------------------------------------------------------------------------
HandScore evaluate_hand(const std::vector<int>& cards);
std::string card_str(int card);

// ---------------------------------------------------------------------------
// RoyalState — the core game state struct
//
// BEGINNER NOTE: In C++, a "struct" is like a Python dataclass. All members
// are stored directly inside the struct — no heap allocation, no GC.
// This means copying a RoyalState is just a fast memcpy of ~200 bytes.
// ---------------------------------------------------------------------------
struct RoyalState {
    // Cards
    int private_cards[2][2];   // hole cards: private_cards[player][0..1]
    int community_cards[5];    // all 5 board cards (only some visible per round)

    // Game state
    int  round;        // 0=Preflop, 1=Flop, 2=Turn, 3=River
    int  pot;
    int  stacks[2];
    int  bets[2];
    int  to_act;       // which player acts next (0 or 1)
    int  raises;       // number of raises in this round
    bool done;
    int  winner;       // -1=tie, 0=P0, 1=P1, -2=undecided

    // Action history (max 32 actions per game is safe upper bound)
    int  history[32];
    int  n_history;
    int  n_acted;      // actions taken in current round

    // Random number generator state (for reproducible deals)
    std::mt19937 rng;

    // -----------------------------------------------------------------
    // Methods — declared here, defined in poker_env.cpp
    // -----------------------------------------------------------------
    RoyalState();                            // constructor: calls reset()
    RoyalState& reset(int seed = -1);        // deal fresh cards
    RoyalState  copy() const;                // return a full copy

    std::vector<int> legal_actions() const;
    RoyalState& apply_action(int action);

    float payoff(int player) const;

    // Returns the community cards visible at the current round
    std::vector<int> visible_community() const;

    std::string to_string() const;

private:
    bool         _round_over() const;
    void         _advance_round();
    int          _showdown_winner() const;
};
