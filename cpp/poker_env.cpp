/*
 * poker_env.cpp — Royal Hold'em game logic.
 */

#include "poker_env.h"
#include <sstream>
#include <numeric>
#include <stdexcept>
#include <cstring>

static const char* RANKS = "TJQKA";
static const char* SUITS = "cdhs";

std::string card_str(int card) {
    std::string s;
    s += RANKS[card % 5];
    s += SUITS[card / 5];
    return s;
}

HandScore evaluate_hand(const std::vector<int>& cards) {
    int rank_counts[5] = {};
    int suit_counts[4] = {};
    std::vector<int> ranks, suits;

    for (int c : cards) {
        int r = c % 5;
        int s = c / 5;
        rank_counts[r]++;
        suit_counts[s]++;
        ranks.push_back(r);
        suits.push_back(s);
    }

    auto pack = [](int cat, int a=-1, int b=-1, int c=-1, int d=-1, int e=-1) -> long long {
        long long v = (long long)cat << 24;
        if (a >= 0) v |= (long long)(a+1) << 20;
        if (b >= 0) v |= (long long)(b+1) << 16;
        if (c >= 0) v |= (long long)(c+1) << 12;
        if (d >= 0) v |= (long long)(d+1) << 8;
        if (e >= 0) v |= (long long)(e+1) << 4;
        return v;
    };

    // 1. Royal Flush
    for (int s = 0; s < 4; s++)
        if (suit_counts[s] >= 5) return {pack(8)};

    // 2. Four of a Kind
    for (int r = 4; r >= 0; r--) {
        if (rank_counts[r] >= 4) {
            int kicker = -1;
            for (int x : ranks) if (x != r && x > kicker) kicker = x;
            return {pack(7, r, kicker)};
        }
    }

    // 3. Full House
    std::vector<int> trips_list, pair_list;
    for (int r = 4; r >= 0; r--) {
        if (rank_counts[r] >= 3) trips_list.push_back(r);
        if (rank_counts[r] >= 2) pair_list.push_back(r);
    }
    if (trips_list.size() >= 2) {
        return {pack(6, trips_list[0], trips_list[1])};
    } else if (!trips_list.empty()) {
        for (int p : pair_list)
            if (p != trips_list[0]) return {pack(6, trips_list[0], p)};
    }

    // 4. Straight
    bool has_all = true;
    for (int r = 0; r < 5; r++) if (rank_counts[r] == 0) { has_all = false; break; }
    if (has_all) return {pack(5)};

    // 5. Three of a Kind
    if (!trips_list.empty()) {
        std::vector<int> kickers;
        for (int x : ranks) if (x != trips_list[0]) kickers.push_back(x);
        std::sort(kickers.rbegin(), kickers.rend());
        int k1 = kickers.size() > 0 ? kickers[0] : -1;
        int k2 = kickers.size() > 1 ? kickers[1] : -1;
        return {pack(4, trips_list[0], k1, k2)};
    }

    // 6. Two Pair
    if (pair_list.size() >= 2) {
        std::vector<int> kickers;
        for (int x : ranks)
            if (x != pair_list[0] && x != pair_list[1]) kickers.push_back(x);
        std::sort(kickers.rbegin(), kickers.rend());
        int k = kickers.empty() ? -1 : kickers[0];
        return {pack(3, pair_list[0], pair_list[1], k)};
    }

    // 7. One Pair
    if (pair_list.size() == 1) {
        std::vector<int> kickers;
        for (int x : ranks) if (x != pair_list[0]) kickers.push_back(x);
        std::sort(kickers.rbegin(), kickers.rend());
        int k1 = kickers.size() > 0 ? kickers[0] : -1;
        int k2 = kickers.size() > 1 ? kickers[1] : -1;
        int k3 = kickers.size() > 2 ? kickers[2] : -1;
        return {pack(2, pair_list[0], k1, k2, k3)};
    }

    // 8. High Card
    std::vector<int> sorted_ranks = ranks;
    std::sort(sorted_ranks.rbegin(), sorted_ranks.rend());
    while (sorted_ranks.size() < 5) sorted_ranks.push_back(-1);
    return {pack(1, sorted_ranks[0], sorted_ranks[1], sorted_ranks[2],
                 sorted_ranks[3], sorted_ranks[4])};
}

// ---------------------------------------------------------------------------
// RoyalState implementation
// ---------------------------------------------------------------------------

RoyalState::RoyalState() { reset(); }

RoyalState& RoyalState::reset(int seed) {
    if (seed < 0) {
        std::random_device rd;
        rng = std::mt19937(rd());
    } else {
        rng = std::mt19937(seed);
    }

    int deck[N_CARDS];
    std::iota(deck, deck + N_CARDS, 0);
    std::shuffle(deck, deck + N_CARDS, rng);

    private_cards[0][0] = deck[0]; private_cards[0][1] = deck[1];
    private_cards[1][0] = deck[2]; private_cards[1][1] = deck[3];
    for (int i = 0; i < 5; i++) community_cards[i] = deck[4 + i];

    round   = 0;
    pot     = SB + BB;
    stacks[0] = STARTING_STACK - SB;
    stacks[1] = STARTING_STACK - BB;
    bets[0]   = SB;
    bets[1]   = BB;
    to_act  = 0;
    raises  = 0;
    done    = false;
    winner  = -2;
    n_history = 0;
    n_acted = 0;
    memset(history, 0, sizeof(history));
    return *this;
}

RoyalState RoyalState::copy() const { return *this; }

std::vector<int> RoyalState::visible_community() const {
    if (round == 0) return {};
    if (round == 1) return {community_cards[0], community_cards[1], community_cards[2]};
    if (round == 2) return {community_cards[0], community_cards[1],
                            community_cards[2], community_cards[3]};
    return {community_cards[0], community_cards[1], community_cards[2],
            community_cards[3], community_cards[4]};
}

std::vector<int> RoyalState::legal_actions() const {
    if (done) return {};
    std::vector<int> actions = {FOLD, CALL};
    if (raises < MAX_RAISES) {
        int call_amt = bets[1 - to_act] - bets[to_act];
        int raise_sizes[3];
        compute_raise_sizes(pot, call_amt, raise_sizes);
        for (int i = 0; i < 3; i++) {
            int needed = call_amt + raise_sizes[i];
            if (stacks[to_act] >= needed)
                actions.push_back(RAISE_S + i);
        }
    }
    return actions;
}

RoyalState& RoyalState::apply_action(int action) {
    int player   = to_act;
    int opponent = 1 - player;

    history[n_history++] = action;
    n_acted++;

    if (action == FOLD) {
        done = true;
        winner = opponent;
        stacks[opponent] += pot;
        pot = 0;
        return *this;
    }

    if (action == CALL) {
        int call_amt = bets[opponent] - bets[player];
        call_amt = std::min(call_amt, stacks[player]);
        stacks[player] -= call_amt;
        bets[player]   += call_amt;
        pot            += call_amt;
        if (_round_over()) { _advance_round(); return *this; }
    } else {
        // RAISE_S, RAISE_M, or RAISE_L — pot-relative sizing
        int raise_idx = action - RAISE_S;
        int call_amt  = bets[opponent] - bets[player];
        int raise_sizes[3];
        compute_raise_sizes(pot, call_amt, raise_sizes);
        int bet   = raise_sizes[raise_idx];
        int total = std::min(call_amt + bet, stacks[player]);
        stacks[player] -= total;
        bets[player]   += total;
        pot            += total;
        raises++;
    }

    to_act = opponent;
    return *this;
}

bool RoyalState::_round_over() const {
    return bets[0] == bets[1] && n_acted >= 2;
}

void RoyalState::_advance_round() {
    round++;
    bets[0] = bets[1] = 0;
    raises  = 0;
    n_acted = 0;

    if (round > 3) {
        done   = true;
        winner = _showdown_winner();
        if (winner == -1) {
            int half = pot / 2;
            stacks[0] += half;
            stacks[1] += pot - half;
        } else {
            stacks[winner] += pot;
        }
        pot = 0;
    } else {
        to_act = 1;
    }
}

int RoyalState::_showdown_winner() const {
    std::vector<int> hand0 = {private_cards[0][0], private_cards[0][1]};
    std::vector<int> hand1 = {private_cards[1][0], private_cards[1][1]};
    for (int c : community_cards) { hand0.push_back(c); hand1.push_back(c); }
    HandScore s0 = evaluate_hand(hand0);
    HandScore s1 = evaluate_hand(hand1);
    if (s0 > s1) return 0;
    if (s1 > s0) return 1;
    return -1;
}

float RoyalState::payoff(int player) const {
    return static_cast<float>(stacks[player] - STARTING_STACK);
}

std::string RoyalState::to_string() const {
    std::ostringstream ss;
    ss << "Round " << round << " | Pot " << pot << " | P" << to_act << " to act\n";
    ss << "  P0: " << card_str(private_cards[0][0]) << " " << card_str(private_cards[0][1])
       << "  stack=" << stacks[0] << "  bet=" << bets[0] << "\n";
    ss << "  P1: " << card_str(private_cards[1][0]) << " " << card_str(private_cards[1][1])
       << "  stack=" << stacks[1] << "  bet=" << bets[1] << "\n";
    ss << "  History: ";
    for (int i = 0; i < n_history; i++) ss << history[i];
    return ss.str();
}
