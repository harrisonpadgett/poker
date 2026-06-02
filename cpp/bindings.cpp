/*
 * bindings.cpp — pybind11 bindings for Royal Hold'em Deep CFR engine.
 *
 * Architecture:
 *   - 5 actions: Fold/Call/Raise-S/Raise-M/Raise-L
 *   - 122-dim state: 1 player + 20 hole + 20 community + 80 history(16×5) + 1 is_suited
 *   - 8 advantage networks: g_net[player][round]
 *   - 12 buffers: adv[0..7] = player*4+round,  strat[8..11] = 8+round
 *   - update_model_cache(player, round, fc1_w, fc1_b, fc2_w, fc2_b, out_w, out_b)
 *   - run_traversals(k, iter, adv_buf_list[8], strat_buf_list[4])
 */

#include "poker_env.h"
#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <dlfcn.h>
#include <cstring>
#include <thread>
#include <random>
#include <vector>
#include <atomic>

namespace py = pybind11;

// ---------------------------------------------------------------------------
// BLAS acceleration
// ---------------------------------------------------------------------------
typedef void (*cblas_sgemv_t)(int, int, int, int, float, const float*, int,
                              const float*, int, float, float*, int);
static cblas_sgemv_t cblas_sgemv_ptr = nullptr;

void init_blas() {
    void* handle = dlopen("@rpath/libBLAS.dylib", RTLD_NOLOAD);
    if (!handle) handle = dlopen("/System/Library/Frameworks/Accelerate.framework/Accelerate", RTLD_NOLOAD);
    if (!handle) return;
    cblas_sgemv_ptr = (cblas_sgemv_t)dlsym(handle, "cblas_sgemv");
}

// ---------------------------------------------------------------------------
// LinearLayer
// ---------------------------------------------------------------------------
struct LinearLayer {
    std::vector<float> weight, bias;
    int in_features, out_features;

    void forward(const float* input, float* output, bool relu) const {
        if (cblas_sgemv_ptr) {
            cblas_sgemv_ptr(101, 111, out_features, in_features,
                           1.0f, weight.data(), in_features,
                           input, 1,
                           0.0f, output, 1);
        } else {
            for (int i = 0; i < out_features; i++) {
                float sum = 0.0f;
                for (int j = 0; j < in_features; j++)
                    sum += weight[i * in_features + j] * input[j];
                output[i] = sum;
            }
        }
        for (int i = 0; i < out_features; i++)
            output[i] += bias[i];
        if (relu)
            for (int i = 0; i < out_features; i++)
                output[i] = output[i] > 0.0f ? output[i] : 0.0f;
    }
};

// ---------------------------------------------------------------------------
// CppMLP — 122 → 256 → 256 → 5  (per-player, per-round)
// ---------------------------------------------------------------------------
struct CppMLP {
    LinearLayer fc1, fc2, out_layer;
    bool warm = false;

    void forward(const float input[122], float output[5]) const {
        float h1[256], h2[256];
        fc1.forward(input, h1, true);
        fc2.forward(h1,    h2, true);
        out_layer.forward(h2, output, false);
    }
};

// 8 independent specialists: g_net[player][round]
static CppMLP g_net[2][4];

static void load_layer(LinearLayer& layer, py::array_t<float> w_arr,
                       py::array_t<float> b_arr, int in_f, int out_f) {
    layer.in_features  = in_f;
    layer.out_features = out_f;
    auto wbuf = w_arr.request(), bbuf = b_arr.request();
    layer.weight.assign(static_cast<float*>(wbuf.ptr),
                        static_cast<float*>(wbuf.ptr) + in_f * out_f);
    layer.bias.assign(  static_cast<float*>(bbuf.ptr),
                        static_cast<float*>(bbuf.ptr) + out_f);
}

// Sync PyTorch weights for one player/round combo into the C++ cache.
void update_model_cache(
    int player, int round,
    py::array_t<float> fc1_w, py::array_t<float> fc1_b,
    py::array_t<float> fc2_w, py::array_t<float> fc2_b,
    py::array_t<float> out_w, py::array_t<float> out_b
) {
    CppMLP& net = g_net[player][round];
    load_layer(net.fc1,       fc1_w, fc1_b, 122, 256);
    load_layer(net.fc2,       fc2_w, fc2_b, 256, 256);
    load_layer(net.out_layer, out_w, out_b,  256,   5);
    net.warm = true;
}

// ---------------------------------------------------------------------------
// Reservoir buffer
// ---------------------------------------------------------------------------
struct CppReservoirBuffer {
    float* states_ptr;
    float* targets_ptr;
    float* weights_ptr;
    int    capacity;
    int    state_dim;
    int    target_dim;
    int    n_inserted;

    void push(const float* state, const float* target, float weight,
              std::mt19937& rng) {
        int idx;
        if (n_inserted < capacity) {
            idx = n_inserted;
        } else {
            int effective_n = std::min(n_inserted, 2 * capacity);
            std::uniform_int_distribution<int> dist(0, effective_n);
            idx = dist(rng);
            if (idx >= capacity) { n_inserted++; return; }
        }
        memcpy(states_ptr  + (long long)idx * state_dim,  state,
               state_dim  * sizeof(float));
        memcpy(targets_ptr + (long long)idx * target_dim, target,
               target_dim * sizeof(float));
        weights_ptr[idx] = weight;
        n_inserted++;
    }
};

// ---------------------------------------------------------------------------
// Per-thread sample
// ---------------------------------------------------------------------------
struct LocalSample {
    float state[122];   // 122-dim encoding
    float target[5];    // 5-action target
    float weight;
    int   buf_id;       // 0-7: advantage (player*4+round),  8-11: strategy (8+round)
};

// ---------------------------------------------------------------------------
// encode_state — 122-dim layout:
//   [0]       player index
//   [1-20]    hole cards one-hot (20-card deck)
//   [21-40]   visible community cards one-hot
//   [41-120]  action history: 16 slots × N_ACTIONS=5 one-hot bits
//   [121]     is_suited: 1 if both hole cards share the same suit
// ---------------------------------------------------------------------------
void encode_state(const RoyalState& state, int player, float out[122]) {
    memset(out, 0, 122 * sizeof(float));
    out[0] = static_cast<float>(player);

    for (int i = 0; i < 2; i++)
        out[1 + state.private_cards[player][i]] = 1.0f;

    auto vc = state.visible_community();
    for (int c : vc) out[21 + c] = 1.0f;

    for (int i = 0; i < state.n_history && i < 16; i++)
        out[41 + i * N_ACTIONS + state.history[i]] = 1.0f;

    out[121] = (state.private_cards[player][0] / 5 ==
                state.private_cards[player][1] / 5) ? 1.0f : 0.0f;
}

// ---------------------------------------------------------------------------
// traverse — external sampling CFR with Regret-Based Pruning (RBP)
//
// RBP skips recursive exploration of actions that are confidently bad:
//   strategy[a] == 0  (regret matching already ignores it) AND
//   advantages[a] < PRUNE_THRESHOLD  (network is confident it's suboptimal) AND
//   t >= PRUNE_START_ITER  (only after enough training to trust the estimates)
//
// For pruned actions, action_values[a] = advantages[a] is used as a proxy.
// Since strategy[a] == 0, this doesn't affect the EV calculation at all.
// The regret target advantages[a] - ev tells the network "this action is still
// negative relative to the expected value," preserving the correct gradient direction.
//
// Expected speedup: 40-60% at iteration 8k+ (where Fold/Call are already
// confidently negative for strong hands on most boards).
// ---------------------------------------------------------------------------
static constexpr float PRUNE_THRESHOLD  = -0.03f;  // normalized (= -3 chips display)
static constexpr int   PRUNE_START_ITER =  2000;   // wait until networks are mature

// ---------------------------------------------------------------------------
// Discounted CFR (DCFR) — Brown & Sandholm 2019
//
// Instead of Linear CFR's flat weight = t, DCFR weights samples by:
//   advantage samples : t^DCFR_ALPHA   (α = 1.5  → recent regrets dominate)
//   strategy  samples : t^DCFR_GAMMA   (γ = 2.0  → recent strategies dominate even more)
//
// The higher exponent on strategy means the final mixed strategy is almost
// entirely determined by late iterations, accelerating convergence to Nash.
// ---------------------------------------------------------------------------
static constexpr float DCFR_ALPHA = 1.5f;
static constexpr float DCFR_GAMMA = 2.0f;

// ---------------------------------------------------------------------------
// Thread count — settable from Python for the effort toggle.
// 0 means "use hardware_concurrency()" (full speed).
// ---------------------------------------------------------------------------
static int g_num_threads = 0;
void set_num_threads(int n) { g_num_threads = n; }

float traverse(RoyalState state, int traverser, int t,
               std::vector<LocalSample>& samples, std::mt19937& rng) {
    if (state.done) return state.payoff(traverser);

    int player = state.to_act;
    auto legal_actions = state.legal_actions();
    int n_legal = static_cast<int>(legal_actions.size());

    float encoded[122];
    encode_state(state, player, encoded);

    float advantages[5] = {};
    const CppMLP& net = g_net[player][state.round];
    net.forward(encoded, advantages);

    float strategy[5] = {};
    float sum_pos = 0.0f;
    for (int a : legal_actions) sum_pos += std::max(0.0f, advantages[a]);
    if (sum_pos > 0.0f)
        for (int a : legal_actions) strategy[a] = std::max(0.0f, advantages[a]) / sum_pos;
    else
        for (int a : legal_actions) strategy[a] = 1.0f / n_legal;

    if (player == traverser) {
        float action_values[5] = {};
        const bool do_prune = (t >= PRUNE_START_ITER);
        for (int a : legal_actions) {
            // Prune: action is not in the current strategy AND confidently suboptimal.
            if (do_prune && strategy[a] == 0.0f && advantages[a] < PRUNE_THRESHOLD) {
                // Use the network's own estimate as a proxy (no recursion).
                action_values[a] = advantages[a];
            } else {
                RoyalState next = state;
                next.apply_action(a);
                action_values[a] = traverse(next, traverser, t, samples, rng);
            }
        }
        // EV is unbiased: pruned actions have strategy[a]==0 so contribute 0.
        float ev = 0.0f;
        for (int a : legal_actions) ev += strategy[a] * action_values[a];
        
        float sampled_adv[5] = {};
        for (int a : legal_actions) {
            if (do_prune && strategy[a] == 0.0f && advantages[a] < PRUNE_THRESHOLD) {
                // For pruned actions, use the raw advantage estimate as the target
                sampled_adv[a] = advantages[a];
            } else {
                // For explored actions, calculate the true regret
                sampled_adv[a] = action_values[a] - ev;
            }
        }

        LocalSample s;
        memcpy(s.state,  encoded,     122 * sizeof(float));
        memcpy(s.target, sampled_adv,   5 * sizeof(float));
        // DCFR: advantage weight = t^α  (heavier on recent iterations)
        s.weight = std::pow(static_cast<float>(std::max(t, 1)), DCFR_ALPHA);
        s.buf_id = traverser * 4 + state.round;
        samples.push_back(s);
        return ev;
    } else {
        LocalSample s;
        memcpy(s.state,  encoded,  122 * sizeof(float));
        memcpy(s.target, strategy,   5 * sizeof(float));
        // DCFR: strategy weight = t^γ  (even stronger recency bias)
        s.weight = std::pow(static_cast<float>(std::max(t, 1)), DCFR_GAMMA);
        s.buf_id = 8 + state.round;
        samples.push_back(s);

        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng), cum = 0.0f;
        int chosen = legal_actions[0];
        for (int a : legal_actions) {
            cum += strategy[a];
            if (r <= cum) { chosen = a; break; }
        }
        RoyalState next = state;
        next.apply_action(chosen);
        return traverse(next, traverser, t, samples, rng);
    }
}


// ---------------------------------------------------------------------------
// run_traversals
//
// adv_buf_list  — py::list of 8 tuples (states, targets, weights, cap, n_ins)
//                 indexed by player*4+round
// strat_buf_list — py::list of 4 tuples (states, targets, weights, cap, n_ins)
//                 indexed by round (0-3)
// ---------------------------------------------------------------------------
py::dict run_traversals(
    int k_traversals, int iteration,
    py::list adv_buf_list,
    py::list strat_buf_list
) {
    for (int p = 0; p < 2; p++)
        for (int r = 0; r < 4; r++)
            if (!g_net[p][r].warm)
                throw std::runtime_error(
                    "Model cache not warm — call update_model_cache() for all 8 combos.");

    // Unpack 8 advantage buffers
    std::vector<py::array_t<float>> adv_s(8), adv_t(8), adv_w(8);
    std::vector<CppReservoirBuffer> adv_bufs(8);
    for (int i = 0; i < 8; i++) {
        auto tup   = adv_buf_list[i].cast<py::tuple>();
        adv_s[i]   = tup[0].cast<py::array_t<float>>();
        adv_t[i]   = tup[1].cast<py::array_t<float>>();
        adv_w[i]   = tup[2].cast<py::array_t<float>>();
        adv_bufs[i] = {
            static_cast<float*>(adv_s[i].request().ptr),
            static_cast<float*>(adv_t[i].request().ptr),
            static_cast<float*>(adv_w[i].request().ptr),
            tup[3].cast<int>(), 122, 5, tup[4].cast<int>()
        };
    }

    // Unpack 4 strategy buffers
    std::vector<py::array_t<float>> str_s(4), str_t(4), str_w(4);
    std::vector<CppReservoirBuffer> strat_bufs(4);
    for (int i = 0; i < 4; i++) {
        auto tup   = strat_buf_list[i].cast<py::tuple>();
        str_s[i]   = tup[0].cast<py::array_t<float>>();
        str_t[i]   = tup[1].cast<py::array_t<float>>();
        str_w[i]   = tup[2].cast<py::array_t<float>>();
        strat_bufs[i] = {
            static_cast<float*>(str_s[i].request().ptr),
            static_cast<float*>(str_t[i].request().ptr),
            static_cast<float*>(str_w[i].request().ptr),
            tup[3].cast<int>(), 122, 5, tup[4].cast<int>()
        };
    }

    // Phase 1: parallel traversal, no locks
    int n_threads = (g_num_threads > 0)
        ? std::min(g_num_threads, (int)std::thread::hardware_concurrency())
        : std::max(1, (int)std::thread::hardware_concurrency());
    std::vector<std::vector<LocalSample>> per_thread(n_threads);
    std::vector<std::thread> threads;
    threads.reserve(n_threads);

    int base_k = k_traversals / n_threads;
    int extra_k = k_traversals % n_threads;

    for (int t_idx = 0; t_idx < n_threads; t_idx++) {
        int thread_k = base_k + (t_idx < extra_k ? 1 : 0);
        if (thread_k == 0) continue;

        threads.emplace_back([&, t_idx, thread_k]() {
            std::mt19937 rng(std::random_device{}() ^ (uint32_t)t_idx);
            auto& local = per_thread[t_idx];
            local.reserve(thread_k * 100);
            for (int k = 0; k < thread_k; k++) {
                RoyalState state; state.reset();
                traverse(state, 0, iteration, local, rng);
                traverse(state, 1, iteration, local, rng);
            }
        });
    }
    for (auto& th : threads) th.join();

    // Phase 2: serial merge
    std::mt19937 merge_rng(std::random_device{}());
    for (auto& local : per_thread) {
        for (const auto& s : local) {
            if (s.buf_id <= 7) {
                adv_bufs[s.buf_id].push(s.state, s.target, s.weight, merge_rng);
            } else {
                strat_bufs[s.buf_id - 8].push(s.state, s.target, s.weight, merge_rng);
            }
        }
    }

    // Return updated insertion counts
    py::dict result;
    for (int i = 0; i < 8; i++)
        result[py::str("adv_n_" + std::to_string(i))] = adv_bufs[i].n_inserted;
    for (int i = 0; i < 4; i++)
        result[py::str("strat_n_" + std::to_string(i))] = strat_bufs[i].n_inserted;
    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(poker_cpp, m) {
    m.doc() = "Royal Hold'em C++ engine — Deep CFR with 5 actions and per-round networks";

    init_blas();

    py::class_<RoyalState>(m, "RoyalState")
        .def(py::init<>())
        .def("reset", [](RoyalState& s, int seed) -> RoyalState& {
            return s.reset(seed);
        }, py::arg("seed") = -1, py::return_value_policy::reference)
        .def("copy",              &RoyalState::copy)
        .def("legal_actions",     &RoyalState::legal_actions)
        .def("apply_action",      &RoyalState::apply_action,
             py::return_value_policy::reference)
        .def("payoff",            &RoyalState::payoff)
        .def("visible_community", &RoyalState::visible_community)
        .def("__str__",           &RoyalState::to_string)
        .def_readwrite("round",    &RoyalState::round)
        .def_readwrite("pot",      &RoyalState::pot)
        .def_readwrite("to_act",   &RoyalState::to_act)
        .def_readwrite("done",     &RoyalState::done)
        .def_readwrite("winner",   &RoyalState::winner)
        .def_readwrite("raises",   &RoyalState::raises)
        .def_readwrite("n_history",&RoyalState::n_history)
        .def_readwrite("n_acted",  &RoyalState::n_acted)
        .def_property_readonly("stacks", [](const RoyalState& s) {
            return std::vector<int>{s.stacks[0], s.stacks[1]};
        })
        .def_property_readonly("bets", [](const RoyalState& s) {
            return std::vector<int>{s.bets[0], s.bets[1]};
        })
        .def_property_readonly("private", [](const RoyalState& s) {
            return py::make_tuple(
                std::vector<int>{s.private_cards[0][0], s.private_cards[0][1]},
                std::vector<int>{s.private_cards[1][0], s.private_cards[1][1]}
            );
        })
        .def_property_readonly("community_cards", [](const RoyalState& s) {
            return std::vector<int>(s.community_cards, s.community_cards + 5);
        })
        .def_property_readonly("history", [](const RoyalState& s) {
            return std::vector<int>(s.history, s.history + s.n_history);
        });

    m.def("card_str", &card_str);

    m.def("update_model_cache", &update_model_cache,
        "Sync PyTorch weights for one player/round combo (122→256→256→5) to C++ cache.",
        py::arg("player"), py::arg("round"),
        py::arg("fc1_w"), py::arg("fc1_b"),
        py::arg("fc2_w"), py::arg("fc2_b"),
        py::arg("out_w"), py::arg("out_b")
    );

    m.def("run_traversals", &run_traversals,
        py::arg("k_traversals"), py::arg("iteration"),
        py::arg("adv_buf_list"),
        py::arg("strat_buf_list")
    );

    m.def("set_num_threads", &set_num_threads,
        "Cap the number of CPU threads used by run_traversals (0 = all cores).",
        py::arg("n")
    );

    m.attr("FOLD")      = FOLD;
    m.attr("CALL")      = CALL;
    m.attr("RAISE_S")   = RAISE_S;
    m.attr("RAISE_M")   = RAISE_M;
    m.attr("RAISE_L")   = RAISE_L;
    m.attr("N_CARDS")   = N_CARDS;
    m.attr("N_ACTIONS") = N_ACTIONS;
}
