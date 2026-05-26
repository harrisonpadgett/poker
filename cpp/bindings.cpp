/*
 * bindings.cpp — pybind11 glue layer.
 *
 * FIX: Loading libtorch inside a pybind11 extension dlopen'd by a process
 * that already has PyTorch causes a double-libtorch conflict (SIGKILL).
 * SOLUTION: receive weight matrices as plain float arrays from Python and
 * perform the forward pass in raw C++.
 *
 * BLAS: cblas_sgemv is resolved at runtime via dlsym(RTLD_DEFAULT) so we
 * piggyback on the copy PyTorch already loaded — no link-time dependency,
 * no double-init conflict.
 *
 * Network: 89 → 128 → 128 → 3  (matches DeepCFRNetwork in deep_cfr.py)
 *
 * Per-thread buffers: each traversal thread accumulates samples into a
 * thread-local vector; after all threads join a single serial merge loop
 * inserts them into the shared reservoir.  This eliminates the std::mutex
 * that previously serialised every push() call.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include <random>
#include <algorithm>
#include <thread>
#include <vector>
#include "poker_env.h"

#if defined(__APPLE__) || defined(__linux__)
#  include <dlfcn.h>
#  define HAVE_DLSYM 1
#endif

// ---------------------------------------------------------------------------
// Lazy BLAS handle
// ---------------------------------------------------------------------------
enum { CBLAS_ROW_MAJOR = 101, CBLAS_NO_TRANS = 111 };

using cblas_sgemv_t = void (*)(int, int, int, int,
                                float, const float*, int,
                                const float*, int,
                                float, float*, int);

static cblas_sgemv_t g_sgemv      = nullptr;
static bool          g_sgemv_init = false;

static void init_blas() {
    if (!g_sgemv_init) {
#if defined(HAVE_DLSYM)
        g_sgemv = reinterpret_cast<cblas_sgemv_t>(
                      dlsym(RTLD_DEFAULT, "cblas_sgemv"));
#endif
        g_sgemv_init = true;
    }
}

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Lightweight MLP — matches DeepCFRNetwork: 89→128→128→3
// ---------------------------------------------------------------------------
struct LinearLayer {
    std::vector<float> weight;   // [out_features × in_features], row-major
    std::vector<float> bias;     // [out_features]
    int in_features;
    int out_features;

    void forward(const float* input, float* output, bool relu) const {
        if (g_sgemv) {
            memcpy(output, bias.data(), out_features * sizeof(float));
            g_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                    out_features, in_features,
                    1.0f, weight.data(), in_features,
                    input, 1,
                    1.0f, output, 1);
        } else {
            for (int o = 0; o < out_features; o++) {
                float s = bias[o];
                const float* row = weight.data() + o * in_features;
                for (int i = 0; i < in_features; i++) s += row[i] * input[i];
                output[o] = s;
            }
        }
        if (relu)
            for (int i = 0; i < out_features; i++)
                output[i] = output[i] > 0.0f ? output[i] : 0.0f;
    }
};

struct CppMLP {
    LinearLayer fc1, fc2, out_layer;   // 89→256, 256→256, 256→3
    bool warm = false;

    void forward(const float input[89], float output[3]) const {
        float h1[256], h2[256];
        fc1.forward(input, h1, true);
        fc2.forward(h1,    h2, true);
        out_layer.forward(h2, output, false);
    }
};

static CppMLP g_net0, g_net1;

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

// Called from Python after train_adv_network() to sync weights to C++.
// Network is now 89→256→256→3, so 3 layers (6 tensor arguments).
void update_model_cache(
    int player,
    py::array_t<float> fc1_w, py::array_t<float> fc1_b,
    py::array_t<float> fc2_w, py::array_t<float> fc2_b,
    py::array_t<float> out_w, py::array_t<float> out_b
) {
    CppMLP& net = (player == 0) ? g_net0 : g_net1;
    load_layer(net.fc1,       fc1_w, fc1_b,  89, 256);
    load_layer(net.fc2,       fc2_w, fc2_b, 256, 256);
    load_layer(net.out_layer, out_w, out_b,  256,   3);
    net.warm = true;
}

// ---------------------------------------------------------------------------
// Reservoir buffer — now mutex-free.  Only called from the serial merge
// loop after all traversal threads have joined, so no concurrency needed.
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
            // Cap the effective insertion count at 2×capacity.
            // Pure reservoir sampling (denominator = n_inserted) freezes the
            // buffer once n_inserted >> capacity: at n=377M, capacity=100k,
            // each new sample has only a 0.027% insertion probability, making
            // the buffer permanently stuck with ancient data.
            // Capping at 2×capacity keeps insertion probability ≥ 50% forever,
            // so the buffer fully refreshes every ~2*capacity/samples_per_iter
            // iterations (~4k iters for the strategy buffer).
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
// Per-thread sample accumulation
// ---------------------------------------------------------------------------
struct LocalSample {
    float state[89];
    float target[3];
    float weight;
    int   buf_id;   // 0 = adv_buf_0, 1 = adv_buf_1, 2 = strat_buf
};

void encode_state(const RoyalState& state, int player, float out[89]) {
    memset(out, 0, 89 * sizeof(float));
    out[0] = static_cast<float>(player);
    for (int i = 0; i < 2; i++) out[1 + state.private_cards[player][i]] = 1.0f;
    auto vc = state.visible_community();
    for (int c : vc) out[21 + c] = 1.0f;
    for (int i = 0; i < state.n_history && i < 16; i++)
        out[41 + i * 3 + state.history[i]] = 1.0f;
}

// traverse() now collects into a thread-local vector instead of pushing
// directly to a mutex-protected shared buffer.
float traverse(RoyalState state, int traverser, int t,
               std::vector<LocalSample>& samples, std::mt19937& rng) {
    if (state.done) return state.payoff(traverser);

    int player = state.to_act;
    auto legal_actions = state.legal_actions();
    int n_legal = static_cast<int>(legal_actions.size());

    float encoded[89];
    encode_state(state, player, encoded);

    float advantages[3] = {};
    const CppMLP& net = (player == 0) ? g_net0 : g_net1;
    net.forward(encoded, advantages);

    float strategy[3] = {};
    float sum_pos = 0.0f;
    for (int a : legal_actions) sum_pos += std::max(0.0f, advantages[a]);
    if (sum_pos > 0.0f)
        for (int a : legal_actions) strategy[a] = std::max(0.0f, advantages[a]) / sum_pos;
    else
        for (int a : legal_actions) strategy[a] = 1.0f / n_legal;

    if (player == traverser) {
        float action_values[3] = {};
        for (int a : legal_actions) {
            RoyalState next = state;
            next.apply_action(a);
            action_values[a] = traverse(next, traverser, t, samples, rng);
        }
        float ev = 0.0f;
        for (int a : legal_actions) ev += strategy[a] * action_values[a];
        float sampled_adv[3] = {};
        for (int a : legal_actions) sampled_adv[a] = action_values[a] - ev;

        LocalSample s;
        memcpy(s.state,  encoded,     89 * sizeof(float));
        memcpy(s.target, sampled_adv,  3 * sizeof(float));
        s.weight = static_cast<float>(t);
        s.buf_id = traverser;   // 0 or 1
        samples.push_back(s);
        return ev;
    } else {
        LocalSample s;
        memcpy(s.state,  encoded,  89 * sizeof(float));
        memcpy(s.target, strategy,  3 * sizeof(float));
        s.weight = static_cast<float>(t);
        s.buf_id = 2;   // strat buffer
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

py::dict run_traversals(
    int k_traversals, int iteration,
    py::object adv_states_0, py::object adv_targets_0, py::object adv_weights_0,
    int adv_capacity_0, int adv_n_inserted_0,
    py::object adv_states_1, py::object adv_targets_1, py::object adv_weights_1,
    int adv_capacity_1, int adv_n_inserted_1,
    py::object strat_states, py::object strat_targets, py::object strat_weights,
    int strat_capacity, int strat_n_inserted
) {
    if (!g_net0.warm || !g_net1.warm)
        throw std::runtime_error(
            "Model cache not warm. Call update_model_cache() for both players first.");

    auto get_ptr = [](py::object obj) -> float* {
        auto arr = obj.cast<py::array_t<float>>();
        return static_cast<float*>(arr.request().ptr);
    };

    CppReservoirBuffer adv_buf_0 {
        get_ptr(adv_states_0), get_ptr(adv_targets_0), get_ptr(adv_weights_0),
        adv_capacity_0, 89, 3, adv_n_inserted_0
    };
    CppReservoirBuffer adv_buf_1 {
        get_ptr(adv_states_1), get_ptr(adv_targets_1), get_ptr(adv_weights_1),
        adv_capacity_1, 89, 3, adv_n_inserted_1
    };
    CppReservoirBuffer strat_buf {
        get_ptr(strat_states), get_ptr(strat_targets), get_ptr(strat_weights),
        strat_capacity, 89, 3, strat_n_inserted
    };

    // ------------------------------------------------------------------
    // Phase 1: each thread traverses its share and collects samples locally.
    // No mutexes — threads never touch each other's data.
    // ------------------------------------------------------------------
    int n_threads = std::max(1, (int)std::thread::hardware_concurrency());
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
            // Reserve a reasonable upper bound to avoid reallocations.
            // Each traversal-pair generates ~60 samples on average.
            local.reserve(thread_k * 70);
            for (int k = 0; k < thread_k; k++) {
                RoyalState state; state.reset();
                traverse(state, 0, iteration, local, rng);
                traverse(state, 1, iteration, local, rng);
            }
        });
    }
    for (auto& th : threads) th.join();

    // ------------------------------------------------------------------
    // Phase 2: single-threaded merge into the shared reservoir buffers.
    // Serial insertion means reservoir sampling is unbiased and there are
    // no data races.  For k=100 this loop processes ~6k samples in <2ms.
    // ------------------------------------------------------------------
    std::mt19937 merge_rng(std::random_device{}());
    for (auto& local : per_thread) {
        for (const auto& s : local) {
            switch (s.buf_id) {
                case 0: adv_buf_0.push(s.state, s.target, s.weight, merge_rng); break;
                case 1: adv_buf_1.push(s.state, s.target, s.weight, merge_rng); break;
                case 2: strat_buf.push(s.state, s.target, s.weight, merge_rng); break;
            }
        }
    }

    py::dict result;
    result["adv_n_0"] = adv_buf_0.n_inserted;
    result["adv_n_1"] = adv_buf_1.n_inserted;
    result["strat_n"] = strat_buf.n_inserted;
    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(poker_cpp, m) {
    m.doc() = "Royal Hold'em C++ engine + Deep CFR traversal (native MLP, no libtorch)";

    // Resolve cblas_sgemv from PyTorch's already-loaded Accelerate/BLAS.
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
        .def_property_readonly("history", [](const RoyalState& s) {
            return std::vector<int>(s.history, s.history + s.n_history);
        });

    m.def("card_str", &card_str);

    m.def("update_model_cache", &update_model_cache,
        "Sync PyTorch model weights (89→128→128→3) to C++ cache",
        py::arg("player"),
        py::arg("fc1_w"), py::arg("fc1_b"),
        py::arg("fc2_w"), py::arg("fc2_b"),
        py::arg("out_w"), py::arg("out_b")
    );

    m.def("run_traversals", &run_traversals,
        py::arg("k_traversals"), py::arg("iteration"),
        py::arg("adv_states_0"),  py::arg("adv_targets_0"),  py::arg("adv_weights_0"),
        py::arg("adv_capacity_0"), py::arg("adv_n_inserted_0"),
        py::arg("adv_states_1"),  py::arg("adv_targets_1"),  py::arg("adv_weights_1"),
        py::arg("adv_capacity_1"), py::arg("adv_n_inserted_1"),
        py::arg("strat_states"),  py::arg("strat_targets"),  py::arg("strat_weights"),
        py::arg("strat_capacity"), py::arg("strat_n_inserted")
    );

    m.attr("FOLD")    = FOLD;
    m.attr("CALL")    = CALL;
    m.attr("RAISE")   = RAISE_A;
    m.attr("N_CARDS") = N_CARDS;
}
