/*
 * bindings.cpp — pybind11 glue layer (Phase 3: in-memory weight cache).
 *
 * FIX: Loading libtorch's torch/script.h inside a pybind11 extension that is
 * dlopen'd by a Python process that already has PyTorch loaded causes a
 * "double libtorch" conflict — the process gets killed by the OS (OOM / signal 137).
 *
 * SOLUTION: Instead of loading TorchScript models from C++, we receive the
 * model's weight matrices as plain float arrays from Python and perform the
 * forward pass ourselves using raw C++ math. This is possible because our
 * DeepCFRNetwork is simple: 3 linear layers + ReLU + 1 output layer.
 *
 * The forward pass for a linear layer is: output = input @ weight.T + bias
 *
 * BLAS ACCELERATION — why not just `#include <Accelerate/Accelerate.h>`?
 * -------------------------------------------------------------------------
 * Adding Accelerate as a *link-time* dependency of our .so causes a
 * framework double-initialisation conflict: PyTorch's libtorch_cpu.dylib
 * already links Accelerate and both initialisers run in the same process,
 * killing it with SIGKILL (exit 137) — the same class of problem that
 * previously prevented using libtorch here.
 *
 * The fix is to look up cblas_sgemv at *runtime* via dlsym(RTLD_DEFAULT).
 * RTLD_DEFAULT searches the process's already-loaded symbol table, so we
 * find the copy that PyTorch loaded for free — zero extra framework load,
 * zero init conflict, full BLAS performance on macOS (AMX on Apple Silicon,
 * LAPACK-optimised BLAS on x86).  On platforms where Accelerate/BLAS is not
 * present the pointer stays null and we fall back to the scalar loop.
 */

#include <pybind11/pybind11.h>
#include <pybind11/stl.h>
#include <pybind11/numpy.h>
#include <cstring>
#include <cmath>
#include <stdexcept>
#include "poker_env.h"

#if defined(__APPLE__) || defined(__linux__)
#  include <dlfcn.h>
#  define HAVE_DLSYM 1
#endif

// ---------------------------------------------------------------------------
// Lazy BLAS handle — resolved once at first use
//
// cblas_sgemv signature (row-major):
//   void cblas_sgemv(CBLAS_ORDER, CBLAS_TRANSPOSE,
//                    int M, int N,
//                    float alpha, const float *A, int lda,
//                    const float *X, int incX,
//                    float beta,          float *Y, int incY);
// ---------------------------------------------------------------------------
enum { CBLAS_ROW_MAJOR = 101, CBLAS_NO_TRANS = 111 };

using cblas_sgemv_t = void (*)(int order, int trans,
                                int M, int N,
                                float alpha, const float* A, int lda,
                                const float* X, int incX,
                                float beta,  float*       Y, int incY);

static cblas_sgemv_t g_sgemv      = nullptr;
static bool          g_sgemv_init = false;

static cblas_sgemv_t resolve_sgemv() {
#if defined(HAVE_DLSYM)
    void* sym = dlsym(RTLD_DEFAULT, "cblas_sgemv");
    return reinterpret_cast<cblas_sgemv_t>(sym);   // null if not found
#else
    return nullptr;
#endif
}

// Called once; safe to call from module init (single-threaded at that point).
static void init_blas() {
    if (!g_sgemv_init) {
        g_sgemv      = resolve_sgemv();
        g_sgemv_init = true;
    }
}

namespace py = pybind11;

// ---------------------------------------------------------------------------
// Lightweight neural net: 3-layer MLP with ReLU, no libtorch needed
// Matches DeepCFRNetwork: Linear(89,256) -> ReLU -> Linear(256,256) -> ReLU
//                         -> Linear(256,256) -> ReLU -> Linear(256,3)
// ---------------------------------------------------------------------------
struct LinearLayer {
    std::vector<float> weight;  // [out_features, in_features]
    std::vector<float> bias;    // [out_features]
    int in_features;
    int out_features;

    void forward(const float* input, float* output, bool relu) const {
        if (g_sgemv) {
            // Fast path: use BLAS SGEMV resolved from the process symbol table.
            // y = W·x + b  (row-major weight matrix, CblasNoTrans)
            memcpy(output, bias.data(), out_features * sizeof(float));
            g_sgemv(CBLAS_ROW_MAJOR, CBLAS_NO_TRANS,
                    out_features, in_features,
                    1.0f, weight.data(), in_features,
                    input, 1,
                    1.0f, output, 1);
        } else {
            // Scalar fallback — also benefits from -O3 -march=native
            // auto-vectorisation (NEON on Apple Silicon).
            for (int o = 0; o < out_features; o++) {
                float sum = bias[o];
                const float* row = weight.data() + o * in_features;
                for (int i = 0; i < in_features; i++) sum += row[i] * input[i];
                output[o] = sum;
            }
        }
        if (relu)
            for (int i = 0; i < out_features; i++)
                output[i] = output[i] > 0.0f ? output[i] : 0.0f;
    }
};

struct CppMLP {
    LinearLayer fc1, fc2, fc3, out_layer;
    bool warm = false;

    // Temporary buffers for intermediate activations (stack-allocated sizes)
    void forward(const float input[89], float output[3]) const {
        float h1[256], h2[256], h3[256];
        fc1.forward(input,  h1, true);
        fc2.forward(h1,     h2, true);
        fc3.forward(h2,     h3, true);
        out_layer.forward(h3, output, false);
    }
};

// ---------------------------------------------------------------------------
// Static model cache — two MLPs, one per player
// ---------------------------------------------------------------------------
static CppMLP g_net0, g_net1;

static void load_layer(LinearLayer& layer, py::array_t<float> weight_arr,
                       py::array_t<float> bias_arr, int in_f, int out_f) {
    layer.in_features  = in_f;
    layer.out_features = out_f;
    auto wbuf = weight_arr.request();
    auto bbuf = bias_arr.request();
    layer.weight.assign(static_cast<float*>(wbuf.ptr),
                        static_cast<float*>(wbuf.ptr) + in_f * out_f);
    layer.bias.assign(static_cast<float*>(bbuf.ptr),
                      static_cast<float*>(bbuf.ptr) + out_f);
}

// Called from Python after train_adv_network() to sync weights to C++
// Python passes: weight and bias tensors for each of the 4 layers
void update_model_cache(
    int player,
    py::array_t<float> fc1_w, py::array_t<float> fc1_b,
    py::array_t<float> fc2_w, py::array_t<float> fc2_b,
    py::array_t<float> fc3_w, py::array_t<float> fc3_b,
    py::array_t<float> out_w, py::array_t<float> out_b
) {
    CppMLP& net = (player == 0) ? g_net0 : g_net1;
    load_layer(net.fc1,       fc1_w, fc1_b,  89,  256);
    load_layer(net.fc2,       fc2_w, fc2_b, 256,  256);
    load_layer(net.fc3,       fc3_w, fc3_b, 256,  256);
    load_layer(net.out_layer, out_w, out_b, 256,    3);
    net.warm = true;
}

// ---------------------------------------------------------------------------
// Traversal — same algorithm as Python, using the cached MLP for inference
// ---------------------------------------------------------------------------
#include <random>
#include <algorithm>
#include <mutex>
#include <thread>
#include <vector>

struct CppReservoirBuffer {
    float* states_ptr;
    float* targets_ptr;
    float* weights_ptr;
    int    capacity;
    int    state_dim;
    int    target_dim;
    int    n_inserted;
    std::mutex mtx;

    void push(const float* state, const float* target, float weight,
              std::mt19937& rng) {
        std::lock_guard<std::mutex> lock(mtx);
        int idx;
        if (n_inserted < capacity) {
            idx = n_inserted;
        } else {
            std::uniform_int_distribution<int> dist(0, n_inserted);
            idx = dist(rng);
            if (idx >= capacity) { n_inserted++; return; }
        }
        memcpy(states_ptr  + (long long)idx * state_dim,  state,  state_dim  * sizeof(float));
        memcpy(targets_ptr + (long long)idx * target_dim, target, target_dim * sizeof(float));
        weights_ptr[idx] = weight;
        n_inserted++;
    }
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

float traverse(RoyalState state, int traverser, int t,
               CppReservoirBuffer& adv_buf_0, CppReservoirBuffer& adv_buf_1,
               CppReservoirBuffer& strat_buf, std::mt19937& rng) {
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
            action_values[a] = traverse(next, traverser, t,
                                        adv_buf_0, adv_buf_1, strat_buf, rng);
        }
        float ev = 0.0f;
        for (int a : legal_actions) ev += strategy[a] * action_values[a];
        float sampled_adv[3] = {};
        for (int a : legal_actions) sampled_adv[a] = action_values[a] - ev;

        CppReservoirBuffer& abuf = (traverser == 0) ? adv_buf_0 : adv_buf_1;
        abuf.push(encoded, sampled_adv, static_cast<float>(t), rng);
        return ev;
    } else {
        strat_buf.push(encoded, strategy, static_cast<float>(t), rng);
        std::uniform_real_distribution<float> dist(0.0f, 1.0f);
        float r = dist(rng), cum = 0.0f;
        int chosen = legal_actions[0];
        for (int a : legal_actions) {
            cum += strategy[a];
            if (r <= cum) { chosen = a; break; }
        }
        RoyalState next = state;
        next.apply_action(chosen);
        return traverse(next, traverser, t, adv_buf_0, adv_buf_1, strat_buf, rng);
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
        throw std::runtime_error("Model cache not warm. Call update_model_cache() for both players first.");

    auto get_ptr = [](py::object obj) -> float* {
        auto arr = obj.cast<py::array_t<float>>();
        py::buffer_info buf = arr.request();
        return static_cast<float*>(buf.ptr);
    };

    CppReservoirBuffer adv_buf_0 {
        get_ptr(adv_states_0), get_ptr(adv_targets_0), get_ptr(adv_weights_0),
        adv_capacity_0, 89, 3, adv_n_inserted_0
    };
    CppReservoirBuffer adv_buf_1 {
        get_ptr(adv_states_1), get_ptr(adv_targets_1), get_ptr(adv_weights_1),
        adv_capacity_1, 89, 3, adv_n_inserted_1
    };
    CppReservoirBuffer strat_buf_cpp {
        get_ptr(strat_states), get_ptr(strat_targets), get_ptr(strat_weights),
        strat_capacity, 89, 3, strat_n_inserted
    };

    std::random_device rd;
    // Parallelize traversals using std::thread
    int n_threads = std::thread::hardware_concurrency();
    if (n_threads == 0) n_threads = 4;
    std::vector<std::thread> threads;
    
    // Assign traversals to threads
    int base_k = k_traversals / n_threads;
    int extra_k = k_traversals % n_threads;
    
    for (int t_idx = 0; t_idx < n_threads; t_idx++) {
        int thread_k = base_k + (t_idx < extra_k ? 1 : 0);
        if (thread_k == 0) continue;
        
        threads.emplace_back([&, thread_k, t_idx]() {
            // Each thread needs its own RNG to avoid data races
            std::mt19937 thread_rng(std::random_device{}() + t_idx);
            for (int k = 0; k < thread_k; k++) {
                RoyalState state; state.reset();
                traverse(state, 0, iteration, adv_buf_0, adv_buf_1, strat_buf_cpp, thread_rng);
                traverse(state, 1, iteration, adv_buf_0, adv_buf_1, strat_buf_cpp, thread_rng);
            }
        });
    }
    
    for (auto& th : threads) th.join();

    py::dict result;
    result["adv_n_0"] = adv_buf_0.n_inserted;
    result["adv_n_1"] = adv_buf_1.n_inserted;
    result["strat_n"] = strat_buf_cpp.n_inserted;
    return result;
}

// ---------------------------------------------------------------------------
// Module
// ---------------------------------------------------------------------------
PYBIND11_MODULE(poker_cpp, m) {
    m.doc() = "Royal Hold'em C++ engine + Deep CFR traversal (native MLP, no libtorch)";

    // Resolve cblas_sgemv from the process symbol table (PyTorch has already
    // loaded Accelerate/BLAS by the time this module is imported).
    init_blas();

    py::class_<RoyalState>(m, "RoyalState")
        .def(py::init<>())
        .def("reset", [](RoyalState& s, int seed) -> RoyalState& {
            return s.reset(seed);
        }, py::arg("seed") = -1, py::return_value_policy::reference)
        .def("copy",              &RoyalState::copy)
        .def("legal_actions",     &RoyalState::legal_actions)
        .def("apply_action",      &RoyalState::apply_action, py::return_value_policy::reference)
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
        "Sync PyTorch model weights to C++ cache (call after train_adv_network)",
        py::arg("player"),
        py::arg("fc1_w"), py::arg("fc1_b"),
        py::arg("fc2_w"), py::arg("fc2_b"),
        py::arg("fc3_w"), py::arg("fc3_b"),
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
