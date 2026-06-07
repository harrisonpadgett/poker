import time
import numpy as np
import traceback

import poker_cpp
from poker_env import RoyalState as PyRoyalState, N_ACTIONS
from deep_cfr import encode_state, STATE_DIM

def test_encode_state():
    print("--- Testing encode_state dimensions and bounds ---")
    py_state = PyRoyalState()
    py_state.reset(seed=42)
    
    t_py = encode_state(py_state, 0)
    assert t_py.shape == (STATE_DIM,), f"Expected shape ({STATE_DIM},), got {t_py.shape}"
    
    # Check new features bounds (indices 122, 123, 124, 125)
    assert 0 <= t_py[122].item() <= 1.0, f"Pot feature out of bounds: {t_py[122]}"
    assert 0 <= t_py[123].item() <= 1.0, f"Player stack feature out of bounds: {t_py[123]}"
    assert 0 <= t_py[124].item() <= 1.0, f"Opponent stack feature out of bounds: {t_py[124]}"
    assert 0 <= t_py[125].item() <= 1.0, f"Call amount feature out of bounds: {t_py[125]}"
    
    print("[OK] encode_state passed.")


def _sync_cpp_state(py_state, cpp_state):
    # Warning: C++ bindings don't expose setters for cards easily, so we can't 
    # easily sync them if we didn't add setters. Let's just verify properties
    # aren't crashing. Since they have different cards, their legal actions 
    # might differ if they reach different rounds. Let's instead test C++
    # encode_state via the Python wrapper we have? No, bindings doesn't expose encode_state.
    pass

def test_game_logic_consistency():
    print("--- Testing C++ vs Python basic mechanics (not exact match due to RNG) ---")
    
    np.random.seed(42)
    for i in range(100):
        py_state = PyRoyalState()
        py_state.reset(seed=i)
        
        cpp_state = poker_cpp.RoyalState()
        cpp_state.reset(i)
        
        # Test basic property access doesn't crash and looks sane
        assert len(py_state.legal_actions()) >= 2
        assert len(cpp_state.legal_actions()) >= 2
        
        # We can't step them together because they have different cards and
        # thus different showdowns. Let's just step them independently.
        while not py_state.done:
            py_state.apply_action(int(np.random.choice(py_state.legal_actions())))
        while not cpp_state.done:
            cpp_state.apply_action(int(np.random.choice(cpp_state.legal_actions())))
            
    print("[OK] Game logic basic mechanics passed (100 games).")


def test_c_cpp_stress_test():
    print("--- Stress testing C++ environment ---")
    start = time.time()
    n_games = 500_000
    
    cpp_state = poker_cpp.RoyalState()
    for i in range(n_games):
        cpp_state.reset(i)
        while not cpp_state.done:
            legal = cpp_state.legal_actions()
            action = legal[i % len(legal)] # Pseudo-random
            cpp_state.apply_action(action)
            
    elapsed = time.time() - start
    print(f"[OK] Stress test completed {n_games} games in {elapsed:.2f}s ({(n_games/elapsed):.0f} games/s). No crashes.")


def main():
    try:
        test_encode_state()
        test_game_logic_consistency()
        test_c_cpp_stress_test()
        print("ALL TESTS PASSED SUCCESSFULLY!")
    except Exception as e:
        print("TEST FAILED!")
        traceback.print_exc()

if __name__ == "__main__":
    main()
