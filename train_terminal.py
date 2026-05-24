from deep_cfr import DeepCFRTrainer, encode_state
from poker_env import RoyalState
import time

def make_dummy_state(private_cards):
    s = RoyalState().reset()
    s.round = 0
    s.history = []
    s.community = []
    s.private[0] = private_cards
    return s

def main():
    print("=================================================================")
    print("               DEEP CFR TRAINING (ROYAL HOLD'EM)                 ")
    print("=================================================================\n")
    
    trainer = DeepCFRTrainer(adv_buffer_size=100000, strat_buffer_size=100000)
    trainer.load_checkpoint('checkpoint.pt')
    
    print(f"{'ITER':<6} | {'ADV LOSS (MSE)':<15} | {'STRAT LOSS (CE)':<15}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        start_iter = trainer.iterations + 1
        for i in range(start_iter, 10001):
            # Reduced k_traversals from 100 to 20 for much faster iterations
            trainer.train(n_iterations=1, k_traversals=20)
            
            if i % 100 == 0:
                trainer.save_checkpoint('checkpoint.pt')
            
            adv_loss = f"{trainer.adv_loss:.4f}" if trainer.adv_loss else "0.0000"
            strat_loss = f"{trainer.strat_loss:.4f}" if trainer.strat_loss else "0.0000"
            
            if i > 1:
                elapsed = time.time() - start_time
                remaining_iters = 10000 - i
                remaining_time = elapsed * remaining_iters
                print(f"Time for last iteration: {elapsed:.2f}s (Estimated {remaining_time:.2f}s remaining)")

            print(f"{i:<6} | {adv_loss:<15} | {strat_loss:<15}")
                
            # Evaluate Pocket Aces (Ac, Ad -> cards 4, 9)
            s_AA = make_dummy_state([4, 9])
            t_AA = encode_state(s_AA, 0)
            probs_AA = trainer.get_average_strategy_from_tensor(t_AA, [0, 1, 2])
            print(f"       -> P0 holding AA (Preflop): Fold={probs_AA[0]:.2f}, Call={probs_AA[1]:.2f}, Raise={probs_AA[2]:.2f}")
            
            # Evaluate Pocket Tens (Tc, Td -> cards 0, 5)
            s_TT = make_dummy_state([0, 5])
            t_TT = encode_state(s_TT, 0)
            probs_TT = trainer.get_average_strategy_from_tensor(t_TT, [0, 1, 2])
            print(f"       -> P0 holding TT (Preflop): Fold={probs_TT[0]:.2f}, Call={probs_TT[1]:.2f}, Raise={probs_TT[2]:.2f}")
            
            print("-" * 50)
            start_time = time.time()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint('checkpoint.pt')
        print("Exiting...")

if __name__ == "__main__":
    main()
