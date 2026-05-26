from deep_cfr import DeepCFRTrainer, encode_state, ADV_SCALE
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
    
    trainer = DeepCFRTrainer(adv_buffer_size=2000000, strat_buffer_size=2000000)
    trainer.load_checkpoint('checkpoint.pt')
    
    print(f"{'ITER':<6} | {'ADV LOSS (MSE chips)':<22} | {'STRAT LOSS (CE)':<15}")
    print("-" * 50)
    
    try:
        start_time = time.time()
        start_iter = trainer.iterations + 1
        for i in range(start_iter, 100001):
            trainer.train(n_iterations=1, k_traversals=200)
            
            status_msg = ""
            if i % 100 == 0:
                trainer.save_checkpoint('checkpoint.pt', verbose=False)
                status_msg = f"[+] Checkpoint saved to checkpoint.pt"
            
            adv_loss = f"{trainer.adv_loss:.4f}" if trainer.adv_loss else "0.0000"
            strat_loss = f"{trainer.strat_loss:.4f}" if trainer.strat_loss else "0.0000"
            
            # Evaluate Pocket Aces (Ac, Ad -> cards 4, 9)
            s_AA = make_dummy_state([4, 9])
            t_AA = encode_state(s_AA, 0)
            probs_AA = trainer.get_average_strategy_from_tensor(t_AA, [0, 1, 2])
            
            # Evaluate Pocket Tens (Tc, Td -> cards 0, 5)
            s_TT = make_dummy_state([0, 5])
            t_TT = encode_state(s_TT, 0)
            probs_TT = trainer.get_average_strategy_from_tensor(t_TT, [0, 1, 2])
            
            # Evaluate Advantage Network (The Brain)
            import torch
            trainer.adv_nets[0].eval()
            with torch.no_grad():
                adv_AA = trainer.adv_nets[0](t_AA.to(trainer.device).unsqueeze(0)).squeeze(0).cpu().numpy() * ADV_SCALE
                adv_TT = trainer.adv_nets[0](t_TT.to(trainer.device).unsqueeze(0)).squeeze(0).cpu().numpy() * ADV_SCALE

            # Move cursor up 8 lines if we've already printed a block
            if i > start_iter:
                print("\033[8A", end="")
            
            elapsed = time.time() - start_time
            remaining_iters = 100000 - i
            remaining_time = elapsed * remaining_iters
            
            # Print with \033[K to clear the rest of the line (prevents trailing artifacts)
            print(f"Time for last iteration: {elapsed:.2f}s (Estimated {remaining_time:.2f}s remaining)\033[K")
            print(f"{i:<6} | {adv_loss:<22} | {strat_loss:<15}\033[K")
            print(f"       [BUFFER] AA (Preflop): Fold={probs_AA[0]:.2f}, Call={probs_AA[1]:.2f}, Raise={probs_AA[2]:.2f}\033[K")
            print(f"       [BRAIN]  AA (Preflop): Fold={adv_AA[0]:+.2f}, Call={adv_AA[1]:+.2f}, Raise={adv_AA[2]:+.2f}\033[K")
            print(f"       [BUFFER] TT (Preflop): Fold={probs_TT[0]:.2f}, Call={probs_TT[1]:.2f}, Raise={probs_TT[2]:.2f}\033[K")
            print(f"       [BRAIN]  TT (Preflop): Fold={adv_TT[0]:+.2f}, Call={adv_TT[1]:+.2f}, Raise={adv_TT[2]:+.2f}\033[K")
            print("-" * 50 + "\033[K")
            print(f"{status_msg:<50}\033[K")
            
            start_time = time.time()
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Saving final checkpoint...")
        trainer.save_checkpoint('checkpoint.pt')
        print("Exiting...")

if __name__ == "__main__":
    main()
