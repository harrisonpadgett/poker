from deep_cfr import DeepCFRTrainer
import time

def main():
    print("=================================================================")
    print("                     DEEP CFR TRAINING (CPU)                     ")
    print("=================================================================\n")
    
    # Initialize trainer with reasonable buffer sizes
    trainer = DeepCFRTrainer(adv_buffer_size=100000, strat_buffer_size=100000)
    
    # We will print out the progress in a clean tabular format
    print(f"{'ITER':<6} | {'ADV LOSS (MSE)':<15} | {'STRAT LOSS (CE)':<15} | {'EXPLOITABILITY':<15}")
    print("-" * 65)
    
    try:
        start_time = time.time()
        for i in range(1, 10001):
            # 1 iteration now traverses 30 possible deals instead of just 1
            trainer.train(n_iterations=1, k_traversals=50)
            
            if i % 100 == 0 or i == 1:
                adv_loss = f"{trainer.adv_loss:.4f}" if trainer.adv_loss else "0.0000"
                strat_loss = f"{trainer.strat_loss:.4f}" if trainer.strat_loss else "0.0000"
                
                if i > 1:
                    elapsed = time.time() - start_time
                    remaining_iters = 10000 - i
                    remaining_time = (remaining_iters / 100) * elapsed
                    print(f"Time for last 100 iterations: {elapsed:.2f}s (Estimated {remaining_time:.2f}s remaining)")
                    start_time = time.time()
                
                expl = trainer.compute_exploitability()
                expl_str = f"{expl:.4f}"
                print(f"{i:<6} | {adv_loss:<15} | {strat_loss:<15} | {expl_str:<15}")
                
                probs_Jc = trainer.get_average_strategy("P0|Jc|none|", [0, 1, 2])
                print(f"       -> P0 holding Jack (Preflop): Fold={probs_Jc[0]:.2f}, Call={probs_Jc[1]:.2f}, Raise={probs_Jc[2]:.2f}")
                
                probs_Kc = trainer.get_average_strategy("P0|Kc|none|", [0, 1, 2])
                print(f"       -> P0 holding King (Preflop): Fold={probs_Kc[0]:.2f}, Call={probs_Kc[1]:.2f}, Raise={probs_Kc[2]:.2f}")
                print("-" * 65)
    except KeyboardInterrupt:
        print("\nTraining interrupted by user. Exiting...")

if __name__ == "__main__":
    main()
