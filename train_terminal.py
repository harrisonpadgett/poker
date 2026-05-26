from deep_cfr import DeepCFRTrainer, encode_state, ADV_SCALE, N_ACTIONS
from poker_env import RoyalState
import time


def make_dummy_state(private_cards, round=0):
    """Create a fresh preflop state with the given hole cards for Player 0."""
    s = RoyalState().reset()
    s.round   = round
    s.history = []
    s.n_acted = 0
    s.private[0] = list(private_cards)
    return s


def main():
    print("=================================================================")
    print("          DEEP CFR TRAINING (ROYAL HOLD'EM — 5 ACTIONS)          ")
    print("=================================================================\n")

    trainer = DeepCFRTrainer(adv_buffer_size=2_000_000, strat_buffer_size=2_000_000)
    trainer.load_checkpoint('checkpoint.pt')

    # Pocket Aces: Ac=4, Ad=9  |  Pocket Tens: Tc=0, Td=5
    s_AA = make_dummy_state([4, 9])
    s_TT = make_dummy_state([0, 5])
    t_AA = encode_state(s_AA, 0)
    t_TT = encode_state(s_TT, 0)
    legal_AA = s_AA.legal_actions()   # [0,1,2,3,4] from a full stack
    legal_TT = s_TT.legal_actions()

    header = f"{'ITER':<6} | {'ADV LOSS':>10} | {'STRAT LOSS':>10}"
    print(header)
    print("-" * len(header))

    try:
        start_time = time.time()
        start_iter = trainer.iterations + 1
        for i in range(start_iter, 100_001):
            trainer.train(n_iterations=1, k_traversals=800)

            status_msg = ""
            if i % 100 == 0:
                trainer.save_checkpoint('checkpoint.pt', verbose=False)
                status_msg = "[+] Checkpoint saved"

            adv_loss  = f"{trainer.adv_loss:.4f}"  if trainer.adv_loss  else "0.0000"
            strat_loss = f"{trainer.strat_loss:.4f}" if trainer.strat_loss else "0.0000"

            import torch
            # Evaluate the preflop advantage network (round 0) — The Brain
            trainer.adv_nets[0][0].eval()
            with torch.no_grad():
                adv_AA = (trainer.adv_nets[0][0](t_AA.to(trainer.device).unsqueeze(0))
                          .squeeze(0).cpu().numpy() * ADV_SCALE)
                adv_TT = (trainer.adv_nets[0][0](t_TT.to(trainer.device).unsqueeze(0))
                          .squeeze(0).cpu().numpy() * ADV_SCALE)

            probs_AA = trainer.get_average_strategy_from_tensor(t_AA, legal_AA, round=0)
            probs_TT = trainer.get_average_strategy_from_tensor(t_TT, legal_TT, round=0)

            # Move cursor up 10 lines after first iteration
            if i > start_iter:
                print("\033[10A", end="")

            elapsed = time.time() - start_time
            remaining = elapsed * (100_000 - i)

            print(f"Elapsed: {elapsed:.2f}s  ETA: {remaining:.0f}s\033[K")
            print(f"{i:<6} | {adv_loss:>10} | {strat_loss:>10}\033[K")
            print(f"  [BUF] AA: F={probs_AA[0]:.2f} C={probs_AA[1]:.2f} "
                  f"RS={probs_AA[2]:.2f} RM={probs_AA[3]:.2f} RL={probs_AA[4]:.2f}\033[K")
            print(f"  [ADV] AA: F={adv_AA[0]:+.1f} C={adv_AA[1]:+.1f} "
                  f"RS={adv_AA[2]:+.1f} RM={adv_AA[3]:+.1f} RL={adv_AA[4]:+.1f}\033[K")
            print(f"  [BUF] TT: F={probs_TT[0]:.2f} C={probs_TT[1]:.2f} "
                  f"RS={probs_TT[2]:.2f} RM={probs_TT[3]:.2f} RL={probs_TT[4]:.2f}\033[K")
            print(f"  [ADV] TT: F={adv_TT[0]:+.1f} C={adv_TT[1]:+.1f} "
                  f"RS={adv_TT[2]:+.1f} RM={adv_TT[3]:+.1f} RL={adv_TT[4]:+.1f}\033[K")
            print("-" * len(header) + "\033[K")
            print(f"Buffers: adv[0][0]={trainer.adv_buffers[0][0].n_inserted:,} "
                  f"adv[1][0]={trainer.adv_buffers[1][0].n_inserted:,} "
                  f"strat[0]={trainer.strat_buffers[0].n_inserted:,}\033[K")
            print(f"         adv[0][3]={trainer.adv_buffers[0][3].n_inserted:,} "
                  f"adv[1][3]={trainer.adv_buffers[1][3].n_inserted:,} "
                  f"strat[3]={trainer.strat_buffers[3].n_inserted:,}\033[K")
            print(f"{status_msg:<60}\033[K")

            start_time = time.time()

    except KeyboardInterrupt:
        print("\nTraining interrupted. Saving checkpoint...")
        trainer.save_checkpoint('checkpoint.pt')
        print("Done.")


if __name__ == "__main__":
    main()
