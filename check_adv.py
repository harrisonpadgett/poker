import torch
from deep_cfr import DeepCFRTrainer, encode_state
from train_terminal import make_dummy_state

trainer = DeepCFRTrainer(adv_buffer_size=50000, strat_buffer_size=50000)
trainer.load_checkpoint('checkpoint.pt')

# Evaluate Pocket Aces (Cards 4, 9)
s_AA = make_dummy_state([4, 9])
t_AA = encode_state(s_AA, 0)

print("\n--- Raw Advantage Predictions for P0 Pocket Aces (Preflop) ---")
trainer.adv_nets[0].eval()
with torch.no_grad():
    adv_preds = trainer.adv_nets[0](t_AA.unsqueeze(0)).squeeze()
    print(f"Fold Advantage:  {adv_preds[0].item():.4f} chips")
    print(f"Call Advantage:  {adv_preds[1].item():.4f} chips")
    print(f"Raise Advantage: {adv_preds[2].item():.4f} chips")

    # How Regret Matching converts this to strategy:
    pos_adv = torch.clamp(adv_preds, min=0)
    sum_pos = pos_adv.sum()
    if sum_pos > 0:
        strategy = pos_adv / sum_pos
    else:
        strategy = torch.ones(3) / 3
        
    print(f"\nRegret Matched Strategy for this specific iteration:")
    print(f"Fold={strategy[0].item():.2f}, Call={strategy[1].item():.2f}, Raise={strategy[2].item():.2f}")
