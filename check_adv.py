"""
check_adv.py — Quick sanity-check for advantage network predictions.

Evaluates the preflop advantage network (player 0, round 0) on
Pocket Aces and Pocket Tens after loading a checkpoint.
"""
import torch
import numpy as np
from deep_cfr import DeepCFRTrainer, encode_state, N_ACTIONS
from train_terminal import make_dummy_state

trainer = DeepCFRTrainer(adv_buffer_size=50_000, strat_buffer_size=50_000)
trainer.load_checkpoint('checkpoint.pt')

# Pocket Aces: Ac=4, Ad=9  |  Pocket Tens: Tc=0, Td=5
for label, cards in [("Pocket Aces (Ac Ad)", [4, 9]), ("Pocket Tens (Tc Td)", [0, 5])]:
    s = make_dummy_state(cards)
    t = encode_state(s, 0)

    trainer.adv_nets[0][0].eval()
    with torch.no_grad():
        adv = trainer.adv_nets[0][0](t.unsqueeze(0)).squeeze().numpy() * 100.0

    pos = np.maximum(adv, 0)
    total = pos.sum()
    strategy = pos / total if total > 0 else np.ones(N_ACTIONS) / N_ACTIONS

    legal = s.legal_actions()
    strat_probs = trainer.get_average_strategy_from_tensor(t, legal, round=0)

    print(f"\n--- {label} (Preflop, P0) ---")
    action_names = ['Fold', 'Call', 'Raise-S', 'Raise-M', 'Raise-L']
    for i, name in enumerate(action_names):
        avail = "✓" if i in legal else " "
        print(f"  {avail} {name:8s}  ADV={adv[i]:+7.2f}  regret_strat={strategy[i]:.3f}  "
              f"blueprint={strat_probs[i]:.3f}")
