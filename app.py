from flask import Flask, request, jsonify
from flask_cors import CORS
import poker_cpp
from poker_env import RoyalState, N_ACTIONS, ACTION_NAMES, FOLD, get_raise_sizes
from deep_cfr import DeepCFRTrainer, encode_state, ADV_SCALE
import torch
import numpy as np
import random
import os

app = Flask(__name__)
CORS(app)

# ---------------------------------------------------------------------------
# AI initialisation
# ---------------------------------------------------------------------------
print("Loading Deep CFR AI...")
trainer = DeepCFRTrainer(adv_buffer_size=10, strat_buffer_size=10)

if os.path.exists('inference_checkpoint.pt'):
    print("Found inference_checkpoint.pt. Loading...")
    trainer.load_checkpoint('inference_checkpoint.pt')
elif os.path.exists('checkpoint.pt'):
    print("Found checkpoint.pt. Loading...")
    trainer.load_checkpoint('checkpoint.pt')
else:
    print("WARNING: No checkpoint found! AI will play completely randomly.")

# Game state
current_state = None
human_player  = 0

# Last AI decision — sent to the frontend for the Intelligence Panel
_last_ai_move = {'action': None, 'probs': None, 'advantages': None}


def get_state_dict():
    global current_state, human_player, _last_ai_move
    if current_state is None:
        return {"error": "Game not started"}

    ai_player      = 1 - human_player
    community_strs = [poker_cpp.card_str(c) for c in current_state.visible_community()]
    private_strs   = [poker_cpp.card_str(c) for c in current_state.private[human_player]]
    ai_strs        = ([poker_cpp.card_str(c) for c in current_state.private[ai_player]]
                      if current_state.done else [])

    call_amount = (current_state.bets[ai_player] - current_state.bets[human_player])
    call_amount = max(0, min(call_amount, current_state.stacks[human_player]))

    # Pot-relative raise amounts (what the human would put in total for each raise)
    raise_sizes = get_raise_sizes(current_state.pot, call_amount)
    raise_amounts = {
        'raise_amount_s': call_amount + raise_sizes[0],
        'raise_amount_m': call_amount + raise_sizes[1],
        'raise_amount_l': call_amount + raise_sizes[2],
    }

    return {
        "round":          current_state.round,
        "pot":            current_state.pot,
        "to_act":         current_state.to_act,
        "human_player":   human_player,
        "human_stack":    current_state.stacks[human_player],
        "human_bet":      current_state.bets[human_player],
        "ai_stack":       current_state.stacks[ai_player],
        "ai_bet":         current_state.bets[ai_player],
        "done":           current_state.done,
        "winner":         current_state.winner,
        "raises":         current_state.raises,
        "community":      community_strs,
        "private":        private_strs,
        "ai_private":     ai_strs,
        "call_amount":    call_amount,
        "is_check":       call_amount == 0,
        "legal_actions":  current_state.legal_actions(),
        "history":        current_state.history,
        "is_human_turn":  current_state.to_act == human_player and not current_state.done,
        "last_ai_move":   _last_ai_move,
        **raise_amounts,
    }


def ai_play_turn():
    """Execute the AI's turn, capturing the full decision for the Intelligence Panel."""
    global current_state, human_player, _last_ai_move
    if current_state.done or current_state.to_act == human_player:
        return

    ai_player = 1 - human_player
    legal     = current_state.legal_actions()

    # Strategy Network — Learned Nash Equilibrium
    t     = encode_state(current_state, ai_player)
    probs = trainer.get_average_strategy_from_tensor(t, legal, current_state.round)

    # Advantage Network — raw output for the Intelligence Panel
    try:
        with torch.no_grad():
            adv_raw = trainer.adv_nets[ai_player][current_state.round](
                t.to(trainer.device).unsqueeze(0)
            ).squeeze(0).cpu().numpy() * ADV_SCALE
        advantages = [round(float(adv_raw[i]), 3) for i in range(N_ACTIONS)]
    except Exception:
        advantages = [0.0] * N_ACTIONS

    # Sample action from strategy distribution
    p = np.array([probs[a] for a in legal], dtype=np.float64)
    if p.sum() == 0:
        p = np.ones_like(p) / len(p)
    else:
        p /= p.sum()

    action = int(np.random.choice(legal, p=p))
    print(f"AI plays {ACTION_NAMES[action]} | "
          f"F={probs[0]:.2f} C={probs[1]:.2f} "
          f"RS={probs[2]:.2f} RM={probs[3]:.2f} RL={probs[4]:.2f}")

    ai_stack_before = current_state.stacks[ai_player]
    current_state.apply_action(action)
    chips_pushed = ai_stack_before - current_state.stacks[ai_player]

    _last_ai_move = {
        'action':       action,
        'probs':        [round(float(probs[i]), 4) for i in range(N_ACTIONS)],
        'advantages':   advantages,
        'chips_pushed': chips_pushed,
    }


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/api/start', methods=['POST'])
def start_game():
    global current_state, human_player, _last_ai_move

    seed = random.randint(0, 1_000_000)
    current_state = poker_cpp.RoyalState()
    current_state.reset(seed)
    human_player  = random.choice([0, 1])
    _last_ai_move = {'action': None, 'probs': None, 'advantages': None}
    print(f"New game. Human is Player {human_player}")

    if current_state.to_act != human_player and not current_state.done:
        ai_play_turn()

    return jsonify(get_state_dict())


@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(get_state_dict())


@app.route('/api/action', methods=['POST'])
def take_action():
    global current_state, human_player
    data   = request.json
    action = data.get('action')

    if current_state is None or current_state.done:
        return jsonify({"error": "Game is over or not started"}), 400
    if current_state.to_act != human_player:
        return jsonify({"error": "Not your turn"}), 400
    if action not in current_state.legal_actions():
        return jsonify({"error": "Illegal action"}), 400

    current_state.apply_action(action)

    while not current_state.done and current_state.to_act != human_player:
        ai_play_turn()

    return jsonify(get_state_dict())


if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
