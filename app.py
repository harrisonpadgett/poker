from flask import Flask, render_template, request, jsonify
import poker_cpp
from deep_cfr import DeepCFRTrainer, encode_state, subgame_solve
from poker_env import RAISE_AMOUNTS, N_ACTIONS
import torch
import numpy as np
import random
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# ---------------------------------------------------------------------------
# AI initialisation
# ---------------------------------------------------------------------------
print("Loading Deep CFR AI...")
trainer = DeepCFRTrainer(adv_buffer_size=10, strat_buffer_size=10)
if os.path.exists('checkpoint.pt'):
    trainer.load_checkpoint('checkpoint.pt')
else:
    print("WARNING: checkpoint.pt not found! AI will play completely randomly.")

# Game state
current_state = None
human_player  = 0


def get_state_dict():
    """Serialize the current RoyalState for the frontend."""
    global current_state, human_player
    if current_state is None:
        return {"error": "Game not started"}

    ai_player     = 1 - human_player
    community_strs = [poker_cpp.card_str(c) for c in current_state.visible_community()]
    private_strs   = [poker_cpp.card_str(c) for c in current_state.private[human_player]]
    ai_strs        = ([poker_cpp.card_str(c) for c in current_state.private[ai_player]]
                      if current_state.done else [])

    call_amount = (current_state.bets[ai_player] - current_state.bets[human_player])
    call_amount = max(0, min(call_amount, current_state.stacks[human_player]))

    rnd = current_state.round
    raise_amounts = {
        'raise_amount_s': call_amount + RAISE_AMOUNTS[rnd][0],
        'raise_amount_m': call_amount + RAISE_AMOUNTS[rnd][1],
        'raise_amount_l': call_amount + RAISE_AMOUNTS[rnd][2],
    }

    return {
        "round":         current_state.round,
        "pot":           current_state.pot,
        "to_act":        current_state.to_act,
        "human_player":  human_player,
        "human_stack":   current_state.stacks[human_player],
        "human_bet":     current_state.bets[human_player],
        "ai_stack":      current_state.stacks[ai_player],
        "ai_bet":        current_state.bets[ai_player],
        "done":          current_state.done,
        "winner":        current_state.winner,
        "raises":        current_state.raises,
        "community":     community_strs,
        "private":       private_strs,
        "ai_private":    ai_strs,
        "call_amount":   call_amount,
        "legal_actions": current_state.legal_actions(),
        "history":       current_state.history,
        "is_human_turn": current_state.to_act == human_player and not current_state.done,
        **raise_amounts,
    }


def ai_play_turn():
    """Execute the AI's turn using real-time subgame solving."""
    global current_state, human_player
    if current_state.done or current_state.to_act == human_player:
        return

    ai_player = 1 - human_player
    legal     = current_state.legal_actions()

    # Subgame solve: run live CFR from the current state, considering
    # 20 random opponent hand samples × 5 rollouts per action.
    # Falls back to the raw blueprint if something unexpected occurs.
    try:
        probs = subgame_solve(
            current_state, ai_player, trainer,
            n_opponent_samples=20,
            n_rollouts=5,
        )
    except Exception as e:
        print(f"Subgame solve failed ({e}), falling back to blueprint.")
        t     = encode_state(current_state, ai_player)
        probs = trainer.get_average_strategy_from_tensor(t, legal, current_state.round)

    # Sample from the solved distribution
    p = np.array([probs[a] for a in legal], dtype=np.float64)
    if p.sum() == 0:
        p = np.ones_like(p) / len(p)
    else:
        p /= p.sum()

    action = int(np.random.choice(legal, p=p))
    action_names = ['Fold', 'Call', 'Raise-S', 'Raise-M', 'Raise-L']
    print(f"AI plays {action_names[action]} | "
          f"F={probs[0]:.2f} C={probs[1]:.2f} "
          f"RS={probs[2]:.2f} RM={probs[3]:.2f} RL={probs[4]:.2f}")
    current_state.apply_action(action)


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------

@app.route('/')
def index():
    return render_template('index.html')

last_checkpoint_mtime = 0

def check_and_reload_model():
    global last_checkpoint_mtime
    if os.path.exists('checkpoint.pt'):
        mtime = os.path.getmtime('checkpoint.pt')
        if mtime > last_checkpoint_mtime:
            print("New checkpoint detected! Reloading AI model...")
            trainer.load_checkpoint('checkpoint.pt')
            last_checkpoint_mtime = mtime


@app.route('/api/start', methods=['POST'])
def start_game():
    global current_state, human_player
    check_and_reload_model()

    seed = random.randint(0, 1_000_000)
    current_state = poker_cpp.RoyalState()
    current_state.reset(seed)
    human_player  = random.choice([0, 1])
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
