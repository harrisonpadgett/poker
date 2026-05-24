from flask import Flask, render_template, request, jsonify
import poker_cpp
from deep_cfr import DeepCFRTrainer, encode_state
import torch
import numpy as np
import random
import os

app = Flask(__name__, static_folder='static', template_folder='templates')

# Initialize AI
print("Loading Deep CFR AI...")
trainer = DeepCFRTrainer(adv_buffer_size=10, strat_buffer_size=10) # Buffers don't matter for inference
if os.path.exists('checkpoint.pt'):
    trainer.load_checkpoint('checkpoint.pt')
else:
    print("WARNING: checkpoint.pt not found! AI will play completely randomly.")

# Game State
current_state = None
human_player = 0

def get_state_dict():
    """Serializes the current RoyalState to a dictionary for the frontend."""
    global current_state, human_player
    if current_state is None:
        return {"error": "Game not started"}
        
    # Translate cards to strings
    community_strs = [poker_cpp.card_str(c) for c in current_state.visible_community()]
    private_strs = [poker_cpp.card_str(c) for c in current_state.private[human_player]]
    
    # Opponent player ID
    ai_player = 1 - human_player
    
    # AI Cards: only show if the game is done (Showdown)
    ai_strs = [poker_cpp.card_str(c) for c in current_state.private[ai_player]] if current_state.done else []
    
    # Calculate costs
    # To call, human must match AI's bet
    call_amount = current_state.bets[ai_player] - current_state.bets[human_player]
    call_amount = min(call_amount, current_state.stacks[human_player])
    
    # Bet size fixed for limit hold'em
    # round 0,1 -> 2 chips; round 2,3 -> 4 chips
    bet_size = 2 if current_state.round < 2 else 4
    
    return {
        "round": current_state.round,
        "pot": current_state.pot,
        "to_act": current_state.to_act,
        "human_player": human_player,
        "human_stack": current_state.stacks[human_player],
        "human_bet": current_state.bets[human_player],
        "ai_stack": current_state.stacks[ai_player],
        "ai_bet": current_state.bets[ai_player],
        "done": current_state.done,
        "winner": current_state.winner,
        "raises": current_state.raises,
        "community": community_strs,
        "private": private_strs,
        "ai_private": ai_strs,
        "call_amount": call_amount,
        "raise_amount": call_amount + bet_size,
        "legal_actions": current_state.legal_actions(),
        "history": current_state.history,
        "is_human_turn": current_state.to_act == human_player and not current_state.done
    }

def ai_play_turn():
    """Executes the AI's turn."""
    global current_state, human_player
    if current_state.done or current_state.to_act == human_player:
        return
        
    # Encode state and get average strategy
    t = encode_state(current_state, current_state.to_act)
    probs = trainer.get_average_strategy_from_tensor(t, current_state.legal_actions())
    
    # Sample action from probability distribution
    actions = current_state.legal_actions()
    p = np.array([probs[a] for a in actions], dtype=np.float64)
    if p.sum() == 0:
        p = np.ones_like(p) / len(p)
    else:
        p /= p.sum()
        
    action = np.random.choice(actions, p=p)
    print(f"AI plays action: {action} with probabilities: Fold={probs[0]:.2f}, Call={probs[1]:.2f}, Raise={probs[2]:.2f}")
    current_state.apply_action(action)

@app.route('/')
def index():
    return render_template('index.html')

# Track the last modified time of the checkpoint
last_checkpoint_mtime = 0

def check_and_reload_model():
    """Reloads the AI model if the checkpoint file has been updated by the training script."""
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
    
    # Auto-reload the AI if the training script saved a new checkpoint
    check_and_reload_model()
    
    seed = random.randint(0, 1000000)
    current_state = poker_cpp.RoyalState()
    current_state.reset(seed)
    
    # Human plays a random seat
    human_player = random.choice([0, 1])
    print(f"New game started. Human is Player {human_player}")
    
    # If AI acts first
    if current_state.to_act != human_player and not current_state.done:
        ai_play_turn()
        
    return jsonify(get_state_dict())

@app.route('/api/state', methods=['GET'])
def get_state():
    return jsonify(get_state_dict())

@app.route('/api/action', methods=['POST'])
def take_action():
    global current_state, human_player
    data = request.json
    action = data.get('action')
    
    if current_state is None or current_state.done:
        return jsonify({"error": "Game is over or not started"}), 400
        
    if current_state.to_act != human_player:
        return jsonify({"error": "Not your turn"}), 400
        
    if action not in current_state.legal_actions():
        return jsonify({"error": "Illegal action"}), 400
        
    # Human plays
    current_state.apply_action(action)
    
    # AI responds sequentially until human's turn or game over
    while not current_state.done and current_state.to_act != human_player:
        ai_play_turn()
        
    return jsonify(get_state_dict())

if __name__ == '__main__':
    app.run(debug=True, port=5001, use_reloader=False)
