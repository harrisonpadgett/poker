"""
server.py — Flask + SocketIO server for Leduc Hold'em CFR visualizer.

Training loop:
  Runs CFR iterations in the background, emitting state after each batch.
  Every EMIT_EVERY iterations: emit strategy snapshot + exploitability sample.

The UI shows:
  - Current game hand being played (one sample game per batch)
  - Strategy table for all information sets
  - Exploitability over time (convergence to Nash)
  - Regret values and average strategy probabilities
"""

import threading
import time
import numpy as np
from flask import Flask, render_template
from flask_socketio import SocketIO

from poker_env import (LeducState, FOLD, CALL, RAISE,
                       ACTION_NAMES, card_str, N_CARDS)
from cfr import CFRTrainer

import os
_HERE = os.path.dirname(os.path.abspath(__file__))

app = Flask(__name__,
            template_folder=os.path.join(_HERE, 'templates'),
            static_folder=os.path.join(_HERE, 'static'))
app.config['SECRET_KEY'] = 'poker-cfr'
socketio = SocketIO(app, cors_allowed_origins='*', async_mode='threading')

# ---------------------------------------------------------------------------
# Training config
# ---------------------------------------------------------------------------
EMIT_EVERY       = 1      # emit after every batch
EXPLOIT_EVERY    = 10     # compute exploitability every N batches (66ms each)
EXPLOIT_SAMPLES  = 200    # unused

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
state = {
    'running':        False,
    'paused':         False,
    'speed':          0.05,   # seconds between batches
    'batch_size':     5,      # CFR iterations per batch
    'iterations':     0,
    'info_sets':      0,
    'avg_utility':    0.0,
    'exploitability': [],
    'utility_history':[],
    'sample_game':    None,   # last sample game for board display
}
state_lock  = threading.Lock()
trainer_ref = {'trainer': None}


# ---------------------------------------------------------------------------
# Sample game generator — plays one hand using current average strategy
# ---------------------------------------------------------------------------
def play_sample_game(trainer):
    """
    Play one complete game using the current average strategy.
    Returns list of (state_snapshot, action, probs) for visualization.
    """
    game   = LeducState().reset()
    frames = []

    while not game.done:
        player = game.to_act
        legal  = game.legal_actions()
        probs  = trainer.get_average_strategy(game.info_set(player), legal)

        # Sample action from average strategy
        prob_list = [float(probs[a]) for a in range(3)]
        if sum(prob_list) > 0:
            action = int(np.random.choice(3, p=prob_list /
                                          np.array(prob_list).sum()))
            if action not in legal:
                action = legal[0]
        else:
            action = legal[0]

        frames.append({
            'player':     player,
            'to_act':     player,
            'private':    [card_str(game.private[p]) for p in range(2)],
            'community':  card_str(game.community) if game.round==1 else None,
            'pot':        game.pot,
            'stacks':     list(game.stacks),
            'bets':       list(game.bets),
            'round':      game.round,
            'action':     action,
            'action_name':ACTION_NAMES[action],
            'probs':      prob_list,
            'info_set':   game.info_set(player),
            'legal':      legal,
            'history':    list(game.history),
        })

        game.apply_action(action)

    return frames, game


# ---------------------------------------------------------------------------
# Strategy snapshot — key info sets for display
# ---------------------------------------------------------------------------
def get_strategy_snapshot(trainer):
    """
    Return strategy for all visited information sets, sorted by round then card.
    Pulls directly from trainer.regret_sum so nothing is missed.
    """
    snapshot = []

    for info in trainer.regret_sum:
        parts = info.split('|')
        if len(parts) != 3:
            continue
        priv_str, comm_str, hist = parts

        # Determine round from community card
        round_n = 0 if comm_str == 'none' else 1

        legal = [FOLD, CALL, RAISE]
        probs = trainer.get_average_strategy(info, legal)

        snapshot.append({
            'info_set':  info,
            'card':      priv_str,
            'community': comm_str,
            'round':     round_n,
            'history':   hist,
            'fold':      round(float(probs[FOLD]),  3),
            'call':      round(float(probs[CALL]),  3),
            'raise':     round(float(probs[RAISE]), 3),
            'regrets':   [round(float(trainer.regret_sum[info][a]), 2)
                          for a in range(3)],
        })

    # Sort: preflop before flop, then by card, then by history length
    return sorted(snapshot,
                  key=lambda x: (x['round'], x['card'], len(x['history']), x['history']))


# ---------------------------------------------------------------------------
# Full game tree builder — all branches with probabilities
# ---------------------------------------------------------------------------
def build_game_tree(trainer, state, node_id=0, parent_id=None, edge_action=None,
                    edge_prob=None, depth=0, max_depth=6):
    """
    Recursively build the full game tree for one dealt hand.
    Returns a dict with nodes and edges for the SVG renderer.
    """
    player = state.to_act
    legal  = state.legal_actions()
    info   = state.info_set(player)
    probs  = trainer.get_average_strategy(info, legal)

    # Check terminal BEFORE calling legal_actions (round may be > 1)
    if state.done or depth >= max_depth:
        node = {
            'id': node_id, 'parent_id': parent_id,
            'edge_action': edge_action, 'edge_prob': edge_prob,
            'player': state.to_act, 'card': None, 'info_set': None,
            'probs': [0,0,0], 'legal': [], 'round': state.round,
            'pot': state.pot, 'done': True,
            'winner': state.winner, 'depth': depth, 'children': [],
        }
        return node, node_id

    node = {
        'id':          node_id,
        'parent_id':   parent_id,
        'edge_action': edge_action,
        'edge_prob':   edge_prob,
        'player':      player,
        'card':        state.private[player],
        'info_set':    info,
        'probs':       [round(float(p), 3) for p in probs],
        'legal':       legal,
        'round':       state.round,
        'pot':         state.pot,
        'done':        False,
        'winner':      None,
        'depth':       depth,
        'children':    [],
    }

    next_id = node_id + 1
    for a in legal:
        prob = float(probs[a])
        ns   = state.copy()
        ns.apply_action(a)
        child, next_id = build_game_tree(
            trainer, ns, next_id, node_id, a, prob, depth + 1, max_depth)
        node['children'].append(child)

    return node, next_id


def flatten_tree(root):
    """Flatten tree into nodes/edges lists for JSON serialization."""
    nodes = []
    edges = []

    def walk(node):
        nodes.append({k: v for k, v in node.items() if k != 'children'})
        for child in node['children']:
            edges.append({
                'from':   node['id'],
                'to':     child['id'],
                'action': child['edge_action'],
                'prob':   child['edge_prob'],
            })
            walk(child)

    walk(root)
    return nodes, edges


# ---------------------------------------------------------------------------
# Training loop
# ---------------------------------------------------------------------------
def training_loop():
    trainer = CFRTrainer()
    trainer_ref['trainer'] = trainer
    batch_n = 0

    # Run CFR training in a dedicated thread so it never blocks emits
    train_result = {'utility': 0.0, 'ready': False}
    train_lock   = threading.Lock()

    def run_train_batch():
        while True:
            with state_lock:
                if not state['running'] or state['paused']:
                    time.sleep(0.05)
                    continue
                current_batch_size = state['batch_size']
            u = trainer.train(current_batch_size)
            with train_lock:
                train_result['utility'] = u
                train_result['ready']   = True
            time.sleep(0.01)  # yield

    train_thread = threading.Thread(target=run_train_batch, daemon=True)
    train_thread.start()

    while True:
        with state_lock:
            if not state['running']:
                time.sleep(0.1)
                continue
            speed  = state['speed']
            paused = state['paused']

        if paused:
            time.sleep(0.1)
            continue

        # Wait for a training batch to complete
        ready = False
        for _ in range(200):  # max 2s wait
            with train_lock:
                ready = train_result['ready']
            if ready: break
            time.sleep(0.01)

        with train_lock:
            utility             = train_result['utility']
            train_result['ready'] = False

        batch_n += 1

        with state_lock:
            state['iterations']   = trainer.iterations
            state['info_sets']    = trainer.n_info_sets()
            state['avg_utility']  = round(float(trainer.total_utility /
                                                trainer.iterations), 4)
            state['utility_history'].append(round(utility, 4))

        # Play a sample game AND build full tree for that deal
        frames, final_game = play_sample_game(trainer)

        # Build full branching tree every 3 batches (not every batch)
        if batch_n % 3 == 0:
            tree_state = LeducState()
            tree_state.reset()
            tree_state.private   = list(final_game.private)
            tree_state.community = final_game.community
            tree_root, _ = build_game_tree(trainer, tree_state, max_depth=2)
            tree_nodes, tree_edges = flatten_tree(tree_root)
        else:
            tree_nodes, tree_edges = None, None

        # Compute exploitability periodically
        exploitability = None
        if batch_n % EXPLOIT_EVERY == 0:
            exploitability = round(
                float(trainer.compute_exploitability(EXPLOIT_SAMPLES)), 4)
            with state_lock:
                state['exploitability'].append(exploitability)

        # Key strategies (6 preflop rows) — sent every batch, fast
        key_infos = ['Jc|none|','Qc|none|','Kc|none|','Js|none|','Qs|none|','Ks|none|']
        key_strategies = []
        for info in key_infos:
            if info in trainer.regret_sum:
                probs = trainer.get_average_strategy(info, [FOLD, CALL, RAISE])
                key_strategies.append({
                    'info_set': info,
                    'fold':  round(float(probs[FOLD]),  3),
                    'call':  round(float(probs[CALL]),  3),
                    'raise': round(float(probs[RAISE]), 3),
                    'regrets': [round(float(trainer.regret_sum[info][a]),2)
                                for a in range(3)],
                })

        # Full snapshot only every 20 batches — 936 rows is expensive
        snapshot = get_strategy_snapshot(trainer) if batch_n % 20 == 0 else None

        # Emit everything
        socketio.emit('training_update', {
            'iterations':      trainer.iterations,
            'info_sets':       trainer.n_info_sets(),
            'avg_utility':     round(float(trainer.total_utility /
                                           trainer.iterations), 4),
            'exploitability':  exploitability,
            'exploit_history': state['exploitability'][-50:],
            'utility_history': state['utility_history'][-100:],
            'sample_frames':   frames,
            'final_winner':    final_game.winner,
            'final_payoff':    [final_game.payoff(0), final_game.payoff(1)],
            'strategy_snapshot': snapshot,   # full 936-row snapshot (throttled)
            'key_strategies':    key_strategies,  # 6 preflop rows (every batch)
            'tree_nodes':      tree_nodes,
            'tree_edges':      tree_edges,
            'tree_sampled':    [f['action'] for f in frames],  # path taken
        })

        time.sleep(speed)


# ---------------------------------------------------------------------------
# Socket events
# ---------------------------------------------------------------------------
@socketio.on('start_training')
def handle_start():
    with state_lock:
        state['running'] = True
        state['paused']  = False

@socketio.on('pause_training')
def handle_pause():
    with state_lock:
        state['paused'] = True

@socketio.on('resume_training')
def handle_resume():
    with state_lock:
        state['paused'] = False

@socketio.on('set_speed')
def handle_speed(data):
    with state_lock:
        delay = float(data.get('delay', 0.05))
        state['speed'] = delay
        # Drastically increase iterations per batch when slider is maxed
        if delay < 0.01:
            state['batch_size'] = 100
        else:
            state['batch_size'] = 5

@socketio.on('reset_training')
def handle_reset():
    with state_lock:
        state['running']          = False
        state['paused']           = False
        state['iterations']       = 0
        state['info_sets']        = 0
        state['avg_utility']      = 0.0
        state['exploitability']   = []
        state['utility_history']  = []
    time.sleep(0.2)
    threading.Thread(target=training_loop, daemon=True).start()
    socketio.emit('training_reset')

@socketio.on('query_strategy')
def handle_query(data):
    """Return full strategy for a specific info set."""
    trainer = trainer_ref.get('trainer')
    if trainer is None:
        return
    info  = data.get('info_set', '')
    legal = [FOLD, CALL, RAISE]
    probs = trainer.get_average_strategy(info, legal)
    regrets = [float(trainer.regret_sum[info][a]) for a in range(3)]
    socketio.emit('strategy_detail', {
        'info_set': info,
        'probs':    [round(float(p), 4) for p in probs],
        'regrets':  [round(r, 4) for r in regrets],
    })


# ---------------------------------------------------------------------------
# Routes
# ---------------------------------------------------------------------------
@app.route('/')
def index():
    return render_template('index.html')


# ---------------------------------------------------------------------------
# Entry point
# ---------------------------------------------------------------------------
if __name__ == '__main__':
    t = threading.Thread(target=training_loop, daemon=True)
    t.start()
    print("\n  Poker CFR  →  http://localhost:5000\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)