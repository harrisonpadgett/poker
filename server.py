"""
server.py — Flask + SocketIO server for Leduc Hold'em CFR visualizer.

Architecture change from original:
  Train thread  → runs flat out, no coordination with emit loop
  Emit loop     → wakes every max(speed, 100ms), reads whatever trained so far

This decoupling means:
  - At max speed: ~1600 iter/s, ~160 iterations per UI update
  - No freezing: the two threads never block each other
  - UI always gets fresh data at a human-readable rate
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
EXPLOIT_EVERY   = 500    # compute exploitability every N iterations
MIN_EMIT_MS     = 100    # never emit faster than 10Hz regardless of speed

# ---------------------------------------------------------------------------
# Shared state
# ---------------------------------------------------------------------------
state = {
    'running':         False,
    'paused':          False,
    'speed':           0.05,   # seconds between emits (0 = max)
    'batch_size':      20,     # iterations per train() call
    'iterations':      0,
    'info_sets':       0,
    'avg_utility':     0.0,
    'exploitability':  [],
    'utility_history': [],
}
state_lock  = threading.Lock()
trainer_ref = {'trainer': None}


# ---------------------------------------------------------------------------
# Sample game — plays one hand with current average strategy
# ---------------------------------------------------------------------------
def play_sample_game(trainer):
    game   = LeducState().reset()
    frames = []

    while not game.done:
        player = game.to_act
        legal  = game.legal_actions()
        probs  = trainer.get_average_strategy(game.info_set(player), legal)

        prob_list = [float(probs[a]) for a in range(3)]
        total     = sum(prob_list)
        if total > 0:
            action = int(np.random.choice(3, p=np.array(prob_list) / total))
            if action not in legal:
                action = legal[0]
        else:
            action = legal[0]

        frames.append({
            'player':      player,
            'to_act':      player,
            'private':     [card_str(game.private[p]) for p in range(2)],
            'community':   card_str(game.community) if game.round == 1 else None,
            'pot':         game.pot,
            'stacks':      list(game.stacks),
            'bets':        list(game.bets),
            'round':       game.round,
            'action':      action,
            'action_name': ACTION_NAMES[action],
            'probs':       prob_list,
            'info_set':    game.info_set(player),
            'legal':       legal,
            'history':     list(game.history),
        })

        game.apply_action(action)

    return frames, game


# ---------------------------------------------------------------------------
# Strategy snapshot
# ---------------------------------------------------------------------------
def get_strategy_snapshot(trainer):
    snapshot = []
    for info in trainer.regret_sum:
        parts = info.split('|')
        if len(parts) == 4:
            _, priv_str, comm_str, hist = parts
        elif len(parts) == 3:
            priv_str, comm_str, hist = parts
        else:
            continue
        round_n = 0 if comm_str == 'none' else 1
        legal   = [FOLD, CALL, RAISE]
        probs   = trainer.get_average_strategy(info, legal)
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
    return sorted(snapshot,
                  key=lambda x: (x['round'], x['card'],
                                 len(x['history']), x['history']))


# ---------------------------------------------------------------------------
# Game tree builder
# ---------------------------------------------------------------------------
def build_game_tree(trainer, state, node_id=0, parent_id=None,
                    edge_action=None, edge_prob=None,
                    depth=0, max_depth=6):
    player = state.to_act
    legal  = state.legal_actions()
    info   = state.info_set(player)
    probs  = trainer.get_average_strategy(info, legal)

    if state.done or depth >= max_depth:
        return {
            'id': node_id, 'parent_id': parent_id,
            'edge_action': edge_action, 'edge_prob': edge_prob,
            'player': state.to_act, 'card': None, 'info_set': None,
            'probs': [0, 0, 0], 'legal': [], 'round': state.round,
            'pot': state.pot, 'done': True,
            'winner': state.winner, 'depth': depth, 'children': [],
        }, node_id

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
# Training loop  —  decoupled train thread + emit loop
# ---------------------------------------------------------------------------
def training_loop():
    trainer = CFRTrainer()
    trainer_ref['trainer'] = trainer

    # ── Train thread: runs flat out, writes to trainer directly ──────────
    def run_train():
        while True:
            with state_lock:
                if not state['running'] or state['paused']:
                    time.sleep(0.05)
                    continue
                batch = state['batch_size']

            trainer.train(batch)

            if 0 in (trainer.iterations % 10000, trainer.iterations % 10000 + 20, trainer.iterations % 10000 - 20):

                print(
                    "\nITER", trainer.iterations,
                    "\nmax regret:", np.max(np.abs(trainer._regret)),
                    "\nmean regret:", np.mean(np.abs(trainer._regret)),
                    "\nmax strat_sum:", np.max(np.abs(trainer._strat_sum)),
                    "\n"
                )

                for info in [
                    'P0|Jc|none|',
                    'P0|Qc|none|',
                    'P0|Kc|none|'
                ]:

                    probs = trainer.get_average_strategy(
                        info,
                        [FOLD, CALL, RAISE]
                    )

                    print(info, probs)

            with state_lock:
                state['iterations']  = trainer.iterations
                state['info_sets']   = trainer.n_info_sets()
                state['avg_utility'] = round(
                    float(trainer.total_utility / max(1, trainer.iterations)), 4)

            # Tiny yield so the emit loop and socket can breathe
            time.sleep(0.001)

    train_thread = threading.Thread(target=run_train, daemon=True)
    train_thread.start()

    # ── Emit loop: wakes on a fixed wall-clock interval ──────────────────
    batch_n          = 0
    last_exploit_iter = 0

    while True:
        with state_lock:
            if not state['running']:
                time.sleep(0.1)
                continue
            paused = state['paused']
            speed  = state['speed']

        if paused:
            time.sleep(0.1)
            continue

        batch_n += 1
        is_max_speed = speed < 0.01

        # Exploitability: only when enough new iterations have passed
        exploitability = None
        with state_lock:
            cur_iters = state['iterations']
        if cur_iters - last_exploit_iter >= EXPLOIT_EVERY:
            exploitability = round(float(trainer.compute_exploitability()), 4)


            with state_lock:
                state['exploitability'].append(exploitability)
                util = state['avg_utility']
                state['utility_history'].append(util)
            last_exploit_iter = cur_iters

        # Sample game + tree — skip most frames at max speed
        if not is_max_speed or batch_n % 5 == 0:
            frames, final_game = play_sample_game(trainer)
        else:
            frames, final_game = [], None

        if not is_max_speed and batch_n % 3 == 0 and final_game:
            tree_state          = LeducState()
            tree_state.reset()
            tree_state.private  = list(final_game.private)
            tree_state.community= final_game.community
            tree_root, _        = build_game_tree(trainer, tree_state, max_depth=2)
            tree_nodes, tree_edges = flatten_tree(tree_root)
        else:
            tree_nodes, tree_edges = None, None

        # Key preflop strategies — cheap, every emit
        key_infos = [
            'P0|Jc|none|',
            'P0|Qc|none|',
            'P0|Kc|none|',
            'P0|Js|none|',
            'P0|Qs|none|',
            'P0|Ks|none|'
        ]
        key_strategies = []
        regret_cache   = trainer.regret_sum   # single property call
        for info in key_infos:
            if info in regret_cache:
                probs = trainer.get_average_strategy(info, [FOLD, CALL, RAISE])
                key_strategies.append({
                    'info_set': info,
                    'fold':  round(float(probs[FOLD]),  3),
                    'call':  round(float(probs[CALL]),  3),
                    'raise': round(float(probs[RAISE]), 3),
                    'regrets': [round(float(regret_cache[info][a]), 2)
                                for a in range(3)],
                })

        # Full strategy snapshot — only every 20 emits (expensive)
        snapshot = get_strategy_snapshot(trainer) if batch_n % 20 == 0 else None

        with state_lock:
            cur_iters    = state['iterations']
            cur_isets    = state['info_sets']
            cur_utility  = state['avg_utility']
            exploit_hist = state['exploitability'][-50:]
            util_hist    = state['utility_history'][-100:]

        socketio.emit('training_update', {
            'iterations':        cur_iters,
            'info_sets':         cur_isets,
            'avg_utility':       cur_utility,
            'exploitability':    exploitability,
            'exploit_history':   exploit_hist,
            'utility_history':   util_hist,
            'sample_frames':     frames,
            'final_winner':      final_game.winner if final_game else None,
            'final_payoff':      ([final_game.payoff(0), final_game.payoff(1)]
                                  if final_game else None),
            'strategy_snapshot': snapshot,
            'key_strategies':    key_strategies,
            'tree_nodes':        tree_nodes,
            'tree_edges':        tree_edges,
            'tree_sampled':      [f['action'] for f in frames],
        })

        # Emit interval floor: never faster than MIN_EMIT_MS
        emit_delay = max(speed, MIN_EMIT_MS / 1000.0)
        time.sleep(emit_delay)


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
        # Batch size: small bursts so the GIL yields regularly
        if delay < 0.01:
            state['batch_size'] = 20     # flat-out: ~1600 iter/s
        elif delay < 0.1:
            state['batch_size'] = 10
        elif delay < 0.5:
            state['batch_size'] = 5
        else:
            state['batch_size'] = 2      # slow/step mode


@socketio.on('reset_training')
def handle_reset():
    with state_lock:
        state['running']         = False
        state['paused']          = False
        state['iterations']      = 0
        state['info_sets']       = 0
        state['avg_utility']     = 0.0
        state['exploitability']  = []
        state['utility_history'] = []
    time.sleep(0.2)
    threading.Thread(target=training_loop, daemon=True).start()
    socketio.emit('training_reset')


@socketio.on('query_strategy')
def handle_query(data):
    trainer = trainer_ref.get('trainer')
    if trainer is None:
        return
    info    = data.get('info_set', '')
    legal   = [FOLD, CALL, RAISE]
    probs   = trainer.get_average_strategy(info, legal)
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
    print("  Note: First Start click triggers Numba JIT compile (~3s).\n"
          "  Subsequent runs use cached binary — no recompilation.\n")
    socketio.run(app, host='0.0.0.0', port=5000, debug=False)