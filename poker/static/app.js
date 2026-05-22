/* app.js — Poker CFR Dashboard */

const socket = io();
let paused = false;
let isResetting = false;
let tableBuilt = false;
let tableRowMap = {};
let lastFrames = [];
let lastWinner = null;
let lastPayoffs = null;
let frameTimer = null;
let frameIdx = 0;
let gameAnimDone = false;

// ── Connection ──────────────────────────────────────────────────────────────
socket.on('connect', () => {
    document.getElementById('conn-badge').textContent = 'Connected';
    document.getElementById('conn-badge').classList.add('connected');
});
socket.on('disconnect', () => {
    document.getElementById('conn-badge').textContent = 'Disconnected';
    document.getElementById('conn-badge').classList.remove('connected');
    setPhase('disconnected');
});

// ── Training update ──────────────────────────────────────────────────────────
socket.on('training_update', d => {
    if (isResetting) return;
    setPhase('running');
    updateStats(d);
    updateCharts(d);
    document.getElementById('iter-num').textContent = d.iterations.toLocaleString();
    document.getElementById('info-sets-num').textContent = d.info_sets;

    // Key strategies (6 rows) — update every batch
    if (d.key_strategies) updateKeyStrategies(d.key_strategies);

    // Full strategy table — only when server sends it (every 20 batches)
    if (d.strategy_snapshot) updateStrategyTable(d.strategy_snapshot);

    // Game animation — only start if not already running
    if (!frameTimer && d.sample_frames && d.sample_frames.length > 0) {
        startGameAnimation(d.sample_frames, d.final_winner, d.final_payoff);
    }

    // Game tree — only when server sends it (every 3 batches)
    if (d.tree_nodes && d.tree_nodes.length > 0) {
        renderGameTree(d.tree_nodes, d.tree_edges, d.tree_sampled, 0);
    }
});

socket.on('training_reset', () => { isResetting = false; });
socket.on('strategy_detail', d => { renderRegretDetail(d); });

// ── Phase indicator ──────────────────────────────────────────────────────────
function setPhase(phase) {
    const dot = document.getElementById('phase-dot');
    const label = document.getElementById('phase-label');
    dot.className = 'phase-dot';
    const phases = {
        idle: ['', 'Idle — press Start to begin'],
        running: ['active', 'Training — iterating CFR…'],
        paused: ['paused-dot', 'Paused'],
        disconnected: ['', 'Disconnected from server'],
    };
    const [cls, text] = phases[phase] || ['', phase];
    if (cls) dot.classList.add(cls);
    label.textContent = text;
}

// ── Stats ────────────────────────────────────────────────────────────────────
function updateStats(d) {
    document.getElementById('stat-iter').textContent = d.iterations.toLocaleString();
    document.getElementById('stat-isets').textContent = d.info_sets;
    document.getElementById('stat-util').textContent = d.avg_utility !== undefined
        ? d.avg_utility.toFixed(3) : '—';
    if (d.exploitability != null) {
        const e = d.exploitability;
        const el = document.getElementById('stat-exploit');
        el.textContent = e.toFixed(3);
        el.style.color = e < 0.3 ? 'var(--green)' : e < 1.5 ? 'var(--amber)' : 'var(--red)';
    }
}

// ── Card rendering ───────────────────────────────────────────────────────────
function makeCard(cardStr, opts = {}) {
    const el = document.createElement('div');
    el.className = 'playing-card';
    if (opts.faceDown) { el.classList.add('face-down'); el.innerHTML = '<span class="rank">?</span>'; return el; }
    if (opts.active) el.classList.add('active');
    if (opts.winner) el.classList.add('winner');
    const rank = cardStr[0];
    const suit = cardStr[1];
    el.classList.add(suit === 's' ? 'spades' : 'clubs');
    const suitSym = { c: '♣', s: '♠', d: '♦', h: '♥' }[suit] || suit;
    el.innerHTML = `<span class="rank">${rank}</span><span class="suit">${suitSym}</span>`;
    return el;
}

// ── Game animation ───────────────────────────────────────────────────────────
const FRAME_MS = 900;  // ms between frames — slow enough to read

function stopAnim() {
    clearInterval(frameTimer);
    frameTimer = null;
}

function startGameAnimation(frames, winner, payoffs) {
    if (!frames || frames.length === 0) return;
    lastFrames = frames;
    lastWinner = winner;
    lastPayoffs = payoffs;
    frameIdx = 0;
    gameAnimDone = false;
    stopAnim();
    showGameFrame(frames[0], false);
    frameTimer = setInterval(() => {
        if (paused) { stopAnim(); return; }
        frameIdx++;
        if (frameIdx >= frames.length) {
            stopAnim();
            gameAnimDone = true;
            // Reveal P1's cards on final frame
            showGameFrame(frames[frames.length - 1], true);
            showGameResult(winner, payoffs);
            return;
        }
        showGameFrame(frames[frameIdx], false);
    }, FRAME_MS);
}

function showGameFrame(frame, isFinal) {
    const isP0Acting = frame.to_act === 0;
    const isP1Acting = frame.to_act === 1;

    // Player labels — highlight active player
    document.getElementById('p0-label').className = 'player-label' + (isP0Acting ? ' active' : '');
    document.getElementById('p1-label').className = 'player-label' + (isP1Acting ? ' active' : '');

    // P0 cards — always visible
    const p0c = document.getElementById('p0-cards');
    p0c.innerHTML = '';
    p0c.appendChild(makeCard(frame.private[0], { active: isP0Acting && !isFinal, winner: isFinal && lastWinner === 0 }));

    // P1 cards — face down until final
    const p1c = document.getElementById('p1-cards');
    p1c.innerHTML = '';
    p1c.appendChild(makeCard(frame.private[1], { faceDown: !isFinal, active: isP1Acting && !isFinal, winner: isFinal && lastWinner === 1 }));

    // Community
    const comm = document.getElementById('community-area');
    if (frame.community) {
        comm.innerHTML = '';
        comm.appendChild(makeCard(frame.community));
    } else {
        comm.innerHTML = '<span class="community-placeholder">Flop card revealed after preflop betting</span>';
    }

    // Pot + round
    document.getElementById('pot-num').textContent = frame.pot;
    document.getElementById('round-badge').textContent = frame.round === 0 ? 'Preflop' : 'Flop';

    // Stacks
    document.getElementById('p0-stack').textContent = `${frame.stacks[0]} chips`;
    document.getElementById('p1-stack').textContent = `${frame.stacks[1]} chips`;

    // Action badges
    const p0a = document.getElementById('p0-action');
    const p1a = document.getElementById('p1-action');
    p0a.textContent = ''; p0a.className = 'action-badge';
    p1a.textContent = ''; p1a.className = 'action-badge';
    if (frame.action !== undefined) {
        const cls = ['act-fold', 'act-call', 'act-raise'][frame.action];
        const name = ['Fold', 'Call', 'Raise'][frame.action];
        if (isP0Acting) { p0a.textContent = name; p0a.className = 'action-badge ' + cls; }
        else { p1a.textContent = name; p1a.className = 'action-badge ' + cls; }
    }

    // History trail
    const hw = document.getElementById('history-wrap');
    hw.innerHTML = '';
    const histCls = ['hist-fold', 'hist-call', 'hist-raise'];
    const histNames = ['Fold', 'Call', 'Raise'];
    frame.history.forEach((a, i) => {
        const chip = document.createElement('span');
        chip.className = 'hist-chip ' + histCls[a];
        chip.textContent = `P${i % 2} ${histNames[a]}`;
        hw.appendChild(chip);
    });

    // Clear result (mid-animation)
    if (!isFinal) {
        document.getElementById('game-result').textContent = '';
        document.getElementById('game-result').style.color = '';
    }

    // Explanation
    renderHandExplanation(frame, isFinal);

    // Tree is rendered from training_update with full branch data
}

function showGameResult(winner, payoffs) {
    const el = document.getElementById('game-result');
    if (winner === -1) {
        el.textContent = '🤝 Tie — pot split';
        el.style.color = 'var(--amber)';
    } else if (winner >= 0 && payoffs) {
        const gain = payoffs[winner];
        el.innerHTML = `P${winner} wins <span style="color:var(--green)">+${gain}</span> chips`;
        el.style.color = 'var(--text)';
    }
}

// ── Hand explanation ─────────────────────────────────────────────────────────
function renderHandExplanation(frame, isFinal) {
    const el = document.getElementById('cfr-explanation');
    const names = ['Fold', 'Call', 'Raise'];
    const clss = ['fold-c', 'call-c', 'raise-c'];
    const probs = frame.probs;
    const action = frame.action;
    const player = frame.to_act;
    const card = frame.private[player];

    let html = `<b>P${player}</b> holds <b>${card}</b> · info set: <span class="info-set-key">${frame.info_set}</span><br>`;

    // Prob bar
    const barParts = [0, 1, 2].map(a => {
        const pct = Math.round(probs[a] * 100);
        const w = Math.max(1, pct);
        return `<div class="prob-seg prob-${names[a].toLowerCase()}" style="width:${w}%">${pct > 9 ? pct + '%' : ''}</div>`;
    }).join('');
    html += `<div class="prob-bar-wrap" style="margin:6px 0">${barParts}</div>`;

    const chosen = names[action];
    const chosenCls = clss[action];
    const pct = Math.round(probs[action] * 100);
    html += `CFR sampled <b class="${chosenCls}">${chosen}</b> (${pct}% probability). `;

    const topA = frame.legal.reduce((b, a) => probs[a] > probs[b] ? a : b, frame.legal[0]);
    if (action !== topA) {
        html += `Top choice was <b class="${clss[topA]}">${names[topA]}</b> (${Math.round(probs[topA] * 100)}%) — mixed strategy maintains unpredictability.`;
    } else {
        html += `This was the highest-probability action.`;
    }

    if (isFinal && lastWinner !== null) {
        const result = lastWinner === -1 ? '🤝 Tied' : `P${lastWinner} won`;
        html += `<br><b>${result}</b>`;
    }

    el.innerHTML = html;
}

// ── Game tree visualization ──────────────────────────────────────────────────
// Proper tree layout using Reingold-Tilford approach:
//   1. Assign each leaf a unique y-slot
//   2. Interior nodes get the mean y of their children
//   3. x is simply depth * hGap
// This guarantees no overlap and fills the space evenly.

let treeNodesData = [];

function renderGameTree(nodes, edges, sampledPath) {
    const svg = document.getElementById('game-tree-svg');
    const wrap = document.getElementById('game-tree-wrap');
    if (!nodes || nodes.length === 0) return;

    const PCOLS = ['#58a6ff', '#bc8cff'];          // P0=blue, P1=purple
    const ECOLS = ['#f85149', '#3fb950', '#58a6ff']; // fold, call, raise
    const ENAMES = ['Fold', 'Call', 'Raise'];

    // ── Build adjacency ────────────────────────────────────────────
    const childEdges = {};  // parentId → [edge, ...]
    edges.forEach(e => {
        if (!childEdges[e.from]) childEdges[e.from] = [];
        childEdges[e.from].push(e);
    });

    const root = nodes.find(n => n.depth === 0);
    if (!root) return;

    // ── Count leaves & depth ──────────────────────────────────────
    function countLeaves(id) {
        const kids = childEdges[id];
        if (!kids || kids.length === 0) return 1;
        return kids.reduce((s, e) => s + countLeaves(e.to), 0);
    }
    const totalLeaves = countLeaves(root.id);
    const maxDepth = Math.max(...nodes.map(n => n.depth));

    // ── Fixed logical spacing ─────────────────────────────────────
    // max_depth=2 → at most 9 leaves, so these fit a 420px container cleanly
    const hGap  = 110;
    const vSlot = 46;
    const nodeR = 16;

    // ── Assign positions in logical space ────────────────────────
    const posMap = {};
    let leafSlot = 0;

    function assignY(id, depth) {
        const kids = childEdges[id];
        if (!kids || kids.length === 0) {
            posMap[id] = {
                x: depth * hGap + nodeR + 8,
                y: leafSlot * vSlot + nodeR + 8
            };
            leafSlot++;
            return;
        }
        kids.forEach(e => assignY(e.to, depth + 1));
        const childYs = kids.map(e => posMap[e.to].y);
        posMap[id] = {
            x: depth * hGap + nodeR + 8,
            y: (Math.min(...childYs) + Math.max(...childYs)) / 2
        };
    }
    assignY(root.id, 0);

    // ── Tight viewBox from actual node bounds ─────────────────────
    // Diamond terminals extend nodeR*√2 ≈ nodeR*1.42 from center after 45° rotation
    const margin = Math.ceil(nodeR * 1.5) + 8;
    const allX = Object.values(posMap).map(p => p.x);
    const allY = Object.values(posMap).map(p => p.y);
    const vbX = Math.min(...allX) - margin;
    const vbY = Math.min(...allY) - margin;
    const vbW = Math.max(...allX) - vbX + margin;
    const vbH = Math.max(...allY) - vbY + margin;

    svg.setAttribute('viewBox', `${vbX} ${vbY} ${vbW} ${vbH}`);
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // ── Sampled path ──────────────────────────────────────────────
    const sampledIds = new Set([root.id]);
    let cur = root.id;
    (sampledPath || []).forEach(action => {
        const e = (childEdges[cur] || []).find(e2 => e2.action === action);
        if (e) { sampledIds.add(e.to); cur = e.to; }
    });

    // ── Draw ──────────────────────────────────────────────────────
    let html = `<defs>
    <marker id="arrt" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
      <path d="M0,0 L5,2.5 L0,5 Z" fill="#484f58"/>
    </marker>
    <marker id="arrw" markerWidth="5" markerHeight="5" refX="4" refY="2.5" orient="auto">
      <path d="M0,0 L5,2.5 L0,5 Z" fill="rgba(255,255,255,0.8)"/>
    </marker>
  </defs>`;

    // Edges first
    edges.forEach(e => {
        const fp = posMap[e.from];
        const tp = posMap[e.to];
        if (!fp || !tp) return;

        const onPath = sampledIds.has(e.from) && sampledIds.has(e.to);
        const col = onPath ? 'rgba(255,255,255,0.9)' : (ECOLS[e.action] || '#484f58');
        const thick = Math.max(1.5, e.prob * 8);
        const opacity = onPath ? 1.0 : Math.max(0.15, e.prob * 0.85);
        const marker = onPath ? 'url(#arrw)' : 'url(#arrt)';

        const x1 = fp.x + nodeR, y1 = fp.y;
        const x2 = tp.x - nodeR, y2 = tp.y;
        const cx = (x1 + x2) / 2;

        html += `<path d="M${x1},${y1} C${cx},${y1} ${cx},${y2} ${x2},${y2}"
      fill="none" stroke="${col}" stroke-width="${thick}"
      opacity="${opacity}" marker-end="${marker}"/>`;

        // Label at t≈0.65 along bezier — edges have diverged here, no overlap
        const pct = Math.round(e.prob * 100);
        if (pct >= 5 || onPath) {
            // Cubic bezier at t=0.65: x≈0.35*x1+0.65*x2, y=(1-t)²(1+2t)*y1+t²(3-2t)*y2
            const lx = 0.35 * x1 + 0.65 * x2;
            const ly = 0.2817 * y1 + 0.7183 * y2 - 7;
            const label = onPath ? `${ENAMES[e.action]} ${pct}%` : ENAMES[e.action];
            const lOpacity = onPath ? 1.0 : Math.min(0.65, opacity + 0.15);
            html += `<text x="${lx}" y="${ly}" text-anchor="middle"
        font-size="9" fill="${col}" opacity="${lOpacity}"
        >${label}</text>`;
        }
    });

    // Nodes on top
    nodes.forEach(n => {
        const p = posMap[n.id];
        if (!p) return;

        const onPath = sampledIds.has(n.id);
        const nodeColor = n.done ? '#484f58' : (PCOLS[n.player] || '#8b949e');
        const strokeC = onPath ? (n.done ? '#aaa' : '#fff') : (n.done ? '#333' : nodeColor);
        const strokeW = onPath ? 2.5 : 1.5;
        const fillC = onPath ? '#1e2840' : '#0d1117';

        if (n.done) {
            // Terminal: diamond
            const s = nodeR - 4;
            html += `<rect x="${p.x - s}" y="${p.y - s}" width="${s * 2}" height="${s * 2}"
        rx="2" fill="${fillC}" stroke="${strokeC}" stroke-width="${strokeW}"
        transform="rotate(45,${p.x},${p.y})" style="cursor:pointer"
        onclick="treeNodeClick(${n.id})"/>`;
            const wl = n.winner === -1 ? 'T' : n.winner !== null ? `P${n.winner}` : '—';
            const wc = n.winner === 0 ? PCOLS[0] : n.winner === 1 ? PCOLS[1] : '#8b949e';
            html += `<text x="${p.x}" y="${p.y + 4}" text-anchor="middle"
        font-size="10" font-weight="700" fill="${wc}"
        style="pointer-events:none">${wl}</text>`;
        } else {
            // Decision node: circle
            html += `<circle cx="${p.x}" cy="${p.y}" r="${nodeR}"
        fill="${fillC}" stroke="${strokeC}" stroke-width="${strokeW}"
        style="cursor:pointer" onclick="treeNodeClick(${n.id})"/>`;

            // Card inside
            const card = typeof n.card === 'number'
                ? ['Jc', 'Qc', 'Kc', 'Js', 'Qs', 'Ks'][n.card] : (n.card || '?');
            html += `<text x="${p.x}" y="${p.y - 1}" text-anchor="middle"
        font-size="12" font-weight="700" fill="${nodeColor}"
        style="pointer-events:none">${card}</text>
      <text x="${p.x}" y="${p.y + 11}" text-anchor="middle"
        font-size="8" fill="#8b949e"
        style="pointer-events:none">P${n.player}</text>`;

            // Mini prob bar below
            if (n.probs && n.legal && n.legal.length > 0) {
                const bw = nodeR * 2, bx = p.x - nodeR, by = p.y + nodeR + 4;
                let cx2 = bx;
                n.legal.forEach(a => {
                    const w = n.probs[a] * bw;
                    if (w > 0.5) {
                        html += `<rect x="${cx2}" y="${by}" width="${w}" height="4"
              fill="${ECOLS[a]}" rx="1" opacity="${onPath ? 0.9 : 0.45}"/>`;
                        cx2 += w;
                    }
                });
            }
        }
    });

    svg.innerHTML = html;
    treeNodesData = nodes;

    // Scroll so root is visible
    wrap.scrollLeft = 0;
}


// Node click handler — show detail for tree node
function treeNodeClick(id) {
    const detail = document.getElementById('tree-node-detail');
    detail.style.display = 'block';
    const node = treeNodesData.find(n => n.id === id);
    if (!node) { detail.textContent = `Node ${id}`; return; }
    if (node.done) {
        const w = node.winner === -1 ? 'Tie' : node.winner !== null ? `P${node.winner} wins` : 'Terminal';
        detail.innerHTML = `<b>Terminal</b> — ${w} · Pot: ${node.pot}`;
        return;
    }
    const names = ['Fold', 'Call', 'Raise'];
    const probs = node.legal.map(a =>
        `<span style="color:${{ 0: 'var(--fold-c)', 1: 'var(--call-c)', 2: 'var(--raise-c)' }[a]}">${names[a]}: ${Math.round(node.probs[a] * 100)}%</span>`
    ).join(' · ');
    const card = typeof node.card === 'number'
        ? ['Jc', 'Qc', 'Kc', 'Js', 'Qs', 'Ks'][node.card] : node.card;
    detail.innerHTML = `<b>P${node.player}</b> holds <b>${card}</b> · ${node.info_set || ''} · ${probs}`;
}

// ── Strategy table ───────────────────────────────────────────────────────────
function updateStrategyTable(snapshot) {
    if (!snapshot || snapshot.length === 0) return;
    const tbody = document.getElementById('strategy-tbody');

    if (!tableBuilt || tbody.children.length !== snapshot.length) {
        tbody.innerHTML = '';
        tableRowMap = {};
        tableBuilt = true;
        snapshot.forEach(row => {
            const tr = document.createElement('tr');
            tr.style.cursor = 'pointer';
            tr.dataset.infoSet = row.info_set;
            tr.innerHTML = `
        <td class="info-set-key"></td>
        <td><span style="font-size:10px;color:var(--text3)"></span></td>
        <td></td><td></td><td></td><td></td>`;
            tr.onclick = () => {
                document.querySelectorAll('#strategy-tbody tr').forEach(r =>
                    r.style.background = r === tr ? 'var(--surface2)' : '');
                socket.emit('query_strategy', { info_set: row.info_set });
                renderRegretBars(row.regrets, row.info_set);
            };
            tbody.appendChild(tr);
            tableRowMap[row.info_set] = tr;
        });
    }

    snapshot.forEach(row => {
        const tr = tableRowMap[row.info_set];
        if (!tr) return;
        const fold = Math.round(row.fold * 100);
        const call = Math.round(row.call * 100);
        const raise = Math.round(row.raise * 100);
        const rnd = row.round === 0 ? 'Pre' : 'Flop';
        const maxP = Math.max(fold, call, raise);
        const [nc, nl] = maxP > 80 ? ['nash-good', '✓'] : maxP > 50 ? ['nash-med', '~'] : ['nash-bad', '?'];
        tr.cells[0].textContent = row.info_set;
        tr.cells[1].querySelector('span').textContent = rnd;
        tr.cells[2].innerHTML = `<span style="color:var(--fold-c)">${fold}%</span>`;
        tr.cells[3].innerHTML = `<span style="color:var(--call-c)">${call}%</span>`;
        tr.cells[4].innerHTML = `<span style="color:var(--raise-c)">${raise}%</span>`;
        tr.cells[5].innerHTML = `<span class="nash-badge ${nc}">${nl}</span>`;
    });
}

// ── Key strategies summary ───────────────────────────────────────────────────
function updateKeyStrategies(data) {
    if (!data || data.length === 0) return;
    const el = document.getElementById('key-strategies');
    // Accept either full snapshot or pre-filtered key_strategies array
    const preflop = data[0] && data[0].round !== undefined
        ? data.filter(r => r.round === 0 && r.history === '')
        : data;  // key_strategies is already filtered
    if (preflop.length === 0) return;

    const cards = ['Jc', 'Qc', 'Kc', 'Js', 'Qs', 'Ks'];
    let html = '';
    preflop.forEach(row => {
        if (!cards.includes(row.card)) return;
        const f = Math.round(row.fold * 100);
        const c = Math.round(row.call * 100);
        const r = Math.round(row.raise * 100);
        const barW = 140;
        html += `<div style="display:flex;align-items:center;gap:8px;margin-bottom:5px">
      <span style="width:28px;font-weight:600;color:var(--text)">${row.card}</span>
      <div style="flex:1;height:14px;display:flex;border-radius:3px;overflow:hidden">
        <div style="width:${f}%;background:var(--fold-c)" title="Fold ${f}%"></div>
        <div style="width:${c}%;background:var(--call-c)" title="Call ${c}%"></div>
        <div style="width:${r}%;background:var(--raise-c)" title="Raise ${r}%"></div>
      </div>
      <span style="font-size:10px;color:var(--text3);width:80px">
        <span style="color:var(--fold-c)">${f}%</span>/<span style="color:var(--call-c)">${c}%</span>/<span style="color:var(--raise-c)">${r}%</span>
      </span>
    </div>`;
    });
    if (html) el.innerHTML = `<div style="margin-bottom:4px;font-size:10px;color:var(--text3)">Card · Fold / Call / Raise</div>` + html;
}

// ── Regret bars ──────────────────────────────────────────────────────────────
function renderRegretBars(regrets, infoSet) {
    document.getElementById('regret-info-label').textContent = infoSet || '';
    const el = document.getElementById('regret-display');
    const names = ['Fold', 'Call', 'Raise'];
    const colors = ['var(--fold-c)', 'var(--call-c)', 'var(--raise-c)'];
    const maxAbs = Math.max(1, ...regrets.map(Math.abs));
    let html = '';
    regrets.forEach((r, i) => {
        const pct = Math.abs(r) / maxAbs * 100;
        const cls = r >= 0 ? 'regret-pos-fill' : 'regret-neg-fill';
        html += `<div class="regret-row">
      <span class="regret-label" style="color:${colors[i]}">${names[i]}</span>
      <div class="regret-track">
        <div class="regret-fill ${cls}" style="width:${Math.max(2, pct)}%"></div>
      </div>
      <span class="regret-val" style="color:${r >= 0 ? 'var(--green)' : 'var(--red)'}">
        ${r > 0 ? '+' : ''}${r.toFixed(1)}
      </span>
    </div>`;
    });
    html += `<div style="font-size:11px;color:var(--text3);margin-top:4px">
    Positive → take this action more. CFR strategy ∝ positive regrets.</div>`;
    el.innerHTML = html;
}

function renderRegretDetail(d) {
    renderRegretBars(d.regrets, d.info_set);
}

// ── Charts ───────────────────────────────────────────────────────────────────
const baseOpts = {
    responsive: true,
    maintainAspectRatio: false,
    animation: { duration: 0 },
    plugins: { legend: { display: false }, tooltip: { enabled: true } },
    scales: {
        x: { display: false },
        y: { grid: { color: '#21262d' }, ticks: { color: '#8b949e', font: { size: 10 } } }
    }
};

function mkChart(id, color) {
    const ctx = document.getElementById(id).getContext('2d');
    return new Chart(ctx, {
        type: 'line',
        data: {
            labels: [], datasets: [{
                data: [], borderColor: color,
                backgroundColor: color + '18', borderWidth: 2, pointRadius: 0,
                fill: true, tension: 0.3
            }]
        },
        options: { ...baseOpts }
    });
}

const chartExploit = mkChart('chart-exploit', '#f85149');
const chartUtility = mkChart('chart-utility', '#3fb950');

function updateCharts(d) {
    if (d.exploit_history?.length > 0) {
        chartExploit.data.labels = d.exploit_history.map((_, i) => i);
        chartExploit.data.datasets[0].data = d.exploit_history;
        chartExploit.update();
    }
    if (d.utility_history?.length > 0) {
        chartUtility.data.labels = d.utility_history.map((_, i) => i);
        chartUtility.data.datasets[0].data = d.utility_history;
        chartUtility.update();
    }
}

// ── Controls ─────────────────────────────────────────────────────────────────
function startTraining() {
    isResetting = false;
    socket.emit('start_training');
    document.getElementById('btn-start').disabled = true;
    document.getElementById('btn-pause').disabled = false;
    setPhase('running');
}

function togglePause() {
    paused = !paused;
    socket.emit(paused ? 'pause_training' : 'resume_training');
    document.getElementById('btn-pause').textContent = paused ? '▶ Resume' : '⏸ Pause';
    if (paused) {
        stopAnim();
        setPhase('paused');
        const el = document.getElementById('game-result');
        el.textContent = '⏸ Paused';
        el.style.color = 'var(--text3)';
    } else {
        setPhase('running');
        document.getElementById('game-result').textContent = '';
        // Resume animation if we have frames
        if (lastFrames.length > 0 && !gameAnimDone) {
            frameTimer = setInterval(() => {
                if (paused) { stopAnim(); return; }
                frameIdx++;
                if (frameIdx >= lastFrames.length) {
                    stopAnim(); gameAnimDone = true;
                    showGameFrame(lastFrames[lastFrames.length - 1], true);
                    showGameResult(lastWinner, lastPayoffs);
                    return;
                }
                showGameFrame(lastFrames[frameIdx], false);
            }, FRAME_MS);
        }
    }
}

function resetTraining() {
    isResetting = true;
    paused = false;
    stopAnim();
    setIdleUI();
    chartExploit.data.labels = []; chartExploit.data.datasets[0].data = []; chartExploit.update();
    chartUtility.data.labels = []; chartUtility.data.datasets[0].data = []; chartUtility.update();
    socket.emit('reset_training');
}

function setSpeed(val) {
    const delay = parseInt(val) / 1000;
    socket.emit('set_speed', { delay });
    const hint = document.getElementById('speed-hint');
    hint.textContent = delay < 0.05 ? 'max' : delay.toFixed(1) + 's';
}

// ── Idle UI reset ─────────────────────────────────────────────────────────────
function setIdleUI() {
    stopAnim();
    tableBuilt = false; tableRowMap = {};
    lastFrames = []; lastWinner = null; lastPayoffs = null;
    gameAnimDone = false; frameIdx = 0;

    setPhase('idle');

    document.getElementById('iter-num').textContent = '0';
    document.getElementById('info-sets-num').textContent = '0';
    document.getElementById('btn-reset').textContent = '↺ Reset';
    document.getElementById('btn-start').disabled = false;
    document.getElementById('btn-pause').disabled = true;
    document.getElementById('btn-pause').textContent = '⏸ Pause';

    ['stat-iter', 'stat-isets', 'stat-util', 'stat-exploit'].forEach(id => {
        const el = document.getElementById(id);
        el.textContent = (id === 'stat-iter' || id === 'stat-isets') ? '0' : '—';
        el.style.color = '';
    });

    // Game display
    document.getElementById('p0-cards').innerHTML = '';
    document.getElementById('p1-cards').innerHTML = '';
    document.getElementById('p0-action').textContent = '';
    document.getElementById('p0-action').className = 'action-badge';
    document.getElementById('p1-action').textContent = '';
    document.getElementById('p1-action').className = 'action-badge';
    document.getElementById('p0-stack').textContent = '';
    document.getElementById('p1-stack').textContent = '';
    document.getElementById('p0-label').className = 'player-label';
    document.getElementById('p1-label').className = 'player-label';
    document.getElementById('pot-num').textContent = '2';
    document.getElementById('round-badge').textContent = 'Preflop';
    document.getElementById('history-wrap').innerHTML = '';
    document.getElementById('community-area').innerHTML =
        '<span class="community-placeholder">Flop card revealed after preflop betting</span>';
    document.getElementById('game-result').textContent = '';
    document.getElementById('game-result').style.color = '';
    document.getElementById('tree-hand-label').textContent = '';
    document.getElementById('tree-node-detail').style.display = 'none';

    // Tree
    const svg = document.getElementById('game-tree-svg');
    svg.innerHTML = '';
    svg.setAttribute('viewBox', '0 0 400 300');
    svg.setAttribute('width', '100%');
    svg.setAttribute('height', '100%');
    svg.setAttribute('preserveAspectRatio', 'xMidYMid meet');

    // Text panels
    document.getElementById('cfr-explanation').innerHTML = 'Press <b>Start</b> to begin CFR training.';
    document.getElementById('strategy-tbody').innerHTML = '';
    document.getElementById('regret-display').innerHTML =
        '<div style="color:var(--text3);font-size:12px">Click any row in the strategy table.</div>';
    document.getElementById('regret-info-label').textContent = 'Click a row above';
    document.getElementById('key-strategies').innerHTML =
        '<div style="color:var(--text3)">Training not yet started.</div>';
}