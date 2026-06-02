/* ── Constants ─────────────────────────────────────────────────────────── */
const ACTION_NAMES   = ['Fold', 'Call', 'Raise-S', 'Raise-M', 'Raise-L'];
const ACTION_CLASSES = ['fold', 'call', 'raise-s', 'raise-m', 'raise-l'];
const ACTION_COLORS  = ['#ef4444', '#06b6d4', '#8b5cf6', '#6366f1', '#a855f7'];

const SUIT_SYMBOLS = { h: '♥', d: '♦', c: '♣', s: '♠' };

/* ── DOM refs ──────────────────────────────────────────────────────────── */
const $  = id => document.getElementById(id);
const startBtn       = $('start-btn');
const gameBoard      = $('game-board');
const startPrompt    = $('start-prompt');
const potAmount      = $('pot-amount');
const roundState     = $('round-state');
const communityCards = $('community-cards');
const privateCards   = $('private-cards');
const aiCards        = $('ai-cards');
const statusText     = $('status-text');
const actionButtons  = $('action-buttons');
const historyLog     = $('history-log');
const winnerBanner   = $('winner-banner');
const winnerText     = $('winner-text');
const humanStack     = $('human-stack');
const humanBet       = $('human-bet');
const aiStack        = $('ai-stack');
const aiBet          = $('ai-bet');
const btnFold        = $('btn-fold');
const btnCall        = $('btn-call');
const btnRaiseS      = $('btn-raise-s');
const btnRaiseM      = $('btn-raise-m');
const btnRaiseL      = $('btn-raise-l');
const raiseRow       = $('raise-row');

// Intelligence panel
const intelEmpty     = $('intel-empty');
const intelActive    = $('intel-active');
const lastActionName = $('last-action-name');
const probBars       = $('prob-bars');
const advBars        = $('adv-bars');

// Training panel
// (removed)

// Session
const sessionWins    = $('session-wins');
const sessionLosses  = $('session-losses');

/* ── Session state ─────────────────────────────────────────────────────── */
let wins = 0, losses = 0;

/* ── Cards ─────────────────────────────────────────────────────────────── */
function cardHtml(cardStr) {
    if (!cardStr) return '<div class="playing-card card-back"></div>';
    const rank = cardStr[0];
    const suit = cardStr[1];
    const colorClass = (suit === 'h' || suit === 'd') ? 'card-red' : 'card-black';
    const sym = SUIT_SYMBOLS[suit] || suit;
    return `<div class="playing-card ${colorClass}">
        <div class="card-top">${rank}${sym}</div>
        <div class="card-center">${sym}</div>
        <div class="card-bottom">${rank}${sym}</div>
    </div>`;
}

/* ── Intelligence Panel ────────────────────────────────────────────────── */
function renderIntelPanel(lastMove) {
    const emptyState = document.getElementById('intel-empty');
    const activeState = document.getElementById('intel-active');

    if (!lastMove || lastMove.action === null) {
        emptyState.classList.remove('hidden');
        activeState.classList.add('hidden');
        return;
    }

    emptyState.classList.add('hidden');
    activeState.classList.remove('hidden');

    let actionSentence = '';
    const chips = lastMove.chips_pushed || 0;
    
    switch (lastMove.action) {
        case 0:
            actionSentence = `The AI decided to Fold.`;
            break;
        case 1:
            actionSentence = chips > 0 
                ? `The AI decided to Call, matching the bet by putting ${chips} chips into the pot.` 
                : `The AI decided to Check (free).`;
            break;
        case 2:
            actionSentence = `The AI made a Small Bet, pushing ${chips} chips into the pot.`;
            break;
        case 3:
            actionSentence = `The AI made a Medium Bet, pushing ${chips} chips into the pot.`;
            break;
        case 4:
            actionSentence = `The AI made a Large Bet, pushing ${chips} chips into the pot.`;
            break;
    }
    
    document.getElementById('last-action-name').textContent = actionSentence;

    const probs  = lastMove.probs;
    const advs   = lastMove.advantages || Array(5).fill(0);

    // ── Probability bars ───────────────────────────────────────────────
    const maxProb = Math.max(...probs, 0.001);
    
    // Sort probabilities highest to lowest
    const probData = probs.map((p, i) => ({ p, i })).sort((a, b) => b.p - a.p);
    
    probBars.innerHTML = probData.map(({ p, i }) => {
        const pct     = (p * 100).toFixed(1);
        const barW    = (p / maxProb * 100).toFixed(1);
        const chosen  = i === lastMove.action ? 'chosen' : '';
        const color   = ACTION_COLORS[i];
        return `<div class="prob-bar-row ${chosen}" style="--fill-color:${color}">
            <span class="prob-action-name">${ACTION_NAMES[i]}</span>
            <div class="prob-track">
                <div class="prob-fill" style="width:0%;background:${color}" data-w="${barW}"></div>
            </div>
            <span class="prob-pct">${pct}%</span>
        </div>`;
    }).join('');

    // ── Advantage bars ─────────────────────────────────────────────────
    const maxAdv = Math.max(...advs.map(Math.abs), 0.001);
    
    // Sort advantages highest to lowest
    const advData = advs.map((v, i) => ({ v, i })).sort((a, b) => b.v - a.v);
    
    advBars.innerHTML = advData.map(({ v, i }) => {
        const barW    = (Math.abs(v) / maxAdv * 100).toFixed(1);
        const sign    = v >= 0 ? 'positive' : 'negative';
        const label   = (v >= 0 ? '+' : '') + v.toFixed(1);
        const chosen  = i === lastMove.action ? 'chosen' : '';
        return `<div class="adv-bar-row ${chosen}">
            <span class="adv-action-name">${ACTION_NAMES[i]}</span>
            <div class="adv-track">
                <div class="adv-fill ${sign}" style="width:0%" data-w="${barW}"></div>
            </div>
            <span class="adv-value ${sign}">${label}</span>
        </div>`;
    }).join('');

    // Trigger bar animations after a microtask (let DOM paint first)
    requestAnimationFrame(() => requestAnimationFrame(() => {
        document.querySelectorAll('.prob-fill[data-w], .adv-fill[data-w]').forEach(el => {
            el.style.width = el.dataset.w + '%';
        });
    }));
}


/* ── Main UI update ────────────────────────────────────────────────────── */
function updateUI(state) {
    if (state.error) { console.error(state.error); return; }

    gameBoard.classList.remove('hidden');

    potAmount.textContent = state.pot;
    roundState.textContent = ['Preflop', 'Flop', 'Turn', 'River'][state.round] || 'Showdown';
    humanStack.textContent = state.human_stack;
    humanBet.textContent   = state.human_bet;
    aiStack.textContent    = state.ai_stack;
    aiBet.textContent      = state.ai_bet;

    // Community cards
    communityCards.innerHTML = state.community.length
        ? state.community.map(cardHtml).join('')
        : '<div class="no-cards-hint">Community cards will appear here</div>';

    // Private cards
    privateCards.innerHTML = state.private.map(cardHtml).join('');

    // AI cards (face-down, revealed at showdown)
    if (state.done && state.ai_private && state.ai_private.length) {
        aiCards.innerHTML = state.ai_private.map(cardHtml).join('');
    } else {
        aiCards.innerHTML = cardHtml(null) + cardHtml(null);
    }

    // ── Game over ──────────────────────────────────────────────────────
    if (state.done) {
        actionButtons.classList.add('hidden');
        winnerBanner.classList.remove('hidden');
        if (state.winner === -1) {
            winnerText.textContent = 'Tie — chips split';
        } else if (state.winner === state.human_player) {
            winnerText.textContent = 'You Win!';
            wins++;
            sessionWins.textContent = wins;
        } else {
            winnerText.textContent = 'AI Wins';
            losses++;
            sessionLosses.textContent = losses;
        }
        statusText.textContent = 'Game over — press New Game to play again';
    } else {
        winnerBanner.classList.add('hidden');

        if (state.is_human_turn) {
            statusText.textContent = 'Your turn';
            actionButtons.classList.remove('hidden');

            btnFold.disabled = !state.legal_actions.includes(0);

            // Call / Check
            if (state.legal_actions.includes(1)) {
                btnCall.disabled   = false;
                btnCall.textContent = state.call_amount === 0
                    ? 'Check' : `Call (+${state.call_amount})`;
            } else {
                btnCall.disabled    = true;
                btnCall.textContent = 'Call';
            }

            // Raise buttons
            const raiseDefs = [
                { btn: btnRaiseS, action: 2, label: 'S', amt: state.raise_amount_s },
                { btn: btnRaiseM, action: 3, label: 'M', amt: state.raise_amount_m },
                { btn: btnRaiseL, action: 4, label: 'L', amt: state.raise_amount_l },
            ];
            let anyRaise = false;
            for (const r of raiseDefs) {
                if (state.legal_actions.includes(r.action)) {
                    r.btn.disabled   = false;
                    r.btn.textContent = `Bet ${r.label} (+${r.amt})`;
                    anyRaise = true;
                } else {
                    r.btn.disabled   = true;
                    r.btn.textContent = `Bet ${r.label}`;
                }
            }
            raiseRow.style.display = anyRaise ? '' : 'none';

        } else {
            statusText.textContent = 'AI is thinking…';
            actionButtons.classList.add('hidden');
        }
    }

    // ── History log ────────────────────────────────────────────────────
    const actionNames = ['Fold', 'Call', 'Raise-S', 'Raise-M', 'Raise-L'];
    historyLog.textContent = state.history
        .slice(-6)
        .map((a, i) => {
            const globalIdx = state.history.length - Math.min(state.history.length, 6) + i;
            const who = (globalIdx % 2 === state.human_player) ? 'You' : 'AI';
            return `${who}: ${actionNames[a]}`;
        })
        .join(' › ');

    // ── Intelligence panel ──────────────────────────────────────────────
    renderIntelPanel(state.last_ai_move);
}

/* ── API calls ─────────────────────────────────────────────────────────── */
async function startGame() {
    startBtn.textContent = 'Restart';
    const res   = await fetch('/api/start', { method: 'POST' });
    const state = await res.json();
    updateUI(state);
}

async function takeAction(actionIndex) {
    statusText.textContent = 'Sending…';
    actionButtons.classList.add('hidden');
    const res   = await fetch('/api/action', {
        method:  'POST',
        headers: { 'Content-Type': 'application/json' },
        body:    JSON.stringify({ action: actionIndex }),
    });
    const state = await res.json();
    updateUI(state);
}

/* ── Event listeners ───────────────────────────────────────────────────── */
document.addEventListener('DOMContentLoaded', () => {
    startBtn.addEventListener('click', startGame);
    document.getElementById('btn-play-again').addEventListener('click', startGame);
    btnFold.addEventListener('click',   () => takeAction(0));
    btnCall.addEventListener('click',   () => takeAction(1));
    btnRaiseS.addEventListener('click', () => takeAction(2));
    btnRaiseM.addEventListener('click', () => takeAction(3));
    btnRaiseL.addEventListener('click', () => takeAction(4));

    // Auto-start game since start prompt is removed
    startGame();
});
