document.addEventListener('DOMContentLoaded', () => {
    const startBtn      = document.getElementById('start-btn');
    const gameBoard     = document.getElementById('game-board');

    // UI elements
    const potAmount     = document.getElementById('pot-amount');
    const roundState    = document.getElementById('round-state');
    const communityCards= document.getElementById('community-cards');
    const privateCards  = document.getElementById('private-cards');
    const aiCards       = document.getElementById('ai-cards');
    const statusText    = document.getElementById('status-text');
    const actionButtons = document.getElementById('action-buttons');
    const historyLog    = document.getElementById('history-log');
    const winnerBanner  = document.getElementById('winner-banner');
    const winnerText    = document.getElementById('winner-text');

    // Stacks
    const humanStack    = document.getElementById('human-stack');
    const humanBet      = document.getElementById('human-bet');
    const aiStack       = document.getElementById('ai-stack');
    const aiBet         = document.getElementById('ai-bet');

    // Buttons
    const btnFold       = document.getElementById('btn-fold');
    const btnCall       = document.getElementById('btn-call');
    const btnRaiseS     = document.getElementById('btn-raise-s');
    const btnRaiseM     = document.getElementById('btn-raise-m');
    const btnRaiseL     = document.getElementById('btn-raise-l');
    const raiseRow      = document.getElementById('raise-row');

    const suitSymbols = { h: '♥', d: '♦', c: '♣', s: '♠' };

    function getCardHtml(cardStr) {
        if (!cardStr) return '<div class="playing-card card-back"></div>';
        const rank = cardStr[0];
        const suit = cardStr[1];
        const colorClass = (suit === 'h' || suit === 'd') ? 'card-red' : 'card-black';
        const sym = suitSymbols[suit];
        return `
            <div class="playing-card ${colorClass}">
                <div class="card-top">${rank}${sym}</div>
                <div class="card-center">${sym}</div>
                <div class="card-bottom">${rank}${sym}</div>
            </div>`;
    }

    function updateUI(state) {
        if (state.error) { alert(state.error); return; }

        gameBoard.classList.remove('hidden');

        potAmount.textContent = state.pot;
        roundState.textContent = ["Preflop", "Flop", "Turn", "River"][state.round] || "Showdown";

        humanStack.textContent = state.human_stack;
        humanBet.textContent   = state.human_bet;
        aiStack.textContent    = state.ai_stack;
        aiBet.textContent      = state.ai_bet;

        communityCards.innerHTML = state.community.map(getCardHtml).join('')
            || '<div class="status-text">No community cards</div>';
        privateCards.innerHTML = state.private.map(getCardHtml).join('');

        if (state.done && state.ai_private && state.ai_private.length > 0) {
            aiCards.innerHTML = state.ai_private.map(getCardHtml).join('');
        } else {
            aiCards.innerHTML = '<div class="playing-card card-back"></div>'
                              + '<div class="playing-card card-back"></div>';
        }

        // Game over
        if (state.done) {
            actionButtons.classList.add('hidden');
            statusText.textContent = "Game Over!";
            winnerBanner.classList.remove('hidden');
            if (state.winner === -1) {
                winnerText.textContent = "Tie!";
            } else if (state.winner === state.human_player) {
                winnerText.textContent = "You Win! 🎉";
            } else {
                winnerText.textContent = "AI Wins 🤖";
            }
            return;
        }

        winnerBanner.classList.add('hidden');

        if (state.is_human_turn) {
            statusText.textContent = "Your turn:";
            actionButtons.classList.remove('hidden');

            // Fold — always available
            btnFold.disabled = !state.legal_actions.includes(0);

            // Call / Check
            if (state.legal_actions.includes(1)) {
                btnCall.disabled = false;
                btnCall.textContent = state.call_amount === 0
                    ? "Check" : `Call (${state.call_amount})`;
            } else {
                btnCall.disabled = true;
                btnCall.textContent = "Call";
            }

            // Three raise buttons
            const raiseActions = [
                { id: btnRaiseS, action: 2, label: 'S', amount: state.raise_amount_s },
                { id: btnRaiseM, action: 3, label: 'M', amount: state.raise_amount_m },
                { id: btnRaiseL, action: 4, label: 'L', amount: state.raise_amount_l },
            ];
            let anyRaiseAvailable = false;
            for (const rb of raiseActions) {
                if (state.legal_actions.includes(rb.action)) {
                    rb.id.disabled = false;
                    rb.id.textContent = `Bet ${rb.label} (+${rb.amount})`;
                    anyRaiseAvailable = true;
                } else {
                    rb.id.disabled = true;
                    rb.id.textContent = `Bet ${rb.label}`;
                }
            }
            raiseRow.style.display = anyRaiseAvailable ? '' : 'none';

        } else {
            statusText.textContent = "AI is thinking...";
            actionButtons.classList.add('hidden');
        }

        // History log
        const actionNames = ["Fold", "Call", "Raise-S", "Raise-M", "Raise-L"];
        const lastActions  = state.history.slice(-5);
        historyLog.innerHTML = lastActions.map((a, i) => {
            const globalIdx = state.history.length - lastActions.length + i;
            const who = (globalIdx % 2 === state.human_player) ? "You" : "AI";
            return `${who}: ${actionNames[a]}`;
        }).join(' → ');
    }

    async function startGame() {
        startBtn.textContent = "Restart";
        const res   = await fetch('/api/start', { method: 'POST' });
        const state = await res.json();
        updateUI(state);
    }

    async function takeAction(actionIndex) {
        statusText.textContent = "Sending...";
        actionButtons.classList.add('hidden');
        const res   = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionIndex }),
        });
        const state = await res.json();
        updateUI(state);
    }

    startBtn.addEventListener('click', startGame);
    btnFold.addEventListener('click',   () => takeAction(0));
    btnCall.addEventListener('click',   () => takeAction(1));
    btnRaiseS.addEventListener('click', () => takeAction(2));
    btnRaiseM.addEventListener('click', () => takeAction(3));
    btnRaiseL.addEventListener('click', () => takeAction(4));
});
