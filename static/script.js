document.addEventListener('DOMContentLoaded', () => {
    const startBtn = document.getElementById('start-btn');
    const gameBoard = document.getElementById('game-board');
    
    // UI Elements
    const potAmount = document.getElementById('pot-amount');
    const roundState = document.getElementById('round-state');
    const communityCards = document.getElementById('community-cards');
    const privateCards = document.getElementById('private-cards');
    const aiCards = document.getElementById('ai-cards');
    const statusText = document.getElementById('status-text');
    const actionButtons = document.getElementById('action-buttons');
    const historyLog = document.getElementById('history-log');
    const winnerBanner = document.getElementById('winner-banner');
    const winnerText = document.getElementById('winner-text');
    
    // Player Stats
    const humanStack = document.getElementById('human-stack');
    const humanBet = document.getElementById('human-bet');
    const aiStack = document.getElementById('ai-stack');
    const aiBet = document.getElementById('ai-bet');
    
    // Action Buttons
    const btnFold = document.getElementById('btn-fold');
    const btnCall = document.getElementById('btn-call');
    const btnRaise = document.getElementById('btn-raise');
    
    const suitSymbols = {
        'h': '♥', 'd': '♦', 'c': '♣', 's': '♠'
    };
    
    function getCardHtml(cardStr) {
        if (!cardStr) return '<div class="playing-card card-back"></div>';
        const rank = cardStr[0];
        const suit = cardStr[1];
        const colorClass = (suit === 'h' || suit === 'd') ? 'card-red' : 'card-black';
        const symbol = suitSymbols[suit];
        
        return `
            <div class="playing-card ${colorClass}">
                <div class="card-top">${rank}${symbol}</div>
                <div class="card-center">${symbol}</div>
                <div class="card-bottom">${rank}${symbol}</div>
            </div>
        `;
    }
    
    function updateUI(state) {
        if (state.error) {
            alert(state.error);
            return;
        }
        
        gameBoard.classList.remove('hidden');
        
        // Update Table Info
        potAmount.textContent = state.pot;
        const rounds = ["Preflop", "Flop", "Turn", "River"];
        roundState.textContent = rounds[state.round] || "Showdown";
        
        // Stacks & Bets
        humanStack.textContent = state.human_stack;
        humanBet.textContent = state.human_bet;
        aiStack.textContent = state.ai_stack;
        aiBet.textContent = state.ai_bet;
        
        // Update Cards
        communityCards.innerHTML = state.community.map(getCardHtml).join('') || '<div class="status-text">No community cards</div>';
        privateCards.innerHTML = state.private.map(getCardHtml).join('');
        
        // AI Cards (Face down unless game over and ai_private has cards)
        if (state.done && state.ai_private && state.ai_private.length > 0) {
            aiCards.innerHTML = state.ai_private.map(getCardHtml).join('');
        } else {
            // Render 2 face down cards
            aiCards.innerHTML = '<div class="playing-card card-back"></div><div class="playing-card card-back"></div>';
        }
        
        // Handle Game Over
        if (state.done) {
            actionButtons.classList.add('hidden');
            statusText.textContent = "Game Over!";
            winnerBanner.classList.remove('hidden');
            
            if (state.winner === -1) {
                winnerText.textContent = "Tie!";
            } else if (state.winner === state.human_player) {
                winnerText.textContent = "You Win!";
            } else {
                winnerText.textContent = "AI Wins!";
            }
        } else {
            winnerBanner.classList.add('hidden');
            if (state.is_human_turn) {
                statusText.textContent = "Your turn. Choose an action:";
                actionButtons.classList.remove('hidden');
                
                btnFold.disabled = !state.legal_actions.includes(0);
                
                if (state.legal_actions.includes(1)) {
                    btnCall.disabled = false;
                    btnCall.textContent = state.call_amount === 0 ? "Check" : `Call (${state.call_amount})`;
                } else {
                    btnCall.disabled = true;
                }
                
                if (state.legal_actions.includes(2)) {
                    btnRaise.disabled = false;
                    btnRaise.textContent = `Raise (+${state.raise_amount})`;
                } else {
                    btnRaise.disabled = true;
                    btnRaise.textContent = "Raise";
                }
                
            } else {
                statusText.textContent = "AI is thinking...";
                actionButtons.classList.add('hidden');
            }
        }
        
        // Update History Log
        const actionNames = ["Fold", "Call", "Raise"];
        const lastFewActions = state.history.slice(-4); // just show last 4 actions to keep it clean
        historyLog.innerHTML = lastFewActions.map((a, i) => {
            const globalIndex = state.history.length - lastFewActions.length + i;
            const player = (globalIndex % 2 === state.human_player) ? "You" : "AI";
            return `${player} ${actionNames[a]}ed`;
        }).join(' &rarr; ');
    }
    
    async function startGame() {
        startBtn.textContent = "Restart Game";
        const res = await fetch('/api/start', { method: 'POST' });
        const state = await res.json();
        updateUI(state);
    }
    
    async function takeAction(actionIndex) {
        statusText.textContent = "Sending action...";
        actionButtons.classList.add('hidden');
        
        const res = await fetch('/api/action', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ action: actionIndex })
        });
        const state = await res.json();
        updateUI(state);
    }
    
    startBtn.addEventListener('click', startGame);
    btnFold.addEventListener('click', () => takeAction(0));
    btnCall.addEventListener('click', () => takeAction(1));
    btnRaise.addEventListener('click', () => takeAction(2));
});
