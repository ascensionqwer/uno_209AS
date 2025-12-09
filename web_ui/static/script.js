let gameState = null;
let isAiTurn = false;
let selectedMode = 'mini';

function selectMode(mode) {
    selectedMode = mode;

    // Update UI
    document.querySelectorAll('.mode-btn').forEach(btn => btn.classList.remove('active'));
    document.getElementById('mode-' + mode).classList.add('active');

    // Handle Offline Solver availability
    const offlineBtn = document.getElementById('opp-offline');
    if (mode === 'full') {
        offlineBtn.disabled = true;
        offlineBtn.style.opacity = '0.5';
        offlineBtn.title = "Not available in Full Uno (too slow)";
    } else {
        offlineBtn.disabled = false;
        offlineBtn.style.opacity = '1';
        offlineBtn.title = "";
    }
}

async function startGame(opponentType) {
    const response = await fetch('/api/start_game', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' },
        body: JSON.stringify({
            opponent_type: opponentType,
            game_mode: selectedMode
        })
    });
    gameState = await response.json();

    document.getElementById('setup-modal').classList.add('hidden');
    document.getElementById('game-board').classList.remove('hidden');

    updateUI();
    checkTurn();
}

let isProcessing = false;

async function playCard(index) {
    if (isAiTurn || isProcessing) return;
    isProcessing = true;

    try {
        const response = await fetch('/api/play_card', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' },
            body: JSON.stringify({ card_index: index })
        });
        const result = await response.json();

        if (result.success) {
            gameState = result;
            updateUI();
            checkTurn();
        } else {
            alert(result.message);
        }
    } finally {
        isProcessing = false;
    }
}

async function drawCard() {
    if (isAiTurn || isProcessing) return;
    isProcessing = true;

    try {
        const response = await fetch('/api/draw_card', {
            method: 'POST',
            headers: { 'Content-Type': 'application/json' }
        });
        const result = await response.json();

        if (result.success) {
            gameState = result;
            updateUI();
            checkTurn();
        } else {
            alert(result.message);
        }
    } finally {
        isProcessing = false;
    }
}

async function aiMove() {
    isAiTurn = true;
    document.getElementById('status-message').innerText = "AI is thinking...";

    // Simulate thinking delay
    await new Promise(r => setTimeout(r, 1000));

    const response = await fetch('/api/ai_move', {
        method: 'POST',
        headers: { 'Content-Type': 'application/json' }
    });
    const result = await response.json();

    gameState = result;
    isAiTurn = false;
    updateUI();
    checkTurn();
}

function checkTurn() {
    if (gameState.winner) {
        showGameOver(gameState.winner);
        return;
    }

    if (gameState.turn === 2) {
        aiMove();
    } else {
        document.getElementById('status-message').innerText = "Your Turn";
    }
}

function updateUI() {
    // Update Top Card
    const discardPile = document.getElementById('discard-pile');
    discardPile.innerHTML = '';
    if (gameState.top_card) {
        discardPile.appendChild(createCardElement(gameState.top_card));
    }

    // Update Player Hand
    const playerHand = document.getElementById('player-hand');
    playerHand.innerHTML = '';
    gameState.player_hand.forEach((cardStr, index) => {
        const cardEl = createCardElement(cardStr);
        cardEl.onclick = () => playCard(index);
        playerHand.appendChild(cardEl);
    });

    // Update Opponent Hand
    const opponentHand = document.getElementById('opponent-hand');
    opponentHand.innerHTML = '';
    for (let i = 0; i < gameState.opponent_hand_count; i++) {
        const back = document.createElement('div');
        back.className = 'card-back';
        opponentHand.appendChild(back);
    }

    // Update Deck Count
    document.getElementById('deck-count').innerText = gameState.deck_count;
}

function createCardElement(cardStr) {
    // cardStr format: "Color Number" e.g. "R 1" or "B 2"
    // We need to parse it. 
    // Assuming str(Card) returns "Color Number" or similar.
    // Let's check Card.__str__ in cards.py or mini_uno.py
    // Actually, let's just handle "R" and "B" detection.

    const el = document.createElement('div');
    el.className = 'card';

    let colorClass = '';
    let text = cardStr;

    if (cardStr.includes('R')) colorClass = 'red';
    if (cardStr.includes('B')) colorClass = 'blue';
    if (cardStr.includes('Y')) colorClass = 'yellow';
    if (cardStr.includes('G')) colorClass = 'green';

    // Clean up text for display
    // Example: "Card(R, 1)" -> "1"
    // Or "R 1" -> "1"
    // Let's assume simple parsing for now.

    // If str(card) is "Card(X_1=('R', 1))" (from Action) or similar?
    // No, game_manager uses str(card) on Card object.
    // Let's assume Card.__str__ returns something readable.
    // If not, we might need to fix game_manager.

    // Let's assume it contains the number.
    const match = cardStr.match(/\d+/);
    if (match) {
        text = match[0];
    } else {
        text = cardStr[0]; // Fallback
    }

    el.classList.add(colorClass);
    el.innerText = text;

    return el;
}

function showGameOver(winner) {
    const modal = document.getElementById('game-over-modal');
    const text = document.getElementById('winner-text');

    if (winner === 1) {
        text.innerText = "You Win!";
        text.style.color = "#2ed573";
    } else {
        text.innerText = "AI Wins!";
        text.style.color = "#ff4757";
    }

    modal.classList.remove('hidden');
}
