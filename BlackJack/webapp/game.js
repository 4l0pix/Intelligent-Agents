/**
 * Blackjack Game Controller
 * Handles UI, game logic, and interaction with Monte Carlo AI
 */

// Game state
let bot = null;
let gameState = {
    playerHand: [],
    botHand: [],
    dealerHand: [],
    dealerHiddenCard: null,
    isPlayerTurn: false,
    roundOver: false
};

// Scores
let scores = {
    player: { wins: 0, losses: 0, ties: 0 },
    bot: { wins: 0, losses: 0, ties: 0 }
};

// Card suits and display
const SUITS = ['♠', '♥', '♦', '♣'];
const CARD_NAMES = ['A', '2', '3', '4', '5', '6', '7', '8', '9', '10', 'J', 'Q', 'K'];

// ===== TRAINING FUNCTIONS =====

function setEpisodes(value) {
    document.getElementById('episodes').value = value;
    document.getElementById('episodes-value').textContent = value.toLocaleString();
    
    // Update button selection
    document.querySelectorAll('.preset-buttons button').forEach(btn => {
        btn.classList.remove('selected');
    });
    event.target.classList.add('selected');
}

// Update slider value display
document.getElementById('episodes').addEventListener('input', function() {
    document.getElementById('episodes-value').textContent = parseInt(this.value).toLocaleString();
    
    // Clear button selection
    document.querySelectorAll('.preset-buttons button').forEach(btn => {
        btn.classList.remove('selected');
    });
});

async function startTraining() {
    const episodes = parseInt(document.getElementById('episodes').value);
    
    document.getElementById('train-btn').disabled = true;
    document.getElementById('training-progress').classList.remove('hidden');
    
    bot = new MonteCarloES();
    
    await bot.train(episodes, (progress, current) => {
        const percentage = Math.round(progress * 100);
        document.getElementById('progress-fill').style.width = percentage + '%';
        document.getElementById('progress-text').textContent = 
            `Training... ${percentage}% (${current.toLocaleString()} / ${episodes.toLocaleString()} episodes)`;
    });
    
    // Training complete
    document.getElementById('progress-fill').style.width = '100%';
    document.getElementById('progress-text').textContent = 'Training Complete! Starting game...';
    document.getElementById('trained-episodes').textContent = episodes.toLocaleString();
    
    // Generate policy tables
    generatePolicyTables();
    
    // Switch to game screen
    setTimeout(() => {
        document.getElementById('training-screen').classList.add('hidden');
        document.getElementById('game-screen').classList.remove('hidden');
        newRound();
    }, 1000);
}

function generatePolicyTables() {
    for (const usableAce of [true, false]) {
        const containerId = usableAce ? 'policy-ace' : 'policy-no-ace';
        const container = document.getElementById(containerId);
        const grid = bot.getPolicyGrid(usableAce);
        
        let html = '<table><tr><th></th>';
        // Header: dealer cards
        for (let d = 1; d <= 10; d++) {
            html += `<th>${d === 1 ? 'A' : d}</th>`;
        }
        html += '</tr>';
        
        // Rows: player sums
        for (const row of grid) {
            html += `<tr><td class="row-header">${row.playerSum}</td>`;
            for (const action of row.actions) {
                const className = action === STICK ? 'stick' : 'hit';
                const text = action === STICK ? 'S' : 'H';
                html += `<td class="${className}">${text}</td>`;
            }
            html += '</tr>';
        }
        html += '</table>';
        
        container.innerHTML = html;
    }
}

// ===== MODAL FUNCTIONS =====

function showInfoModal() {
    document.getElementById('info-modal').classList.remove('hidden');
}

function closeInfoModal() {
    document.getElementById('info-modal').classList.add('hidden');
}

// Close modal on outside click
document.getElementById('info-modal')?.addEventListener('click', function(e) {
    if (e.target === this) {
        closeInfoModal();
    }
});

// ===== CARD FUNCTIONS =====

function drawCard() {
    const CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];
    const valueIndex = Math.floor(Math.random() * 13);
    const value = CARD_VALUES[valueIndex];
    const name = CARD_NAMES[valueIndex];
    const suit = SUITS[Math.floor(Math.random() * 4)];
    const isRed = suit === '♥' || suit === '♦';
    
    return { value, name, suit, isRed };
}

function drawHand() {
    return [drawCard(), drawCard()];
}

function usableAce(hand) {
    const hasAce = hand.some(card => card.value === 1);
    const rawSum = hand.reduce((sum, card) => sum + card.value, 0);
    return hasAce && rawSum + 10 <= 21;
}

function sumHand(hand) {
    const rawSum = hand.reduce((sum, card) => sum + card.value, 0);
    if (usableAce(hand)) {
        return rawSum + 10;
    }
    return rawSum;
}

function isBust(hand) {
    return sumHand(hand) > 21;
}

function renderCard(card, hidden = false) {
    if (hidden) {
        return '<div class="card hidden-card"></div>';
    }
    const colorClass = card.isRed ? 'red' : 'black';
    return `
        <div class="card ${colorClass}">
            <div class="top">${card.name}${card.suit}</div>
            <div class="center">${card.suit}</div>
            <div class="bottom">${card.name}${card.suit}</div>
        </div>
    `;
}

function renderHand(hand, containerId, hideFirst = false) {
    const container = document.getElementById(containerId);
    let html = '';
    hand.forEach((card, index) => {
        html += renderCard(card, hideFirst && index === 1);
    });
    container.innerHTML = html;
}

function updateSum(hand, elementId, hidden = false) {
    const element = document.getElementById(elementId);
    if (hidden) {
        element.textContent = hand[0].value === 1 ? '11 or 1' : hand[0].value;
    } else {
        element.textContent = sumHand(hand);
    }
}

// ===== GAME FUNCTIONS =====

function newRound() {
    // Reset game state
    gameState.playerHand = drawHand();
    gameState.botHand = drawHand();
    gameState.dealerHand = drawHand();
    gameState.isPlayerTurn = true;
    gameState.roundOver = false;
    
    // Clear results
    document.getElementById('player-result').className = 'result-badge';
    document.getElementById('player-result').textContent = '';
    document.getElementById('bot-result').className = 'result-badge';
    document.getElementById('bot-result').textContent = '';
    document.getElementById('bot-action').textContent = '';
    
    // Render hands
    renderHand(gameState.dealerHand, 'dealer-cards', true);
    renderHand(gameState.botHand, 'bot-cards');
    renderHand(gameState.playerHand, 'player-cards');
    
    // Update sums
    updateSum(gameState.dealerHand, 'dealer-sum', true);
    updateSum(gameState.botHand, 'bot-sum');
    updateSum(gameState.playerHand, 'player-sum');
    
    // Enable buttons
    document.getElementById('hit-btn').disabled = false;
    document.getElementById('stick-btn').disabled = false;
    document.getElementById('hit-btn').classList.remove('hidden');
    document.getElementById('stick-btn').classList.remove('hidden');
    document.getElementById('new-round-btn').classList.add('hidden');
    
    // Log
    addLog('New round started!', 'result');
    addLog(`Your hand: ${formatHand(gameState.playerHand)} (${sumHand(gameState.playerHand)})`);
    addLog(`Dealer shows: ${gameState.dealerHand[0].name}${gameState.dealerHand[0].suit}`);
    
    // Bot plays immediately
    playBot();
}

function formatHand(hand) {
    return hand.map(c => `${c.name}${c.suit}`).join(' ');
}

function addLog(message, type = '') {
    const container = document.getElementById('log-container');
    const entry = document.createElement('div');
    entry.className = `log-entry ${type}`;
    entry.textContent = message;
    container.insertBefore(entry, container.firstChild);
    
    // Keep only last 20 entries
    while (container.children.length > 20) {
        container.removeChild(container.lastChild);
    }
}

function playBot() {
    const dealerShowing = gameState.dealerHand[0].value;
    
    addLog(`Carlos's turn...`, 'action');
    
    // Bot plays using learned policy
    while (true) {
        const botSum = sumHand(gameState.botHand);
        
        // Auto-hit if under 12
        if (botSum < 12) {
            gameState.botHand.push(drawCard());
            continue;
        }
        
        if (isBust(gameState.botHand)) {
            break;
        }
        
        const state = {
            playerSum: botSum,
            dealerShowing: dealerShowing,
            usableAce: usableAce(gameState.botHand)
        };
        
        const action = bot.getAction(state);
        
        if (action === STICK) {
            document.getElementById('bot-action').textContent = 'STICKS';
            addLog(`Carlos sticks at ${botSum}`);
            break;
        } else {
            document.getElementById('bot-action').textContent = 'HITS';
            gameState.botHand.push(drawCard());
            addLog(`Carlos hits, now has ${sumHand(gameState.botHand)}`);
            
            // Update display
            renderHand(gameState.botHand, 'bot-cards');
            updateSum(gameState.botHand, 'bot-sum');
        }
    }
    
    // Final bot display
    renderHand(gameState.botHand, 'bot-cards');
    updateSum(gameState.botHand, 'bot-sum');
    
    if (isBust(gameState.botHand)) {
        document.getElementById('bot-action').textContent = 'BUST!';
        addLog(`Carlos busts with ${sumHand(gameState.botHand)}!`);
    }
}

function playerHit() {
    if (!gameState.isPlayerTurn || gameState.roundOver) return;
    
    gameState.playerHand.push(drawCard());
    renderHand(gameState.playerHand, 'player-cards');
    updateSum(gameState.playerHand, 'player-sum');
    
    addLog(`You hit, now have ${sumHand(gameState.playerHand)}`);
    
    if (isBust(gameState.playerHand)) {
        addLog(`You bust with ${sumHand(gameState.playerHand)}!`);
        endRound();
    }
}

function playerStick() {
    if (!gameState.isPlayerTurn || gameState.roundOver) return;
    
    addLog(`You stick at ${sumHand(gameState.playerHand)}`);
    gameState.isPlayerTurn = false;
    endRound();
}

function endRound() {
    gameState.roundOver = true;
    
    // Disable action buttons
    document.getElementById('hit-btn').disabled = true;
    document.getElementById('stick-btn').disabled = true;
    
    // Reveal dealer's hand
    renderHand(gameState.dealerHand, 'dealer-cards', false);
    addLog(`Dealer reveals: ${formatHand(gameState.dealerHand)} (${sumHand(gameState.dealerHand)})`);
    
    // Dealer plays
    while (sumHand(gameState.dealerHand) < 17) {
        gameState.dealerHand.push(drawCard());
        addLog(`Dealer hits, now has ${sumHand(gameState.dealerHand)}`);
    }
    
    renderHand(gameState.dealerHand, 'dealer-cards');
    updateSum(gameState.dealerHand, 'dealer-sum');
    
    if (isBust(gameState.dealerHand)) {
        addLog(`Dealer busts with ${sumHand(gameState.dealerHand)}!`);
    } else {
        addLog(`Dealer sticks at ${sumHand(gameState.dealerHand)}`);
    }
    
    // Determine results
    const playerSum = sumHand(gameState.playerHand);
    const botSum = sumHand(gameState.botHand);
    const dealerSum = sumHand(gameState.dealerHand);
    const playerBust = isBust(gameState.playerHand);
    const botBust = isBust(gameState.botHand);
    const dealerBust = isBust(gameState.dealerHand);
    
    // Player result
    let playerResult, playerClass;
    if (playerBust) {
        playerResult = 'BUST!';
        playerClass = 'bust';
        scores.player.losses++;
    } else if (dealerBust) {
        playerResult = 'WIN!';
        playerClass = 'win';
        scores.player.wins++;
    } else if (playerSum > dealerSum) {
        playerResult = 'WIN!';
        playerClass = 'win';
        scores.player.wins++;
    } else if (playerSum < dealerSum) {
        playerResult = 'LOSE';
        playerClass = 'lose';
        scores.player.losses++;
    } else {
        playerResult = 'TIE';
        playerClass = 'tie';
        scores.player.ties++;
    }
    
    // Bot result
    let botResult, botClass;
    if (botBust) {
        botResult = 'BUST!';
        botClass = 'bust';
        scores.bot.losses++;
    } else if (dealerBust) {
        botResult = 'WIN!';
        botClass = 'win';
        scores.bot.wins++;
    } else if (botSum > dealerSum) {
        botResult = 'WIN!';
        botClass = 'win';
        scores.bot.wins++;
    } else if (botSum < dealerSum) {
        botResult = 'LOSE';
        botClass = 'lose';
        scores.bot.losses++;
    } else {
        botResult = 'TIE';
        botClass = 'tie';
        scores.bot.ties++;
    }
    
    // Display results
    const playerResultEl = document.getElementById('player-result');
    playerResultEl.textContent = playerResult;
    playerResultEl.className = `result-badge ${playerClass}`;
    
    const botResultEl = document.getElementById('bot-result');
    botResultEl.textContent = botResult;
    botResultEl.className = `result-badge ${botClass}`;
    
    // Log results
    addLog(`Results - You: ${playerResult}, Bot: ${botResult}`, 'result');
    
    // Update scoreboard
    updateScoreboard();
    
    // Show new round button
    document.getElementById('hit-btn').classList.add('hidden');
    document.getElementById('stick-btn').classList.add('hidden');
    document.getElementById('new-round-btn').classList.remove('hidden');
}

function updateScoreboard() {
    document.getElementById('player-wins').textContent = scores.player.wins;
    document.getElementById('player-losses').textContent = scores.player.losses;
    document.getElementById('player-ties').textContent = scores.player.ties;
    
    document.getElementById('bot-wins').textContent = scores.bot.wins;
    document.getElementById('bot-losses').textContent = scores.bot.losses;
    document.getElementById('bot-ties').textContent = scores.bot.ties;
}

function resetScores() {
    scores = {
        player: { wins: 0, losses: 0, ties: 0 },
        bot: { wins: 0, losses: 0, ties: 0 }
    };
    updateScoreboard();
    addLog('Scores reset!', 'result');
}
