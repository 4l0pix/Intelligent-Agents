// Game Controller for Kuhn Poker

class KuhnPokerGame {
    constructor() {
        this.cfr = null;
        this.iterations = 10000;
        
        // Game state
        this.playerChips = 50;
        this.botChips = 50;
        this.pot = 0;
        this.playerCard = null;
        this.botCard = null;
        this.history = '';
        this.isPlayerTurn = true;
        this.gameActive = false;
        
        // Stats
        this.playerWins = 0;
        this.botWins = 0;
        this.totalHands = 0;
        this.gameLog = [];
        
        // Card display
        this.cardSymbols = { 0: 'J', 1: 'Q', 2: 'K' };
        this.cardClasses = { 0: 'jack', 1: 'queen', 2: 'king' };
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Iterations slider
        const iterSlider = document.getElementById('iterations-slider');
        const iterValue = document.getElementById('iterations-value');
        iterSlider.addEventListener('input', () => {
            this.iterations = parseInt(iterSlider.value);
            iterValue.textContent = this.iterations.toLocaleString();
        });
        
        // Start training
        document.getElementById('start-btn').addEventListener('click', () => this.startTraining());
        
        // Game actions
        document.getElementById('check-btn').addEventListener('click', () => this.playerAction(0));
        document.getElementById('bet-btn').addEventListener('click', () => this.playerAction(1));
        document.getElementById('fold-btn').addEventListener('click', () => this.playerAction(0));
        document.getElementById('call-btn').addEventListener('click', () => this.playerAction(1));
        document.getElementById('deal-btn').addEventListener('click', () => this.dealNewHand());
        
        // Reset
        document.getElementById('reset-btn').addEventListener('click', () => this.resetGame());
        
        // Info modal
        document.getElementById('info-btn').addEventListener('click', () => this.showInfoModal());
        document.querySelector('.close-btn').addEventListener('click', () => this.hideInfoModal());
        document.getElementById('info-modal').addEventListener('click', (e) => {
            if (e.target.id === 'info-modal') this.hideInfoModal();
        });
    }
    
    async startTraining() {
        const startBtn = document.getElementById('start-btn');
        startBtn.disabled = true;
        startBtn.textContent = 'Training...';
        
        const progressBar = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        document.getElementById('progress-container').classList.remove('hidden');
        
        this.cfr = new CFRTrainer();
        
        try {
            await this.cfr.train(this.iterations, async (progress, current, total) => {
                progressBar.style.width = `${progress}%`;
                progressText.textContent = `Training: ${current.toLocaleString()} / ${total.toLocaleString()} iterations`;
            });
            
            progressText.textContent = 'Training complete!';
            
            setTimeout(() => {
                this.showGameScreen();
            }, 500);
            
        } catch (error) {
            console.error('Training error:', error);
            progressText.textContent = 'Training failed!';
            startBtn.disabled = false;
            startBtn.textContent = 'Train Bot';
        }
    }
    
    showGameScreen() {
        document.getElementById('training-screen').classList.add('hidden');
        document.getElementById('game-screen').classList.remove('hidden');
        this.updateDisplay();
    }
    
    dealNewHand() {
        if (this.playerChips <= 0 || this.botChips <= 0) {
            this.addLogEntry('Game over! Reset to play again.', 'action');
            return;
        }
        
        // Shuffle and deal
        const deck = [0, 1, 2];
        for (let i = deck.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [deck[i], deck[j]] = [deck[j], deck[i]];
        }
        
        this.playerCard = deck[0];
        this.botCard = deck[1];
        this.history = '';
        this.pot = 2; // Both ante 1
        this.playerChips -= 1;
        this.botChips -= 1;
        this.gameActive = true;
        
        // Randomly decide who acts first
        this.isPlayerTurn = Math.random() < 0.5;
        
        this.updateDisplay();
        
        const cardName = this.cardSymbols[this.playerCard];
        this.addLogEntry(`New hand dealt. You have ${cardName}.`, 'action');
        this.setStatus(`You have ${cardName}. ${this.isPlayerTurn ? 'Your turn.' : 'Bot is thinking...'}`);
        
        if (!this.isPlayerTurn) {
            setTimeout(() => this.botMove(), 1000);
        }
    }
    
    playerAction(action) {
        if (!this.gameActive || !this.isPlayerTurn) return;
        
        const actionChar = action === 0 ? 'p' : 'b';
        this.history += actionChar;
        
        if (action === 1) {
            // Bet or call
            this.playerChips -= 1;
            this.pot += 1;
            this.addLogEntry(this.history.length <= 1 || this.history[this.history.length - 2] !== 'b' 
                ? 'You bet 1.' : 'You call.', 'action');
        } else {
            // Check or fold
            if (this.history.length > 1 && this.history[this.history.length - 2] === 'b') {
                this.addLogEntry('You fold.', 'action');
            } else {
                this.addLogEntry('You check.', 'action');
            }
        }
        
        this.isPlayerTurn = false;
        this.updateDisplay();
        
        // Check if game is over
        if (this.checkGameEnd()) {
            return;
        }
        
        // Bot's turn
        this.setStatus('Bot is thinking...');
        setTimeout(() => this.botMove(), 1000);
    }
    
    botMove() {
        if (!this.gameActive) return;
        
        const action = this.cfr.getAction(this.botCard, this.history, this.history.length % 2 === 0);
        const actionChar = action === 0 ? 'p' : 'b';
        this.history += actionChar;
        
        if (action === 1) {
            this.botChips -= 1;
            this.pot += 1;
            this.addLogEntry(this.history.length <= 1 || this.history[this.history.length - 2] !== 'b' 
                ? 'Bot bets 1.' : 'Bot calls.', 'action');
        } else {
            if (this.history.length > 1 && this.history[this.history.length - 2] === 'b') {
                this.addLogEntry('Bot folds.', 'action');
            } else {
                this.addLogEntry('Bot checks.', 'action');
            }
        }
        
        this.isPlayerTurn = true;
        this.updateDisplay();
        
        // Check if game is over
        if (this.checkGameEnd()) {
            return;
        }
        
        this.setStatus('Your turn.');
    }
    
    checkGameEnd() {
        const result = this.getResult();
        if (result === null) return false;
        
        this.gameActive = false;
        this.totalHands++;
        
        // Reveal bot's card
        document.getElementById('opponent-card').textContent = this.cardSymbols[this.botCard];
        document.getElementById('opponent-card').className = `card ${this.cardClasses[this.botCard]}`;
        
        let message;
        let logClass;
        
        if (result > 0) {
            this.playerWins++;
            this.playerChips += this.pot;
            message = `You win ${this.pot} chips! (${this.cardSymbols[this.playerCard]} vs ${this.cardSymbols[this.botCard]})`;
            logClass = 'win';
        } else if (result < 0) {
            this.botWins++;
            this.botChips += this.pot;
            message = `Bot wins ${this.pot} chips. (${this.cardSymbols[this.playerCard]} vs ${this.cardSymbols[this.botCard]})`;
            logClass = 'lose';
        } else {
            // Split (shouldn't happen in Kuhn)
            this.playerChips += this.pot / 2;
            this.botChips += this.pot / 2;
            message = 'Split pot.';
            logClass = 'action';
        }
        
        this.pot = 0;
        this.addLogEntry(message, logClass);
        this.setStatus(message + ' Deal a new hand.');
        this.updateDisplay();
        
        return true;
    }
    
    getResult() {
        // Determine winner based on history
        // Returns positive for player win, negative for bot win, null if not terminal
        
        if (this.history.length < 2) return null;
        
        const last = this.history[this.history.length - 1];
        const secondLast = this.history[this.history.length - 2];
        
        // Fold
        if (last === 'p' && secondLast === 'b') {
            // Whoever folded loses
            const folderIsPlayer = (this.history.length % 2 === 1) === this.isPlayerTurn;
            // Actually, we need to track who made which move
            // In our setup: if history length is odd and it was player's turn structure
            // Let's simplify: last action is fold, previous was bet
            // The folder is whoever made the last 'p' after 'b'
            
            // Count moves to determine who folded
            // If player moved first: odd positions are player's
            const playerMovedFirst = this.history.length > 0;
            const folderIndex = this.history.length - 1;
            
            // This is getting complex - let's use a simpler approach
            // Check the last action and who just acted
            if (!this.isPlayerTurn) {
                // Player just folded
                return -1;
            } else {
                // Bot just folded
                return 1;
            }
        }
        
        // Showdown
        if (this.history === 'pp' || this.history === 'bb' || this.history === 'pbb' || this.history === 'pbp') {
            if (this.history === 'pbp') {
                // Fold after pass-bet
                if (!this.isPlayerTurn) {
                    return -1; // Player folded
                } else {
                    return 1; // Bot folded
                }
            }
            
            // Actual showdown
            if (this.playerCard > this.botCard) {
                return 1;
            } else {
                return -1;
            }
        }
        
        return null;
    }
    
    updateDisplay() {
        // Update chips
        document.getElementById('player-chips').textContent = this.playerChips;
        document.getElementById('opponent-chips').textContent = this.botChips;
        document.getElementById('pot-value').textContent = this.pot;
        
        // Update cards
        if (this.playerCard !== null) {
            document.getElementById('player-card').textContent = this.cardSymbols[this.playerCard];
            document.getElementById('player-card').className = `card ${this.cardClasses[this.playerCard]}`;
        } else {
            document.getElementById('player-card').textContent = '-';
            document.getElementById('player-card').className = 'card';
        }
        
        if (!this.gameActive && this.botCard !== null) {
            document.getElementById('opponent-card').textContent = this.cardSymbols[this.botCard];
            document.getElementById('opponent-card').className = `card ${this.cardClasses[this.botCard]}`;
        } else {
            document.getElementById('opponent-card').textContent = '?';
            document.getElementById('opponent-card').className = 'card back';
        }
        
        // Update action buttons
        const checkBtn = document.getElementById('check-btn');
        const betBtn = document.getElementById('bet-btn');
        const foldBtn = document.getElementById('fold-btn');
        const callBtn = document.getElementById('call-btn');
        const dealBtn = document.getElementById('deal-btn');
        
        if (!this.gameActive) {
            checkBtn.classList.add('hidden');
            betBtn.classList.add('hidden');
            foldBtn.classList.add('hidden');
            callBtn.classList.add('hidden');
            dealBtn.disabled = false;
        } else if (this.isPlayerTurn) {
            // Check what actions are available
            const facingBet = this.history.length > 0 && this.history[this.history.length - 1] === 'b';
            
            if (facingBet) {
                checkBtn.classList.add('hidden');
                betBtn.classList.add('hidden');
                foldBtn.classList.remove('hidden');
                callBtn.classList.remove('hidden');
                foldBtn.disabled = false;
                callBtn.disabled = false;
            } else {
                checkBtn.classList.remove('hidden');
                betBtn.classList.remove('hidden');
                foldBtn.classList.add('hidden');
                callBtn.classList.add('hidden');
                checkBtn.disabled = false;
                betBtn.disabled = false;
            }
            dealBtn.disabled = true;
        } else {
            checkBtn.disabled = true;
            betBtn.disabled = true;
            foldBtn.disabled = true;
            callBtn.disabled = true;
            dealBtn.disabled = true;
        }
        
        // Update scores
        document.getElementById('player-wins').textContent = this.playerWins;
        document.getElementById('bot-wins').textContent = this.botWins;
        document.getElementById('total-hands').textContent = this.totalHands;
    }
    
    setStatus(message) {
        document.getElementById('game-status').textContent = message;
    }
    
    addLogEntry(message, type = '') {
        this.gameLog.unshift({ message, type, time: new Date() });
        if (this.gameLog.length > 50) this.gameLog.pop();
        this.renderLog();
    }
    
    renderLog() {
        const container = document.getElementById('log-container');
        container.innerHTML = this.gameLog.map(entry => 
            `<div class="log-entry ${entry.type}">${entry.message}</div>`
        ).join('');
    }
    
    resetGame() {
        this.playerChips = 50;
        this.botChips = 50;
        this.pot = 0;
        this.playerCard = null;
        this.botCard = null;
        this.history = '';
        this.gameActive = false;
        this.playerWins = 0;
        this.botWins = 0;
        this.totalHands = 0;
        this.gameLog = [];
        
        this.updateDisplay();
        this.renderLog();
        this.setStatus('Click "Deal New Hand" to start');
    }
    
    showInfoModal() {
        document.getElementById('info-modal').classList.remove('hidden');
        document.getElementById('trained-iterations').textContent = this.iterations.toLocaleString();
        
        const strategies = this.cfr.getAllStrategies();
        
        // Render Player 1 strategies
        const p1Container = document.getElementById('strategy-p1');
        p1Container.innerHTML = '';
        
        for (const card of ['J', 'Q', 'K']) {
            const strat = strategies.player1[card];
            const row = document.createElement('div');
            row.className = 'strategy-row';
            
            const cardClass = card === 'J' ? 'jack' : card === 'Q' ? 'queen' : 'king';
            
            row.innerHTML = `
                <span class="strategy-card ${cardClass}">${card}</span>
                <div class="strategy-actions">
                    <div class="strategy-action">
                        <span>Check:</span>
                        <span>${(strat.initial[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div class="strategy-action">
                        <span>Bet:</span>
                        <span>${(strat.initial[1] * 100).toFixed(1)}%</span>
                    </div>
                    ${strat.afterBet ? `
                    <div class="strategy-action">
                        <span>Fold (vs bet):</span>
                        <span>${(strat.afterBet[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div class="strategy-action">
                        <span>Call (vs bet):</span>
                        <span>${(strat.afterBet[1] * 100).toFixed(1)}%</span>
                    </div>
                    ` : ''}
                </div>
            `;
            
            p1Container.appendChild(row);
        }
        
        // Render Player 2 strategies
        const p2Container = document.getElementById('strategy-p2');
        p2Container.innerHTML = '';
        
        for (const card of ['J', 'Q', 'K']) {
            const strat = strategies.player2[card];
            const row = document.createElement('div');
            row.className = 'strategy-row';
            
            const cardClass = card === 'J' ? 'jack' : card === 'Q' ? 'queen' : 'king';
            
            row.innerHTML = `
                <span class="strategy-card ${cardClass}">${card}</span>
                <div class="strategy-actions">
                    <div class="strategy-action">
                        <span>Check (vs check):</span>
                        <span>${(strat.afterPass[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div class="strategy-action">
                        <span>Bet (vs check):</span>
                        <span>${(strat.afterPass[1] * 100).toFixed(1)}%</span>
                    </div>
                    ${strat.afterBet ? `
                    <div class="strategy-action">
                        <span>Fold (vs bet):</span>
                        <span>${(strat.afterBet[0] * 100).toFixed(1)}%</span>
                    </div>
                    <div class="strategy-action">
                        <span>Call (vs bet):</span>
                        <span>${(strat.afterBet[1] * 100).toFixed(1)}%</span>
                    </div>
                    ` : ''}
                </div>
            `;
            
            p2Container.appendChild(row);
        }
    }
    
    hideInfoModal() {
        document.getElementById('info-modal').classList.add('hidden');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.game = new KuhnPokerGame();
});
