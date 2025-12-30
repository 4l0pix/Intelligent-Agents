// Texas Hold'em Game Controller

class TexasHoldemGame {
    constructor() {
        this.engine = new PokerEngine();
        this.ai = null;
        this.difficulty = 'medium';
        this.startingChips = 100;
        
        // Game state
        this.playerChips = 100;
        this.botChips = 100;
        this.pot = 0;
        this.playerHole = [];
        this.botHole = [];
        this.community = [];
        this.deck = [];
        
        // Betting state
        this.currentBet = 0;
        this.playerBet = 0;
        this.botBet = 0;
        this.isPlayerTurn = true;
        this.gamePhase = 'idle'; // idle, preflop, flop, turn, river, showdown
        this.lastAction = { player: '', bot: '' };
        
        // Blinds
        this.smallBlind = 1;
        this.bigBlind = 2;
        this.dealerIsPlayer = true; // Player has button
        
        // Stats
        this.playerWins = 0;
        this.botWins = 0;
        this.totalHands = 0;
        
        this.initializeEventListeners();
    }

    initializeEventListeners() {
        // Difficulty buttons
        document.querySelectorAll('.difficulty-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.difficulty-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.difficulty = btn.dataset.difficulty;
            });
        });

        // Chips slider
        const chipsSlider = document.getElementById('chips-slider');
        const chipsValue = document.getElementById('chips-value');
        chipsSlider.addEventListener('input', () => {
            this.startingChips = parseInt(chipsSlider.value);
            chipsValue.textContent = this.startingChips;
        });

        // Start button
        document.getElementById('start-btn').addEventListener('click', () => this.startGame());

        // Action buttons
        document.getElementById('fold-btn').addEventListener('click', () => this.playerAction('fold'));
        document.getElementById('check-btn').addEventListener('click', () => this.playerAction('check'));
        document.getElementById('call-btn').addEventListener('click', () => this.playerAction('call'));
        document.getElementById('bet-btn').addEventListener('click', () => this.showBetControls('bet'));
        document.getElementById('raise-btn').addEventListener('click', () => this.showBetControls('raise'));
        document.getElementById('allin-btn').addEventListener('click', () => this.playerAction('allin'));
        document.getElementById('deal-btn').addEventListener('click', () => this.dealNewHand());

        // Bet controls
        const betSlider = document.getElementById('bet-slider');
        const betAmount = document.getElementById('bet-amount');
        betSlider.addEventListener('input', () => {
            betAmount.textContent = betSlider.value;
        });
        document.getElementById('confirm-bet').addEventListener('click', () => this.confirmBet());
        document.getElementById('cancel-bet').addEventListener('click', () => this.hideBetControls());

        // Reset
        document.getElementById('reset-btn').addEventListener('click', () => this.resetGame());

        // Info modal
        document.getElementById('info-btn').addEventListener('click', () => this.showInfoModal());
        document.querySelector('.close-btn').addEventListener('click', () => this.hideInfoModal());
        document.getElementById('info-modal').addEventListener('click', (e) => {
            if (e.target.id === 'info-modal') this.hideInfoModal();
        });
    }

    startGame() {
        this.ai = new PokerAI(this.difficulty);
        this.playerChips = this.startingChips;
        this.botChips = this.startingChips;
        
        document.getElementById('training-screen').classList.add('hidden');
        document.getElementById('game-screen').classList.remove('hidden');
        document.getElementById('difficulty-display').textContent = 
            this.difficulty.charAt(0).toUpperCase() + this.difficulty.slice(1);
        
        this.updateDisplay();
    }

    dealNewHand() {
        if (this.playerChips <= 0 || this.botChips <= 0) {
            this.setStatus('Game over! Reset to play again.');
            return;
        }

        // Reset hand state
        this.pot = 0;
        this.currentBet = 0;
        this.playerBet = 0;
        this.botBet = 0;
        this.community = [];
        this.lastAction = { player: '', bot: '' };

        // Alternate dealer
        this.dealerIsPlayer = !this.dealerIsPlayer;

        // Create and shuffle deck
        this.deck = this.engine.shuffleDeck(this.engine.createDeck());

        // Deal hole cards
        this.playerHole = [this.deck.pop(), this.deck.pop()];
        this.botHole = [this.deck.pop(), this.deck.pop()];

        // Post blinds
        if (this.dealerIsPlayer) {
            // Player is dealer (small blind), bot is big blind
            this.postBlind(true, this.smallBlind);
            this.postBlind(false, this.bigBlind);
            this.isPlayerTurn = true; // Dealer acts first preflop in heads-up
        } else {
            // Bot is dealer (small blind), player is big blind
            this.postBlind(false, this.smallBlind);
            this.postBlind(true, this.bigBlind);
            this.isPlayerTurn = false;
        }

        this.currentBet = this.bigBlind;
        this.gamePhase = 'preflop';
        this.totalHands++;

        this.updateDisplay();
        this.setStatus(this.isPlayerTurn ? 'Your turn' : 'Bot is thinking...');

        if (!this.isPlayerTurn) {
            setTimeout(() => this.botTurn(), 1000);
        }
    }

    postBlind(isPlayer, amount) {
        const actualAmount = Math.min(amount, isPlayer ? this.playerChips : this.botChips);
        
        if (isPlayer) {
            this.playerChips -= actualAmount;
            this.playerBet = actualAmount;
        } else {
            this.botChips -= actualAmount;
            this.botBet = actualAmount;
        }
        this.pot += actualAmount;
    }

    playerAction(action) {
        if (!this.isPlayerTurn || this.gamePhase === 'idle' || this.gamePhase === 'showdown') return;

        const toCall = this.currentBet - this.playerBet;

        switch (action) {
            case 'fold':
                this.lastAction.player = 'FOLD';
                this.endHand(false);
                return;

            case 'check':
                if (toCall > 0) return; // Can't check if there's a bet
                this.lastAction.player = 'CHECK';
                break;

            case 'call':
                const callAmount = Math.min(toCall, this.playerChips);
                this.playerChips -= callAmount;
                this.playerBet += callAmount;
                this.pot += callAmount;
                this.lastAction.player = `CALL ${callAmount}`;
                break;

            case 'allin':
                const allinAmount = this.playerChips;
                this.pot += allinAmount;
                this.playerBet += allinAmount;
                this.playerChips = 0;
                this.currentBet = Math.max(this.currentBet, this.playerBet);
                this.lastAction.player = `ALL IN ${allinAmount}`;
                break;
        }

        this.updateDisplay();
        this.afterPlayerAction();
    }

    showBetControls(type) {
        this.pendingBetType = type;
        const betControls = document.getElementById('bet-controls');
        const betSlider = document.getElementById('bet-slider');
        
        const minBet = type === 'bet' ? this.bigBlind : this.currentBet * 2;
        const maxBet = this.playerChips;
        
        betSlider.min = Math.min(minBet, maxBet);
        betSlider.max = maxBet;
        betSlider.value = Math.min(minBet * 2, maxBet);
        document.getElementById('bet-amount').textContent = betSlider.value;
        
        betControls.classList.remove('hidden');
    }

    hideBetControls() {
        document.getElementById('bet-controls').classList.add('hidden');
    }

    confirmBet() {
        const amount = parseInt(document.getElementById('bet-slider').value);
        
        this.playerChips -= amount;
        this.playerBet += amount;
        this.pot += amount;
        this.currentBet = this.playerBet;
        
        this.lastAction.player = `${this.pendingBetType.toUpperCase()} ${amount}`;
        
        this.hideBetControls();
        this.updateDisplay();
        this.afterPlayerAction();
    }

    afterPlayerAction() {
        // Check if betting round is complete
        if (this.isBettingComplete()) {
            this.advancePhase();
        } else {
            this.isPlayerTurn = false;
            this.setStatus('Bot is thinking...');
            setTimeout(() => this.botTurn(), 1000);
        }
    }

    botTurn() {
        if (this.isPlayerTurn || this.gamePhase === 'idle' || this.gamePhase === 'showdown') return;

        const toCall = this.currentBet - this.botBet;
        const isPreflop = this.gamePhase === 'preflop';
        
        const decision = this.ai.makeDecision(
            this.botHole,
            this.community,
            this.pot,
            toCall,
            this.playerChips,
            this.botChips,
            isPreflop
        );

        switch (decision.action) {
            case 'fold':
                this.lastAction.bot = 'FOLD';
                this.endHand(true);
                return;

            case 'check':
                this.lastAction.bot = 'CHECK';
                break;

            case 'call':
                const callAmount = Math.min(toCall, this.botChips);
                this.botChips -= callAmount;
                this.botBet += callAmount;
                this.pot += callAmount;
                this.lastAction.bot = `CALL ${callAmount}`;
                break;

            case 'bet':
            case 'raise':
                const betAmount = Math.min(decision.amount, this.botChips);
                this.botChips -= betAmount;
                this.botBet += betAmount;
                this.pot += betAmount;
                this.currentBet = this.botBet;
                this.lastAction.bot = `${decision.action.toUpperCase()} ${betAmount}`;
                break;
        }

        this.updateDisplay();

        // Check if betting round is complete
        if (this.isBettingComplete()) {
            setTimeout(() => this.advancePhase(), 800);
        } else {
            this.isPlayerTurn = true;
            this.setStatus('Your turn');
            this.updateActionButtons();
        }
    }

    isBettingComplete() {
        // Betting is complete when both players have acted and bets are equal
        // Or when a player is all-in
        if (this.playerChips === 0 || this.botChips === 0) return true;
        
        const bothActed = this.lastAction.player !== '' && this.lastAction.bot !== '';
        const betsEqual = this.playerBet === this.botBet;
        
        // Special case: if someone just bet/raised, other player needs to act
        if (this.lastAction.player.includes('BET') || this.lastAction.player.includes('RAISE') ||
            this.lastAction.player.includes('ALL IN')) {
            return this.lastAction.bot !== '' && betsEqual;
        }
        if (this.lastAction.bot.includes('BET') || this.lastAction.bot.includes('RAISE')) {
            return this.lastAction.player !== '' && betsEqual;
        }
        
        return bothActed && betsEqual;
    }

    advancePhase() {
        // Reset betting for new round
        this.playerBet = 0;
        this.botBet = 0;
        this.currentBet = 0;
        this.lastAction = { player: '', bot: '' };

        switch (this.gamePhase) {
            case 'preflop':
                // Deal flop
                this.deck.pop(); // Burn
                this.community.push(this.deck.pop(), this.deck.pop(), this.deck.pop());
                this.gamePhase = 'flop';
                this.setStatus('Flop dealt');
                break;

            case 'flop':
                // Deal turn
                this.deck.pop(); // Burn
                this.community.push(this.deck.pop());
                this.gamePhase = 'turn';
                this.setStatus('Turn dealt');
                break;

            case 'turn':
                // Deal river
                this.deck.pop(); // Burn
                this.community.push(this.deck.pop());
                this.gamePhase = 'river';
                this.setStatus('River dealt');
                break;

            case 'river':
                // Showdown
                this.showdown();
                return;
        }

        this.updateDisplay();

        // In heads-up, button acts first postflop
        this.isPlayerTurn = this.dealerIsPlayer;
        
        setTimeout(() => {
            if (!this.isPlayerTurn) {
                this.botTurn();
            } else {
                this.setStatus('Your turn');
                this.updateActionButtons();
            }
        }, 1000);
    }

    showdown() {
        this.gamePhase = 'showdown';
        
        // Reveal bot cards
        this.updateDisplay(true);

        const playerHand = this.engine.bestHand([...this.playerHole, ...this.community]);
        const botHand = this.engine.bestHand([...this.botHole, ...this.community]);
        
        const result = this.engine.compareHands(playerHand, botHand);

        setTimeout(() => {
            if (result > 0) {
                this.playerWins++;
                this.playerChips += this.pot;
                this.setStatus(`You win with ${playerHand.name}! +${this.pot} chips`);
            } else if (result < 0) {
                this.botWins++;
                this.botChips += this.pot;
                this.setStatus(`Bot wins with ${botHand.name}. -${this.pot - this.playerBet} chips`);
            } else {
                // Split pot
                const half = Math.floor(this.pot / 2);
                this.playerChips += half;
                this.botChips += this.pot - half;
                this.setStatus(`Split pot! Both have ${playerHand.name}`);
            }

            this.pot = 0;
            this.updateDisplay(true);
        }, 1500);
    }

    endHand(playerWins) {
        this.gamePhase = 'showdown';
        
        if (playerWins) {
            this.playerWins++;
            this.playerChips += this.pot;
            this.setStatus(`Bot folds! You win ${this.pot} chips`);
        } else {
            this.botWins++;
            this.botChips += this.pot;
            this.setStatus(`You fold. Bot wins ${this.pot} chips`);
        }

        this.pot = 0;
        this.updateDisplay();
    }

    updateDisplay(showBotCards = false) {
        // Update chips
        document.getElementById('player-chips').textContent = this.playerChips;
        document.getElementById('bot-chips').textContent = this.botChips;
        document.getElementById('pot-value').textContent = this.pot;

        // Update scores
        document.getElementById('player-wins').textContent = this.playerWins;
        document.getElementById('bot-wins').textContent = this.botWins;
        document.getElementById('total-hands').textContent = this.totalHands;

        // Update player cards
        this.renderCard('player-card-1', this.playerHole[0]);
        this.renderCard('player-card-2', this.playerHole[1]);

        // Update bot cards
        if (showBotCards || this.gamePhase === 'showdown') {
            this.renderCard('bot-card-1', this.botHole[0]);
            this.renderCard('bot-card-2', this.botHole[1]);
        } else {
            this.renderCardBack('bot-card-1');
            this.renderCardBack('bot-card-2');
        }

        // Update community cards
        for (let i = 1; i <= 5; i++) {
            const card = this.community[i - 1];
            this.renderCard(`comm-${i}`, card);
        }

        // Update actions
        document.getElementById('player-action').textContent = this.lastAction.player;
        document.getElementById('bot-action').textContent = this.lastAction.bot;

        this.updateActionButtons();
    }

    renderCard(elementId, card) {
        const el = document.getElementById(elementId);
        if (!card) {
            el.className = 'card empty';
            el.innerHTML = '';
            return;
        }

        const suitClass = card.suit;
        const symbol = this.engine.SUIT_SYMBOLS[card.suit];
        
        el.className = `card ${suitClass}`;
        el.innerHTML = `
            <span class="rank">${card.rank}</span>
            <span class="suit">${symbol}</span>
            <span class="rank-bottom">${card.rank}</span>
        `;
    }

    renderCardBack(elementId) {
        const el = document.getElementById(elementId);
        if (this.gamePhase === 'idle') {
            el.className = 'card empty';
            el.innerHTML = '';
        } else {
            el.className = 'card back';
            el.innerHTML = '';
        }
    }

    updateActionButtons() {
        const toCall = this.currentBet - this.playerBet;
        const canAct = this.isPlayerTurn && this.gamePhase !== 'idle' && this.gamePhase !== 'showdown';
        
        const foldBtn = document.getElementById('fold-btn');
        const checkBtn = document.getElementById('check-btn');
        const callBtn = document.getElementById('call-btn');
        const betBtn = document.getElementById('bet-btn');
        const raiseBtn = document.getElementById('raise-btn');
        const allinBtn = document.getElementById('allin-btn');
        const dealBtn = document.getElementById('deal-btn');

        // Reset all
        foldBtn.disabled = true;
        checkBtn.disabled = true;
        callBtn.disabled = true;
        betBtn.disabled = true;
        raiseBtn.disabled = true;
        allinBtn.disabled = true;
        
        checkBtn.classList.add('hidden');
        callBtn.classList.add('hidden');
        betBtn.classList.add('hidden');
        raiseBtn.classList.add('hidden');

        if (!canAct) {
            dealBtn.disabled = this.gamePhase !== 'idle' && this.gamePhase !== 'showdown';
            return;
        }

        dealBtn.disabled = true;
        foldBtn.disabled = false;
        allinBtn.disabled = this.playerChips <= 0;

        if (toCall === 0) {
            checkBtn.classList.remove('hidden');
            checkBtn.disabled = false;
            betBtn.classList.remove('hidden');
            betBtn.disabled = this.playerChips <= 0;
        } else {
            callBtn.classList.remove('hidden');
            callBtn.disabled = false;
            callBtn.textContent = `Call ${Math.min(toCall, this.playerChips)}`;
            raiseBtn.classList.remove('hidden');
            raiseBtn.disabled = this.playerChips <= toCall;
        }
    }

    setStatus(message) {
        document.getElementById('game-status').textContent = message;
    }

    resetGame() {
        this.playerChips = this.startingChips;
        this.botChips = this.startingChips;
        this.playerWins = 0;
        this.botWins = 0;
        this.totalHands = 0;
        this.gamePhase = 'idle';
        this.pot = 0;
        this.playerHole = [];
        this.botHole = [];
        this.community = [];
        this.lastAction = { player: '', bot: '' };

        this.updateDisplay();
        this.setStatus('Click "Deal New Hand" to start');
    }

    showInfoModal() {
        document.getElementById('info-modal').classList.remove('hidden');
    }

    hideInfoModal() {
        document.getElementById('info-modal').classList.add('hidden');
    }
}

// Initialize
document.addEventListener('DOMContentLoaded', () => {
    window.game = new TexasHoldemGame();
});
