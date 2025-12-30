// Counterfactual Regret Minimization (CFR) for Kuhn Poker
// This is a simplified implementation that demonstrates how CFR learns optimal poker strategy

class CFRTrainer {
    constructor() {
        // Cards: J=0, Q=1, K=2
        this.JACK = 0;
        this.QUEEN = 1;
        this.KING = 2;
        this.cardNames = ['J', 'Q', 'K'];
        
        // Information sets store regrets and strategy sums
        // Key format: "card:history" e.g., "J:" "Q:b" "K:cb"
        this.regretSum = {};
        this.strategySum = {};
        
        // Actions: 0 = check/fold, 1 = bet/call
        this.NUM_ACTIONS = 2;
    }
    
    // Get current strategy from regrets (regret matching)
    getStrategy(infoSet) {
        const regrets = this.regretSum[infoSet] || [0, 0];
        const strategy = [0, 0];
        let normalizingSum = 0;
        
        // Only consider positive regrets
        for (let a = 0; a < this.NUM_ACTIONS; a++) {
            strategy[a] = Math.max(0, regrets[a]);
            normalizingSum += strategy[a];
        }
        
        // Normalize to get probability distribution
        for (let a = 0; a < this.NUM_ACTIONS; a++) {
            if (normalizingSum > 0) {
                strategy[a] /= normalizingSum;
            } else {
                // If no positive regrets, use uniform strategy
                strategy[a] = 1.0 / this.NUM_ACTIONS;
            }
        }
        
        return strategy;
    }
    
    // Get average strategy (what we've actually played over time)
    getAverageStrategy(infoSet) {
        const strategySum = this.strategySum[infoSet] || [0, 0];
        const avgStrategy = [0, 0];
        let normalizingSum = strategySum[0] + strategySum[1];
        
        for (let a = 0; a < this.NUM_ACTIONS; a++) {
            if (normalizingSum > 0) {
                avgStrategy[a] = strategySum[a] / normalizingSum;
            } else {
                avgStrategy[a] = 1.0 / this.NUM_ACTIONS;
            }
        }
        
        return avgStrategy;
    }
    
    // Main CFR recursive function
    cfr(cards, history, reachProbs, currentPlayer) {
        const plays = history.length;
        const opponent = 1 - currentPlayer;
        
        // Check for terminal states
        if (plays >= 2) {
            const terminalValue = this.getTerminalValue(cards, history);
            if (terminalValue !== null) {
                return terminalValue;
            }
        }
        
        // Get information set for current player
        const infoSet = this.cardNames[cards[currentPlayer]] + ':' + history;
        
        // Get current strategy
        const strategy = this.getStrategy(infoSet);
        
        // Initialize action utilities
        const actionUtility = [0, 0];
        let nodeUtility = 0;
        
        // For each action
        for (let a = 0; a < this.NUM_ACTIONS; a++) {
            const actionChar = a === 0 ? 'p' : 'b'; // p = pass/fold, b = bet/call
            const nextHistory = history + actionChar;
            
            // Update reach probabilities
            const newReachProbs = [...reachProbs];
            newReachProbs[currentPlayer] *= strategy[a];
            
            // Recursively compute utility
            actionUtility[a] = -this.cfr(cards, nextHistory, newReachProbs, opponent);
            
            // Accumulate node utility
            nodeUtility += strategy[a] * actionUtility[a];
        }
        
        // Update regrets and strategy sum
        if (!this.regretSum[infoSet]) {
            this.regretSum[infoSet] = [0, 0];
        }
        if (!this.strategySum[infoSet]) {
            this.strategySum[infoSet] = [0, 0];
        }
        
        const opponentReachProb = reachProbs[opponent];
        
        for (let a = 0; a < this.NUM_ACTIONS; a++) {
            // Regret = how much better this action would have been
            const regret = actionUtility[a] - nodeUtility;
            this.regretSum[infoSet][a] += opponentReachProb * regret;
            
            // Track strategy for averaging
            this.strategySum[infoSet][a] += reachProbs[currentPlayer] * strategy[a];
        }
        
        return nodeUtility;
    }
    
    // Determine terminal value if game is over
    getTerminalValue(cards, history) {
        const plays = history.length;
        
        if (plays < 2) return null;
        
        const lastAction = history[plays - 1];
        const secondLastAction = history[plays - 2];
        
        // Terminal states in Kuhn Poker:
        // pp = both pass, showdown
        // bp = bet then fold, bettor wins ante
        // bb = bet then call, showdown for 2
        // pbp = pass, bet, fold - bettor wins ante
        // pbb = pass, bet, call - showdown for 2
        
        if (history === 'pp') {
            // Both passed, showdown for antes (1 each)
            return cards[0] > cards[1] ? 1 : -1;
        }
        
        if (lastAction === 'p' && secondLastAction === 'b') {
            // Fold after bet - last player to bet wins
            return 1;
        }
        
        if (history === 'bb' || history === 'pbb') {
            // Showdown after bet and call (pot = 4, each put in 2)
            const isP1 = history === 'bb' ? 0 : 1;
            const winner = cards[0] > cards[1] ? 0 : 1;
            return isP1 === winner ? 2 : -2;
        }
        
        if (history === 'pbp') {
            // Player 1 passed, player 2 bet, player 1 folded
            return -1;
        }
        
        return null; // Not terminal
    }
    
    // Run training iterations
    async train(iterations, progressCallback) {
        let totalUtility = 0;
        const batchSize = Math.max(1, Math.floor(iterations / 100));
        
        for (let i = 0; i < iterations; i++) {
            // Shuffle cards
            const cards = this.shuffleCards();
            
            // Run CFR from both perspectives
            totalUtility += this.cfr(cards, '', [1, 1], 0);
            
            // Progress callback
            if (i % batchSize === 0 && progressCallback) {
                const progress = Math.round((i / iterations) * 100);
                await progressCallback(progress, i, iterations);
                await new Promise(resolve => setTimeout(resolve, 0));
            }
        }
        
        if (progressCallback) {
            await progressCallback(100, iterations, iterations);
        }
        
        return totalUtility / iterations;
    }
    
    // Shuffle deck and deal 2 cards
    shuffleCards() {
        const deck = [this.JACK, this.QUEEN, this.KING];
        // Fisher-Yates shuffle
        for (let i = deck.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [deck[i], deck[j]] = [deck[j], deck[i]];
        }
        return [deck[0], deck[1]];
    }
    
    // Get action based on strategy (for playing)
    getAction(card, history, isPlayer1) {
        const infoSet = this.cardNames[card] + ':' + history;
        const strategy = this.getAverageStrategy(infoSet);
        
        // Sample from strategy
        const r = Math.random();
        return r < strategy[0] ? 0 : 1;
    }
    
    // Get all strategies for display
    getAllStrategies() {
        const strategies = {
            player1: {},
            player2: {}
        };
        
        // Player 1 positions (acting first)
        for (let card = 0; card < 3; card++) {
            const cardName = this.cardNames[card];
            
            // First action (empty history)
            const infoSet1 = cardName + ':';
            strategies.player1[cardName] = {
                initial: this.getAverageStrategy(infoSet1),
                afterBet: null
            };
            
            // After opponent bets (history = 'pb')
            const infoSet2 = cardName + ':pb';
            if (this.strategySum[infoSet2]) {
                strategies.player1[cardName].afterBet = this.getAverageStrategy(infoSet2);
            }
        }
        
        // Player 2 positions (acting second)
        for (let card = 0; card < 3; card++) {
            const cardName = this.cardNames[card];
            
            // After opponent passes
            const infoSetPass = cardName + ':p';
            strategies.player2[cardName] = {
                afterPass: this.getAverageStrategy(infoSetPass),
                afterBet: null
            };
            
            // After opponent bets
            const infoSetBet = cardName + ':b';
            if (this.strategySum[infoSetBet]) {
                strategies.player2[cardName].afterBet = this.getAverageStrategy(infoSetBet);
            }
        }
        
        return strategies;
    }
}

// Export for use in game.js
window.CFRTrainer = CFRTrainer;
