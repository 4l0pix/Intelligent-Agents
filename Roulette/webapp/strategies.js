/**
 * Betting Strategies
 * All proven to lose in the long run due to house edge
 */

class BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        this.baseBet = baseBet;
        this.initialBankroll = bankroll;
        this.bankroll = bankroll;
        this.currentBet = baseBet;
        this.history = [];
        this.spinCount = 0;
        this.maxBetReached = baseBet;
        this.bustCount = 0;
        
        // Strategy-specific state
        this.consecutiveLosses = 0;
        this.consecutiveWins = 0;
        this.fibSequence = [1, 1, 2, 3, 5, 8, 13, 21, 34, 55, 89, 144, 233, 377, 610];
        this.fibIndex = 0;
    }
    
    /**
     * Reset the strategy
     */
    reset() {
        this.bankroll = this.initialBankroll;
        this.currentBet = this.baseBet;
        this.history = [this.initialBankroll];
        this.spinCount = 0;
        this.maxBetReached = this.baseBet;
        this.bustCount = 0;
        this.consecutiveLosses = 0;
        this.consecutiveWins = 0;
        this.fibIndex = 0;
    }
    
    /**
     * Get the current bet amount (to be overridden by specific strategies)
     */
    getBet() {
        return Math.min(this.currentBet, this.bankroll);
    }
    
    /**
     * Get the bet type (what to bet on)
     */
    getBetType() {
        return 'red'; // Most strategies bet on even money
    }
    
    /**
     * Update strategy after a result
     */
    update(won, payout) {
        if (won) {
            this.bankroll += payout;
            this.consecutiveWins++;
            this.consecutiveLosses = 0;
        } else {
            this.bankroll -= this.currentBet;
            this.consecutiveLosses++;
            this.consecutiveWins = 0;
        }
        
        this.spinCount++;
        this.history.push(this.bankroll);
        
        // Check for bust
        if (this.bankroll <= 0) {
            this.bustCount++;
            return false; // Can't continue
        }
        
        // Calculate next bet
        this.calculateNextBet(won);
        
        // Track max bet
        if (this.currentBet > this.maxBetReached) {
            this.maxBetReached = this.currentBet;
        }
        
        return true; // Can continue
    }
    
    /**
     * Calculate the next bet (to be overridden)
     */
    calculateNextBet(won) {
        // Base implementation does nothing
    }
    
    /**
     * Get strategy statistics
     */
    getStats() {
        return {
            bankroll: this.bankroll,
            profit: this.bankroll - this.initialBankroll,
            profitPercent: ((this.bankroll - this.initialBankroll) / this.initialBankroll * 100).toFixed(2),
            spins: this.spinCount,
            maxBet: this.maxBetReached,
            busted: this.bankroll <= 0
        };
    }
}

/**
 * Martingale: Double bet after each loss
 * "The strategy that guarantees catastrophic loss"
 */
class MartingaleStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = 'Martingale';
        this.description = 'Double bet after each loss to recover losses. Catastrophic failure inevitable.';
    }
    
    calculateNextBet(won) {
        if (won) {
            this.currentBet = this.baseBet;
        } else {
            this.currentBet *= 2;
        }
    }
    
    getComment() {
        if (this.consecutiveLosses >= 5) {
            return `${this.consecutiveLosses} losses in a row! Bet is now $${this.currentBet}. This is how Martingale destroys bankrolls.`;
        }
        if (this.currentBet > this.bankroll) {
            return `Can't afford the next bet ($${this.currentBet}). Martingale has failed - as it always does eventually.`;
        }
        return '';
    }
}

/**
 * Reverse Martingale (Paroli): Double bet after each win
 * "The illusion of riding hot streaks"
 */
class ReverseMartingaleStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = 'Reverse Martingale';
        this.description = 'Double bet after each win. Gives back all winnings on first loss.';
        this.maxWinStreak = 3; // Reset after 3 wins
    }
    
    calculateNextBet(won) {
        if (won && this.consecutiveWins < this.maxWinStreak) {
            this.currentBet *= 2;
        } else {
            this.currentBet = this.baseBet;
        }
    }
    
    getComment() {
        if (this.consecutiveWins >= 2) {
            return `${this.consecutiveWins} wins in a row! Betting $${this.currentBet}. One loss erases it all.`;
        }
        return '';
    }
}

/**
 * D'Alembert: Increase by 1 unit after loss, decrease by 1 after win
 * "The slow bleed"
 */
class DAlembertStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = "D'Alembert";
        this.description = 'Add 1 unit after loss, subtract 1 after win. Slower but still loses.';
        this.unit = baseBet;
    }
    
    calculateNextBet(won) {
        if (won) {
            this.currentBet = Math.max(this.unit, this.currentBet - this.unit);
        } else {
            this.currentBet += this.unit;
        }
    }
    
    getComment() {
        if (this.currentBet > this.baseBet * 5) {
            return `Bet has grown to $${this.currentBet}. D'Alembert is slower than Martingale, but the destination is the same.`;
        }
        return '';
    }
}

/**
 * Fibonacci: Follow the Fibonacci sequence for bet sizes
 * "Mathematically elegant, financially doomed"
 */
class FibonacciStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = 'Fibonacci';
        this.description = 'Follow Fibonacci sequence for bets. Elegant mathematics, certain failure.';
        this.fibIndex = 0;
    }
    
    calculateNextBet(won) {
        if (won) {
            // Move back 2 positions in the sequence
            this.fibIndex = Math.max(0, this.fibIndex - 2);
        } else {
            // Move forward 1 position
            this.fibIndex = Math.min(this.fibSequence.length - 1, this.fibIndex + 1);
        }
        this.currentBet = this.baseBet * this.fibSequence[this.fibIndex];
    }
    
    getComment() {
        if (this.fibIndex >= 8) {
            return `Fibonacci position ${this.fibIndex + 1}: betting $${this.currentBet}. The sequence grows fast.`;
        }
        return '';
    }
}

/**
 * Flat Betting: Same bet every time
 * "The honest approach - still loses"
 */
class FlatBettingStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = 'Flat Betting';
        this.description = 'Same bet every spin. Honest approach that clearly shows the house edge.';
    }
    
    calculateNextBet(won) {
        this.currentBet = this.baseBet; // Always the same
    }
    
    getComment() {
        const expectedLoss = (this.spinCount * this.baseBet * 0.0526).toFixed(2);
        const actualProfit = this.bankroll - this.initialBankroll;
        return `After ${this.spinCount} spins, expected loss: $${expectedLoss}. Actual: $${actualProfit.toFixed(2)}`;
    }
}

/**
 * James Bond: Cover most of the table
 * "Looks clever, but math doesn't care about coverage"
 */
class JamesBondStrategy extends BettingStrategy {
    constructor(baseBet = 10, bankroll = 1000) {
        super(baseBet, bankroll);
        this.name = 'James Bond';
        this.description = 'Bet on high numbers, six-line, and zero. Covers 25/37 numbers but still loses.';
        // Standard James Bond bet is $200 total:
        // $140 on 19-36, $50 on 13-18, $10 on 0
        this.totalBet = baseBet * 20; // $200 with $10 base
    }
    
    getBet() {
        return Math.min(this.totalBet, this.bankroll);
    }
    
    getBetType() {
        return 'james-bond'; // Special multi-bet
    }
    
    calculateNextBet(won) {
        // James Bond uses flat betting for the system
        this.currentBet = this.totalBet;
    }
    
    /**
     * Evaluate James Bond bet
     * Returns net result (positive for win, negative for loss)
     */
    evaluateJamesBond(result) {
        const num = result.number;
        const numInt = num === '0' || num === '00' ? 0 : parseInt(num);
        
        // Bet breakdown: $140 on 19-36, $50 on 13-18, $10 on 0
        const bet1936 = this.baseBet * 14; // $140
        const bet1318 = this.baseBet * 5;  // $50
        const bet0 = this.baseBet;          // $10
        
        let payout = 0;
        const totalBet = bet1936 + bet1318 + bet0;
        
        if (numInt >= 19 && numInt <= 36) {
            payout = bet1936; // 1:1 payout
        } else if (numInt >= 13 && numInt <= 18) {
            payout = bet1318 * 5; // 5:1 payout for six-line
        } else if (num === '0') {
            payout = bet0 * 35; // 35:1 payout for straight
        }
        // 00 or 1-12 = total loss
        
        return payout - totalBet;
    }
    
    getComment() {
        return `James Bond covers 25 of 38 numbers (66%), but the payout structure still gives the house a 5.26% edge.`;
    }
}

/**
 * Factory function to create strategies
 */
function createStrategy(name, baseBet = 10, bankroll = 1000) {
    switch (name) {
        case 'martingale':
            return new MartingaleStrategy(baseBet, bankroll);
        case 'reverse-martingale':
            return new ReverseMartingaleStrategy(baseBet, bankroll);
        case 'dalembert':
            return new DAlembertStrategy(baseBet, bankroll);
        case 'fibonacci':
            return new FibonacciStrategy(baseBet, bankroll);
        case 'flat':
            return new FlatBettingStrategy(baseBet, bankroll);
        case 'james-bond':
            return new JamesBondStrategy(baseBet, bankroll);
        default:
            return new FlatBettingStrategy(baseBet, bankroll);
    }
}

// Export
window.createStrategy = createStrategy;
window.BettingStrategy = BettingStrategy;
