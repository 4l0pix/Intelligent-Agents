// Texas Hold'em Poker Engine
// Hand evaluation and Monte Carlo simulation for AI decision making

class PokerEngine {
    constructor() {
        this.SUITS = ['hearts', 'diamonds', 'clubs', 'spades'];
        this.RANKS = ['2', '3', '4', '5', '6', '7', '8', '9', 'T', 'J', 'Q', 'K', 'A'];
        this.SUIT_SYMBOLS = { hearts: 'H', diamonds: 'D', clubs: 'C', spades: 'S' };
    }

    // Create a fresh deck
    createDeck() {
        const deck = [];
        for (const suit of this.SUITS) {
            for (const rank of this.RANKS) {
                deck.push({ rank, suit });
            }
        }
        return deck;
    }

    // Shuffle deck using Fisher-Yates
    shuffleDeck(deck) {
        const shuffled = [...deck];
        for (let i = shuffled.length - 1; i > 0; i--) {
            const j = Math.floor(Math.random() * (i + 1));
            [shuffled[i], shuffled[j]] = [shuffled[j], shuffled[i]];
        }
        return shuffled;
    }

    // Get numeric rank value (2=2, ..., A=14)
    rankValue(rank) {
        const values = { '2': 2, '3': 3, '4': 4, '5': 5, '6': 6, '7': 7, '8': 8, '9': 9, 'T': 10, 'J': 11, 'Q': 12, 'K': 13, 'A': 14 };
        return values[rank];
    }

    // Evaluate a 5-card hand, returns { rank: number, tiebreakers: [] }
    // Rank: 1=High Card, 2=Pair, 3=Two Pair, 4=Three of a Kind, 5=Straight, 6=Flush, 7=Full House, 8=Four of a Kind, 9=Straight Flush
    evaluateHand(cards) {
        const ranks = cards.map(c => this.rankValue(c.rank)).sort((a, b) => b - a);
        const suits = cards.map(c => c.suit);
        
        // Count ranks
        const rankCounts = {};
        for (const r of ranks) {
            rankCounts[r] = (rankCounts[r] || 0) + 1;
        }
        const counts = Object.values(rankCounts).sort((a, b) => b - a);
        
        // Check flush
        const isFlush = suits.every(s => s === suits[0]);
        
        // Check straight
        let isStraight = false;
        const uniqueRanks = [...new Set(ranks)].sort((a, b) => b - a);
        
        if (uniqueRanks.length >= 5) {
            // Check for regular straight
            for (let i = 0; i <= uniqueRanks.length - 5; i++) {
                if (uniqueRanks[i] - uniqueRanks[i + 4] === 4) {
                    isStraight = true;
                    break;
                }
            }
            // Check for wheel (A-2-3-4-5)
            if (!isStraight && uniqueRanks.includes(14) && uniqueRanks.includes(2) && 
                uniqueRanks.includes(3) && uniqueRanks.includes(4) && uniqueRanks.includes(5)) {
                isStraight = true;
            }
        }
        
        // Determine hand rank
        if (isFlush && isStraight) {
            return { rank: 9, name: 'Straight Flush', tiebreakers: ranks };
        }
        if (counts[0] === 4) {
            return { rank: 8, name: 'Four of a Kind', tiebreakers: this.getTiebreakers(rankCounts, [4, 1]) };
        }
        if (counts[0] === 3 && counts[1] === 2) {
            return { rank: 7, name: 'Full House', tiebreakers: this.getTiebreakers(rankCounts, [3, 2]) };
        }
        if (isFlush) {
            return { rank: 6, name: 'Flush', tiebreakers: ranks };
        }
        if (isStraight) {
            return { rank: 5, name: 'Straight', tiebreakers: ranks };
        }
        if (counts[0] === 3) {
            return { rank: 4, name: 'Three of a Kind', tiebreakers: this.getTiebreakers(rankCounts, [3, 1, 1]) };
        }
        if (counts[0] === 2 && counts[1] === 2) {
            return { rank: 3, name: 'Two Pair', tiebreakers: this.getTiebreakers(rankCounts, [2, 2, 1]) };
        }
        if (counts[0] === 2) {
            return { rank: 2, name: 'Pair', tiebreakers: this.getTiebreakers(rankCounts, [2, 1, 1, 1]) };
        }
        return { rank: 1, name: 'High Card', tiebreakers: ranks };
    }

    // Get tiebreakers in order of importance
    getTiebreakers(rankCounts, pattern) {
        const result = [];
        const entries = Object.entries(rankCounts).map(([r, c]) => ({ rank: parseInt(r), count: c }));
        
        for (const count of pattern) {
            const matching = entries.filter(e => e.count === count).sort((a, b) => b.rank - a.rank);
            for (const m of matching) {
                if (!result.includes(m.rank)) {
                    result.push(m.rank);
                    break;
                }
            }
        }
        return result;
    }

    // Find best 5-card hand from 7 cards
    bestHand(cards) {
        if (cards.length < 5) return null;
        
        let best = null;
        const combos = this.combinations(cards, 5);
        
        for (const combo of combos) {
            const hand = this.evaluateHand(combo);
            if (!best || this.compareHands(hand, best) > 0) {
                best = hand;
                best.cards = combo;
            }
        }
        return best;
    }

    // Generate all combinations of size k
    combinations(arr, k) {
        const result = [];
        const combo = [];
        
        const generate = (start) => {
            if (combo.length === k) {
                result.push([...combo]);
                return;
            }
            for (let i = start; i < arr.length; i++) {
                combo.push(arr[i]);
                generate(i + 1);
                combo.pop();
            }
        };
        
        generate(0);
        return result;
    }

    // Compare two hands: returns positive if hand1 wins, negative if hand2 wins, 0 for tie
    compareHands(hand1, hand2) {
        if (hand1.rank !== hand2.rank) {
            return hand1.rank - hand2.rank;
        }
        // Compare tiebreakers
        for (let i = 0; i < Math.min(hand1.tiebreakers.length, hand2.tiebreakers.length); i++) {
            if (hand1.tiebreakers[i] !== hand2.tiebreakers[i]) {
                return hand1.tiebreakers[i] - hand2.tiebreakers[i];
            }
        }
        return 0;
    }

    // Monte Carlo simulation to estimate win probability
    // Returns win rate between 0 and 1
    monteCarloWinRate(holeCards, communityCards, numSimulations = 500) {
        let wins = 0;
        let ties = 0;
        
        // Cards that are in play
        const usedCards = new Set([...holeCards, ...communityCards].map(c => `${c.rank}${c.suit}`));
        
        // Remaining deck
        const remainingDeck = this.createDeck().filter(c => !usedCards.has(`${c.rank}${c.suit}`));
        
        const cardsNeeded = 5 - communityCards.length;
        
        for (let i = 0; i < numSimulations; i++) {
            const shuffled = this.shuffleDeck(remainingDeck);
            
            // Deal opponent hole cards
            const oppHole = [shuffled[0], shuffled[1]];
            
            // Complete community cards
            const fullCommunity = [...communityCards, ...shuffled.slice(2, 2 + cardsNeeded)];
            
            // Evaluate hands
            const myHand = this.bestHand([...holeCards, ...fullCommunity]);
            const oppHand = this.bestHand([...oppHole, ...fullCommunity]);
            
            const result = this.compareHands(myHand, oppHand);
            if (result > 0) wins++;
            else if (result === 0) ties++;
        }
        
        return (wins + ties * 0.5) / numSimulations;
    }

    // Get hand strength category (for display)
    getHandStrengthCategory(winRate) {
        if (winRate >= 0.85) return 'Monster';
        if (winRate >= 0.70) return 'Strong';
        if (winRate >= 0.55) return 'Good';
        if (winRate >= 0.40) return 'Marginal';
        if (winRate >= 0.25) return 'Weak';
        return 'Very Weak';
    }

    // Get preflop hand strength (Chen formula approximation)
    preflopStrength(holeCards) {
        const r1 = this.rankValue(holeCards[0].rank);
        const r2 = this.rankValue(holeCards[1].rank);
        const suited = holeCards[0].suit === holeCards[1].suit;
        const paired = r1 === r2;
        
        const high = Math.max(r1, r2);
        const low = Math.min(r1, r2);
        const gap = high - low - 1;
        
        // Base score from high card
        let score = high;
        if (high === 14) score = 10; // Ace
        else if (high >= 10) score = (high - 10) * 1.5 + 5;
        else score = high / 2;
        
        // Pair bonus
        if (paired) {
            score = Math.max(5, score * 2);
        }
        
        // Suited bonus
        if (suited) score += 2;
        
        // Gap penalty
        if (gap === 1) score -= 1;
        else if (gap === 2) score -= 2;
        else if (gap === 3) score -= 4;
        else if (gap >= 4) score -= 5;
        
        // Connector bonus
        if (gap <= 0 && !paired && low >= 10) score += 1;
        
        return Math.max(0, Math.min(20, score)) / 20; // Normalize to 0-1
    }
}

// AI Decision Making
class PokerAI {
    constructor(difficulty = 'medium') {
        this.engine = new PokerEngine();
        this.setDifficulty(difficulty);
    }

    setDifficulty(difficulty) {
        this.difficulty = difficulty;
        
        // Adjust parameters based on difficulty
        switch (difficulty) {
            case 'easy':
                this.bluffFrequency = 0.05;
                this.callThreshold = 0.35;
                this.raiseThreshold = 0.65;
                this.simulations = 200;
                break;
            case 'medium':
                this.bluffFrequency = 0.15;
                this.callThreshold = 0.30;
                this.raiseThreshold = 0.55;
                this.simulations = 400;
                break;
            case 'hard':
                this.bluffFrequency = 0.20;
                this.callThreshold = 0.25;
                this.raiseThreshold = 0.50;
                this.simulations = 600;
                break;
        }
    }

    // Make a decision given the game state
    makeDecision(holeCards, communityCards, pot, toCall, playerChips, botChips, isPreflop) {
        // Calculate hand strength
        let handStrength;
        if (isPreflop || communityCards.length === 0) {
            handStrength = this.engine.preflopStrength(holeCards);
        } else {
            handStrength = this.engine.monteCarloWinRate(holeCards, communityCards, this.simulations);
        }
        
        // Calculate pot odds
        const potOdds = toCall > 0 ? toCall / (pot + toCall) : 0;
        
        // Decision logic
        const effectiveStack = Math.min(playerChips, botChips);
        const potCommitment = toCall / effectiveStack;
        
        // Bluff consideration
        const shouldBluff = Math.random() < this.bluffFrequency && toCall === 0;
        
        // Very strong hand - raise/bet
        if (handStrength >= this.raiseThreshold || shouldBluff) {
            if (toCall === 0) {
                // Bet
                const betSize = this.calculateBetSize(pot, effectiveStack, handStrength);
                return { action: 'bet', amount: betSize };
            } else {
                // Raise
                const raiseSize = this.calculateRaiseSize(pot, toCall, effectiveStack, handStrength);
                if (raiseSize > toCall) {
                    return { action: 'raise', amount: raiseSize };
                }
                return { action: 'call', amount: toCall };
            }
        }
        
        // Decent hand - call if good pot odds
        if (handStrength >= this.callThreshold || handStrength > potOdds) {
            if (toCall === 0) {
                // Check or small bet
                if (handStrength > 0.5 && Math.random() < 0.4) {
                    const betSize = this.calculateBetSize(pot, effectiveStack, handStrength * 0.7);
                    return { action: 'bet', amount: betSize };
                }
                return { action: 'check', amount: 0 };
            } else {
                // Call if not too expensive
                if (potCommitment < 0.5 || handStrength > 0.6) {
                    return { action: 'call', amount: toCall };
                }
            }
        }
        
        // Weak hand
        if (toCall === 0) {
            return { action: 'check', amount: 0 };
        }
        
        return { action: 'fold', amount: 0 };
    }

    calculateBetSize(pot, effectiveStack, strength) {
        // Bet between 1/3 pot and full pot based on hand strength
        const minBet = Math.max(2, Math.floor(pot * 0.33));
        const maxBet = Math.min(pot, effectiveStack);
        
        const bet = Math.floor(minBet + (maxBet - minBet) * strength);
        return Math.min(bet, effectiveStack);
    }

    calculateRaiseSize(pot, toCall, effectiveStack, strength) {
        // Raise to 2.5x-4x the call amount based on strength
        const multiplier = 2.5 + strength * 1.5;
        const raiseAmount = Math.floor(toCall * multiplier);
        
        return Math.min(raiseAmount, effectiveStack);
    }
}

// Export
window.PokerEngine = PokerEngine;
window.PokerAI = PokerAI;
