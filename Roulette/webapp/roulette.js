/**
 * Roulette Engine
 * Simulates American Roulette (0, 00, 1-36)
 */

class RouletteWheel {
    constructor() {
        // American roulette: 0, 00, 1-36
        this.numbers = ['0', '00', ...Array.from({length: 36}, (_, i) => String(i + 1))];
        
        // Red numbers
        this.redNumbers = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);
        
        // Black numbers
        this.blackNumbers = new Set([2, 4, 6, 8, 10, 11, 13, 15, 17, 20, 22, 24, 26, 28, 29, 31, 33, 35]);
        
        // House edge for American roulette
        this.houseEdge = 5.26; // percentage
    }
    
    /**
     * Spin the wheel and return the result
     */
    spin() {
        const index = Math.floor(Math.random() * this.numbers.length);
        const result = this.numbers[index];
        return {
            number: result,
            color: this.getColor(result),
            isZero: result === '0' || result === '00'
        };
    }
    
    /**
     * Get the color of a number
     */
    getColor(number) {
        if (number === '0' || number === '00') return 'green';
        const num = parseInt(number);
        if (this.redNumbers.has(num)) return 'red';
        if (this.blackNumbers.has(num)) return 'black';
        return 'unknown';
    }
    
    /**
     * Check if a bet wins and calculate payout
     */
    evaluateBet(bet, result) {
        const num = result.number;
        const numInt = num === '0' || num === '00' ? null : parseInt(num);
        
        switch (bet.type) {
            // Straight up (single number) - 35:1
            case 'straight':
                if (num === bet.value) {
                    return { won: true, payout: bet.amount * 35 };
                }
                break;
                
            // Red/Black - 1:1
            case 'red':
                if (result.color === 'red') {
                    return { won: true, payout: bet.amount };
                }
                break;
            case 'black':
                if (result.color === 'black') {
                    return { won: true, payout: bet.amount };
                }
                break;
                
            // Odd/Even - 1:1
            case 'odd':
                if (numInt && numInt % 2 === 1) {
                    return { won: true, payout: bet.amount };
                }
                break;
            case 'even':
                if (numInt && numInt % 2 === 0) {
                    return { won: true, payout: bet.amount };
                }
                break;
                
            // Low (1-18) / High (19-36) - 1:1
            case '1-18':
                if (numInt && numInt >= 1 && numInt <= 18) {
                    return { won: true, payout: bet.amount };
                }
                break;
            case '19-36':
                if (numInt && numInt >= 19 && numInt <= 36) {
                    return { won: true, payout: bet.amount };
                }
                break;
                
            // Dozens - 2:1
            case '1st12':
                if (numInt && numInt >= 1 && numInt <= 12) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
            case '2nd12':
                if (numInt && numInt >= 13 && numInt <= 24) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
            case '3rd12':
                if (numInt && numInt >= 25 && numInt <= 36) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
                
            // Columns - 2:1
            case 'column1':
                if (numInt && numInt % 3 === 1) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
            case 'column2':
                if (numInt && numInt % 3 === 2) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
            case 'column3':
                if (numInt && numInt % 3 === 0) {
                    return { won: true, payout: bet.amount * 2 };
                }
                break;
        }
        
        return { won: false, payout: 0 };
    }
    
    /**
     * Calculate expected value for a bet
     * For American roulette, all bets have EV of -5.26%
     */
    getExpectedValue(betType, betAmount) {
        // For even money bets (red/black, odd/even, high/low):
        // P(win) = 18/38 = 47.37%
        // EV = (18/38 * 1) + (20/38 * -1) = -2/38 = -5.26%
        return -betAmount * (this.houseEdge / 100);
    }
    
    /**
     * Calculate probability of winning for a bet type
     */
    getWinProbability(betType) {
        const totalPockets = 38; // American roulette
        
        switch (betType) {
            case 'straight':
                return 1 / totalPockets; // 2.63%
            case 'red':
            case 'black':
            case 'odd':
            case 'even':
            case '1-18':
            case '19-36':
                return 18 / totalPockets; // 47.37%
            case '1st12':
            case '2nd12':
            case '3rd12':
            case 'column1':
            case 'column2':
            case 'column3':
                return 12 / totalPockets; // 31.58%
            default:
                return 0;
        }
    }
}

// Export for use in other files
window.RouletteWheel = RouletteWheel;
