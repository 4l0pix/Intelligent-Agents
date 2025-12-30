/**
 * Roulette Game Controller
 * Demonstrates why all betting systems fail
 */

class RouletteGame {
    constructor() {
        this.wheel = new RouletteWheel();
        this.strategy = null;
        this.selectedStrategy = 'martingale';
        this.isSpinning = false;
        this.autoSpinning = false;
        this.bankrollHistory = [];
        
        this.initElements();
        this.initEventListeners();
        this.buildNumberGrid();
    }
    
    initElements() {
        // Screens
        this.strategyScreen = document.getElementById('strategy-screen');
        this.gameScreen = document.getElementById('game-screen');
        this.simulationScreen = document.getElementById('simulation-screen');
        
        // Strategy selection
        this.strategyCards = document.querySelectorAll('.strategy-card');
        this.playModeBtn = document.getElementById('play-mode-btn');
        this.simulateModeBtn = document.getElementById('simulate-mode-btn');
        
        // Game elements
        this.wheelEl = document.getElementById('wheel');
        this.ballEl = document.getElementById('ball');
        this.resultNumber = document.getElementById('result-number');
        this.bankrollEl = document.getElementById('bankroll');
        this.currentStrategyEl = document.getElementById('current-strategy');
        this.spinCountEl = document.getElementById('spin-count');
        this.currentBetAmount = document.getElementById('current-bet-amount');
        this.currentBetType = document.getElementById('current-bet-type');
        this.carlosComment = document.getElementById('carlos-comment');
        
        // Controls
        this.spinBtn = document.getElementById('spin-btn');
        this.autoSpinBtn = document.getElementById('auto-spin-btn');
        this.resetBtn = document.getElementById('reset-btn');
        this.backBtn = document.getElementById('back-btn');
        
        // Simulation
        this.runSimBtn = document.getElementById('run-sim-btn');
        this.runManyBtn = document.getElementById('run-many-btn');
        this.simBackBtn = document.getElementById('sim-back-btn');
        this.simComment = document.getElementById('sim-comment');
        this.multiSimResults = document.getElementById('multi-sim-results');
    }
    
    initEventListeners() {
        // Strategy selection
        this.strategyCards.forEach(card => {
            card.addEventListener('click', () => {
                this.strategyCards.forEach(c => c.classList.remove('selected'));
                card.classList.add('selected');
                this.selectedStrategy = card.dataset.strategy;
            });
        });
        
        // Mode buttons
        this.playModeBtn.addEventListener('click', () => this.startGame());
        this.simulateModeBtn.addEventListener('click', () => this.showSimulation());
        
        // Game controls
        this.spinBtn.addEventListener('click', () => this.spin());
        this.autoSpinBtn.addEventListener('click', () => this.autoSpin(100));
        this.resetBtn.addEventListener('click', () => this.resetGame());
        this.backBtn.addEventListener('click', () => this.showStrategyScreen());
        
        // Simulation controls
        this.runSimBtn.addEventListener('click', () => this.runSimulation());
        this.runManyBtn.addEventListener('click', () => this.runManySimulations(100));
        this.simBackBtn.addEventListener('click', () => this.showStrategyScreen());
        
        // Select first strategy by default
        this.strategyCards[0].classList.add('selected');
    }
    
    buildNumberGrid() {
        const grid = document.querySelector('.number-grid');
        if (!grid) return;
        
        const redNumbers = new Set([1, 3, 5, 7, 9, 12, 14, 16, 18, 19, 21, 23, 25, 27, 30, 32, 34, 36]);
        
        // Numbers are arranged in 3 rows of 12
        // Row 1: 3, 6, 9, 12, 15, 18, 21, 24, 27, 30, 33, 36
        // Row 2: 2, 5, 8, 11, 14, 17, 20, 23, 26, 29, 32, 35
        // Row 3: 1, 4, 7, 10, 13, 16, 19, 22, 25, 28, 31, 34
        
        for (let row = 0; row < 3; row++) {
            for (let col = 0; col < 12; col++) {
                const num = (col * 3) + (3 - row);
                const btn = document.createElement('button');
                btn.className = 'bet-cell';
                btn.classList.add(redNumbers.has(num) ? 'number-red' : 'number-black');
                btn.textContent = num;
                btn.dataset.bet = num;
                grid.appendChild(btn);
            }
        }
    }
    
    showScreen(screen) {
        this.strategyScreen.classList.remove('active');
        this.gameScreen.classList.remove('active');
        this.simulationScreen.classList.remove('active');
        screen.classList.add('active');
    }
    
    showStrategyScreen() {
        this.showScreen(this.strategyScreen);
        this.autoSpinning = false;
    }
    
    startGame() {
        this.strategy = createStrategy(this.selectedStrategy, 10, 1000);
        this.strategy.reset();
        this.bankrollHistory = [1000];
        
        this.updateDisplay();
        this.showScreen(this.gameScreen);
        
        this.currentStrategyEl.textContent = this.strategy.name;
        this.carlosComment.textContent = `Using ${this.strategy.name}. ${this.strategy.description}`;
    }
    
    showSimulation() {
        this.showScreen(this.simulationScreen);
        this.multiSimResults.classList.add('hidden');
        
        // Reset all sim cards
        document.querySelectorAll('.sim-card').forEach(card => {
            card.querySelector('.sim-bankroll span').textContent = '1000';
            card.querySelector('.sim-bankroll').className = 'sim-bankroll';
            card.querySelector('.sim-fill').style.width = '50%';
            card.querySelector('.sim-fill').classList.remove('bust');
            card.querySelector('.busts strong').textContent = '0';
            card.querySelector('.max-bet strong').textContent = '$0';
        });
    }
    
    updateDisplay() {
        if (!this.strategy) return;
        
        const stats = this.strategy.getStats();
        this.bankrollEl.textContent = `$${stats.bankroll.toFixed(0)}`;
        this.spinCountEl.textContent = stats.spins;
        
        const bet = this.strategy.getBet();
        this.currentBetAmount.textContent = `$${bet}`;
        
        const betType = this.strategy.getBetType();
        this.currentBetType.textContent = betType === 'james-bond' ? '(James Bond system)' : `on ${betType.toUpperCase()}`;
        
        // Update bankroll color
        if (stats.bankroll > this.strategy.initialBankroll) {
            this.bankrollEl.style.color = 'var(--green-accent)';
        } else if (stats.bankroll < this.strategy.initialBankroll) {
            this.bankrollEl.style.color = 'var(--red-accent)';
        } else {
            this.bankrollEl.style.color = 'var(--green-accent)';
        }
    }
    
    async spin() {
        if (this.isSpinning || !this.strategy) return;
        if (this.strategy.bankroll <= 0) {
            this.carlosComment.textContent = "Bankroll is gone. This is why gambling is a losing game.";
            return;
        }
        
        this.isSpinning = true;
        this.spinBtn.disabled = true;
        
        // Get current bet
        const betAmount = this.strategy.getBet();
        const betType = this.strategy.getBetType();
        
        // Spin animation
        this.ballEl.classList.add('visible');
        const rotations = 5 + Math.random() * 3;
        this.wheelEl.style.transform = `rotate(${rotations * 360}deg)`;
        
        // Wait for spin animation
        await this.delay(4000);
        
        // Get result
        const result = this.wheel.spin();
        
        // Show result
        this.resultNumber.textContent = result.number;
        this.resultNumber.style.color = result.color === 'red' ? 'var(--roulette-red)' : 
                                         result.color === 'black' ? '#fff' : 'var(--green-accent)';
        
        // Evaluate bet
        let won = false;
        let payout = 0;
        
        if (betType === 'james-bond') {
            const netResult = this.strategy.evaluateJamesBond(result);
            won = netResult > 0;
            payout = won ? netResult + betAmount : 0;
            // Adjust for James Bond special handling
            this.strategy.bankroll += netResult;
            this.strategy.spinCount++;
            this.strategy.history.push(this.strategy.bankroll);
        } else {
            const bet = { type: betType, amount: betAmount, value: betType };
            const evaluation = this.wheel.evaluateBet(bet, result);
            won = evaluation.won;
            payout = evaluation.payout;
            this.strategy.update(won, payout);
        }
        
        this.bankrollHistory.push(this.strategy.bankroll);
        
        // Update display
        this.updateDisplay();
        this.updateChart();
        
        // Carlos commentary
        const comment = this.strategy.getComment();
        if (comment) {
            this.carlosComment.textContent = comment;
        } else if (won) {
            this.carlosComment.textContent = `${result.number} ${result.color}. You won $${payout}. But remember: short-term wins don't change the math.`;
        } else {
            this.carlosComment.textContent = `${result.number} ${result.color}. Lost $${betAmount}. The house edge works silently but relentlessly.`;
        }
        
        if (this.strategy.bankroll <= 0) {
            this.carlosComment.textContent = `Bust! After ${this.strategy.spinCount} spins, the bankroll is gone. ${this.strategy.name} has failed - as it mathematically must.`;
        }
        
        this.isSpinning = false;
        this.spinBtn.disabled = false;
        this.ballEl.classList.remove('visible');
    }
    
    async autoSpin(count) {
        if (this.autoSpinning) {
            this.autoSpinning = false;
            this.autoSpinBtn.textContent = 'Auto (100 spins)';
            return;
        }
        
        this.autoSpinning = true;
        this.autoSpinBtn.textContent = 'Stop';
        
        for (let i = 0; i < count && this.autoSpinning && this.strategy.bankroll > 0; i++) {
            await this.spinFast();
            this.spinCountEl.textContent = this.strategy.spinCount;
            
            if (i % 10 === 0) {
                this.updateDisplay();
                this.updateChart();
                await this.delay(50);
            }
        }
        
        this.autoSpinning = false;
        this.autoSpinBtn.textContent = 'Auto (100 spins)';
        this.updateDisplay();
        this.updateChart();
        
        const stats = this.strategy.getStats();
        if (stats.busted) {
            this.carlosComment.textContent = `Busted after ${stats.spins} spins. Max bet reached: $${stats.maxBet}. This is the inevitable result.`;
        } else {
            this.carlosComment.textContent = `After ${stats.spins} spins: ${stats.profit >= 0 ? 'up' : 'down'} $${Math.abs(stats.profit).toFixed(0)} (${stats.profitPercent}%). The house edge is working.`;
        }
    }
    
    async spinFast() {
        if (!this.strategy || this.strategy.bankroll <= 0) return;
        
        const betAmount = this.strategy.getBet();
        const betType = this.strategy.getBetType();
        const result = this.wheel.spin();
        
        if (betType === 'james-bond') {
            const netResult = this.strategy.evaluateJamesBond(result);
            this.strategy.bankroll += netResult;
            this.strategy.spinCount++;
            this.strategy.history.push(this.strategy.bankroll);
        } else {
            const bet = { type: betType, amount: betAmount, value: betType };
            const evaluation = this.wheel.evaluateBet(bet, result);
            this.strategy.update(evaluation.won, evaluation.payout);
        }
        
        this.bankrollHistory.push(this.strategy.bankroll);
    }
    
    resetGame() {
        if (!this.strategy) return;
        this.strategy.reset();
        this.bankrollHistory = [1000];
        this.resultNumber.textContent = '-';
        this.resultNumber.style.color = 'var(--text-primary)';
        this.wheelEl.style.transform = 'rotate(0deg)';
        this.updateDisplay();
        this.updateChart();
        this.carlosComment.textContent = `Reset. ${this.strategy.description}`;
        this.autoSpinning = false;
        this.autoSpinBtn.textContent = 'Auto (100 spins)';
    }
    
    updateChart() {
        const canvas = document.getElementById('bankroll-chart');
        if (!canvas) return;
        
        const ctx = canvas.getContext('2d');
        const width = canvas.parentElement.clientWidth - 32;
        const height = 168;
        
        canvas.width = width;
        canvas.height = height;
        
        ctx.clearRect(0, 0, width, height);
        
        if (this.bankrollHistory.length < 2) return;
        
        const data = this.bankrollHistory;
        const max = Math.max(...data, 1000);
        const min = Math.min(...data, 0);
        const range = max - min || 1;
        
        // Draw starting line
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.2)';
        ctx.setLineDash([5, 5]);
        ctx.beginPath();
        const startY = height - ((1000 - min) / range) * height;
        ctx.moveTo(0, startY);
        ctx.lineTo(width, startY);
        ctx.stroke();
        ctx.setLineDash([]);
        
        // Draw bankroll line
        ctx.strokeStyle = data[data.length - 1] >= 1000 ? '#4ade80' : '#ef4444';
        ctx.lineWidth = 2;
        ctx.beginPath();
        
        for (let i = 0; i < data.length; i++) {
            const x = (i / (data.length - 1)) * width;
            const y = height - ((data[i] - min) / range) * height;
            
            if (i === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        }
        
        ctx.stroke();
        
        // Draw current value
        ctx.fillStyle = 'rgba(255, 255, 255, 0.6)';
        ctx.font = '12px sans-serif';
        ctx.textAlign = 'right';
        ctx.fillText(`$${data[data.length - 1].toFixed(0)}`, width - 5, 15);
    }
    
    async runSimulation() {
        const strategies = ['martingale', 'reverse-martingale', 'dalembert', 'fibonacci', 'flat', 'james-bond'];
        const spins = 1000;
        
        this.simComment.textContent = 'Running simulation...';
        
        for (const stratName of strategies) {
            const strat = createStrategy(stratName, 10, 1000);
            strat.reset();
            
            for (let i = 0; i < spins && strat.bankroll > 0; i++) {
                const betAmount = strat.getBet();
                const betType = strat.getBetType();
                const result = this.wheel.spin();
                
                if (betType === 'james-bond') {
                    const netResult = strat.evaluateJamesBond(result);
                    strat.bankroll += netResult;
                    strat.spinCount++;
                } else {
                    const bet = { type: betType, amount: betAmount, value: betType };
                    const evaluation = this.wheel.evaluateBet(bet, result);
                    strat.update(evaluation.won, evaluation.payout);
                }
            }
            
            // Update card
            const card = document.querySelector(`.sim-card[data-strategy="${stratName}"]`);
            if (card) {
                const stats = strat.getStats();
                const bankrollSpan = card.querySelector('.sim-bankroll span');
                const bankrollDiv = card.querySelector('.sim-bankroll');
                const fill = card.querySelector('.sim-fill');
                const busts = card.querySelector('.busts strong');
                const maxBet = card.querySelector('.max-bet strong');
                
                bankrollSpan.textContent = stats.bankroll.toFixed(0);
                bankrollDiv.classList.remove('positive', 'negative');
                bankrollDiv.classList.add(stats.bankroll >= 1000 ? 'positive' : 'negative');
                
                const fillPercent = Math.min(100, Math.max(0, (stats.bankroll / 2000) * 100));
                fill.style.width = `${fillPercent}%`;
                fill.classList.toggle('bust', stats.busted);
                
                busts.textContent = stats.busted ? '1' : '0';
                maxBet.textContent = `$${stats.maxBet}`;
            }
            
            await this.delay(100);
        }
        
        this.simComment.textContent = 'All strategies eventually lose. The house edge is inescapable. Run 100 simulations to see the statistical certainty.';
    }
    
    async runManySimulations(count) {
        const strategies = ['martingale', 'reverse-martingale', 'dalembert', 'fibonacci', 'flat', 'james-bond'];
        const spinsPerSim = 1000;
        const results = {};
        
        strategies.forEach(s => {
            results[s] = { totalBankroll: 0, busts: 0, maxBet: 0 };
        });
        
        this.simComment.textContent = `Running ${count} simulations...`;
        
        for (let sim = 0; sim < count; sim++) {
            for (const stratName of strategies) {
                const strat = createStrategy(stratName, 10, 1000);
                strat.reset();
                
                for (let i = 0; i < spinsPerSim && strat.bankroll > 0; i++) {
                    const betAmount = strat.getBet();
                    const betType = strat.getBetType();
                    const result = this.wheel.spin();
                    
                    if (betType === 'james-bond') {
                        const netResult = strat.evaluateJamesBond(result);
                        strat.bankroll += netResult;
                        strat.spinCount++;
                    } else {
                        const bet = { type: betType, amount: betAmount, value: betType };
                        const evaluation = this.wheel.evaluateBet(bet, result);
                        strat.update(evaluation.won, evaluation.payout);
                    }
                }
                
                const stats = strat.getStats();
                results[stratName].totalBankroll += stats.bankroll;
                if (stats.busted) results[stratName].busts++;
                if (stats.maxBet > results[stratName].maxBet) {
                    results[stratName].maxBet = stats.maxBet;
                }
            }
            
            if (sim % 10 === 0) {
                this.simComment.textContent = `Running simulation ${sim + 1}/${count}...`;
                await this.delay(0);
            }
        }
        
        // Show results
        this.multiSimResults.classList.remove('hidden');
        const grid = this.multiSimResults.querySelector('.multi-grid');
        grid.innerHTML = '';
        
        const strategyNames = {
            'martingale': 'Martingale',
            'reverse-martingale': 'Reverse Martingale',
            'dalembert': "D'Alembert",
            'fibonacci': 'Fibonacci',
            'flat': 'Flat Betting',
            'james-bond': 'James Bond'
        };
        
        strategies.forEach(stratName => {
            const r = results[stratName];
            const avgBankroll = r.totalBankroll / count;
            const avgProfit = avgBankroll - 1000;
            const bustRate = (r.busts / count * 100).toFixed(1);
            
            const item = document.createElement('div');
            item.className = 'multi-item';
            item.innerHTML = `
                <h4>${strategyNames[stratName]}</h4>
                <div class="value ${avgProfit >= 0 ? 'positive' : 'negative'}">$${avgBankroll.toFixed(0)}</div>
                <div class="bust-rate">${bustRate}% bust rate</div>
            `;
            grid.appendChild(item);
        });
        
        this.simComment.textContent = `After ${count} simulations of 1000 spins each: ALL strategies show negative expected value. The math is absolute.`;
    }
    
    delay(ms) {
        return new Promise(resolve => setTimeout(resolve, ms));
    }
}

// Initialize game
document.addEventListener('DOMContentLoaded', () => {
    window.game = new RouletteGame();
});
