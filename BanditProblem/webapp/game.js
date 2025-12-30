// Game Controller for Slot Machine Bandit Game

class SlotMachineGame {
    constructor() {
        this.agent = null;
        this.nMachines = 5;
        this.strategy = 'epsilon-greedy';
        this.epsilon = 0.1;
        this.temperature = 0.5;
        this.trainingRounds = 1000;
        
        // Game state
        this.playerBalance = 0;
        this.aiBalance = 0;
        this.totalPlays = 0;
        this.playerPlays = 0;
        this.aiPlays = 0;
        this.mode = 'player'; // 'player' or 'ai'
        this.isSpinning = false;
        this.gameLog = [];
        
        // Slot symbols
        this.symbols = ['🍒', '🍋', '🍊', '🍇', '💎', '7️⃣', '🍀'];
        
        this.initializeEventListeners();
    }
    
    initializeEventListeners() {
        // Machine count buttons
        document.querySelectorAll('.machine-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.machine-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.nMachines = parseInt(btn.dataset.count);
            });
        });
        
        // Strategy buttons
        document.querySelectorAll('.strategy-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.strategy-btn').forEach(b => b.classList.remove('selected'));
                btn.classList.add('selected');
                this.strategy = btn.dataset.strategy;
                this.updateParameterVisibility();
            });
        });
        
        // Epsilon slider
        const epsilonSlider = document.getElementById('epsilon-slider');
        const epsilonValue = document.getElementById('epsilon-value');
        epsilonSlider.addEventListener('input', () => {
            this.epsilon = parseFloat(epsilonSlider.value);
            epsilonValue.textContent = this.epsilon.toFixed(2);
        });
        
        // Temperature slider
        const tempSlider = document.getElementById('temp-slider');
        const tempValue = document.getElementById('temp-value');
        tempSlider.addEventListener('input', () => {
            this.temperature = parseFloat(tempSlider.value);
            tempValue.textContent = this.temperature.toFixed(2);
        });
        
        // Training rounds slider
        const roundsSlider = document.getElementById('rounds-slider');
        const roundsValue = document.getElementById('rounds-value');
        roundsSlider.addEventListener('input', () => {
            this.trainingRounds = parseInt(roundsSlider.value);
            roundsValue.textContent = this.trainingRounds.toLocaleString();
        });
        
        // Start training button
        document.getElementById('start-btn').addEventListener('click', () => this.startTraining());
        
        // Mode toggle
        document.querySelectorAll('.mode-btn').forEach(btn => {
            btn.addEventListener('click', () => {
                document.querySelectorAll('.mode-btn').forEach(b => b.classList.remove('active'));
                btn.classList.add('active');
                this.mode = btn.dataset.mode;
            });
        });
        
        // Reset button
        document.getElementById('reset-btn').addEventListener('click', () => this.resetGame());
        
        // Info button
        document.getElementById('info-btn').addEventListener('click', () => this.showInfoModal());
        
        // Modal close
        document.querySelector('.close-btn').addEventListener('click', () => this.hideInfoModal());
        document.getElementById('info-modal').addEventListener('click', (e) => {
            if (e.target.id === 'info-modal') this.hideInfoModal();
        });
    }
    
    updateParameterVisibility() {
        const epsilonGroup = document.getElementById('epsilon-group');
        const tempGroup = document.getElementById('temp-group');
        
        if (this.strategy === 'epsilon-greedy') {
            epsilonGroup.classList.remove('hidden');
            tempGroup.classList.add('hidden');
        } else {
            epsilonGroup.classList.add('hidden');
            tempGroup.classList.remove('hidden');
        }
    }
    
    async startTraining() {
        const startBtn = document.getElementById('start-btn');
        startBtn.disabled = true;
        startBtn.textContent = 'Training...';
        
        const params = {
            epsilon: this.epsilon,
            temperature: this.temperature
        };
        
        const progressBar = document.getElementById('progress-fill');
        const progressText = document.getElementById('progress-text');
        document.getElementById('progress-container').classList.remove('hidden');
        
        try {
            const result = await trainBandit(
                this.nMachines,
                this.strategy,
                params,
                this.trainingRounds,
                async (progress, current, total) => {
                    progressBar.style.width = `${progress}%`;
                    progressText.textContent = `Training: ${current.toLocaleString()} / ${total.toLocaleString()} rounds`;
                }
            );
            
            this.agent = result.agent;
            this.rewardHistory = result.rewardHistory;
            
            progressText.textContent = 'Training complete!';
            
            setTimeout(() => {
                this.showGameScreen();
            }, 500);
            
        } catch (error) {
            console.error('Training error:', error);
            progressText.textContent = 'Training failed!';
            startBtn.disabled = false;
            startBtn.textContent = 'Start Training';
        }
    }
    
    showGameScreen() {
        document.getElementById('training-screen').classList.add('hidden');
        document.getElementById('game-screen').classList.remove('hidden');
        this.createMachines();
        this.updateScoreboard();
    }
    
    createMachines() {
        const container = document.getElementById('machines-container');
        container.innerHTML = '';
        
        for (let i = 0; i < this.nMachines; i++) {
            const machine = document.createElement('div');
            machine.className = 'slot-machine';
            machine.dataset.index = i;
            
            // Random initial symbol
            const symbol = this.symbols[Math.floor(Math.random() * this.symbols.length)];
            
            machine.innerHTML = `
                <div class="machine-number">SLOT ${i + 1}</div>
                <div class="reels">
                    <span class="reel-display">${symbol}</span>
                </div>
                <div class="payout-display">-</div>
                <div class="machine-plays">Plays: 0</div>
            `;
            
            machine.addEventListener('click', () => this.playMachine(i));
            container.appendChild(machine);
        }
    }
    
    async playMachine(index) {
        if (this.isSpinning) return;
        
        this.isSpinning = true;
        const machine = document.querySelector(`.slot-machine[data-index="${index}"]`);
        
        // Spin animation
        machine.classList.add('spinning');
        
        // Animate reel display
        const reelDisplay = machine.querySelector('.reel-display');
        const spinDuration = 500;
        const spinInterval = 50;
        let elapsed = 0;
        
        const spinTimer = setInterval(() => {
            reelDisplay.textContent = this.symbols[Math.floor(Math.random() * this.symbols.length)];
            elapsed += spinInterval;
            if (elapsed >= spinDuration) {
                clearInterval(spinTimer);
            }
        }, spinInterval);
        
        await new Promise(resolve => setTimeout(resolve, spinDuration));
        
        machine.classList.remove('spinning');
        
        // Get reward
        const reward = this.agent.getReward(index);
        const roundedReward = Math.round(reward * 100) / 100;
        
        // Update agent's knowledge (if in AI mode, this happens automatically)
        // For player mode, we still update so AI can learn from player's choices
        const oldQ = [...this.agent.Q];
        this.agent.updateQ(index, reward);
        
        // Final symbol based on reward
        let finalSymbol;
        if (reward > 1.5) {
            finalSymbol = '💎';
        } else if (reward > 0.5) {
            finalSymbol = '7️⃣';
        } else if (reward > 0) {
            finalSymbol = '🍀';
        } else if (reward > -0.5) {
            finalSymbol = '🍊';
        } else {
            finalSymbol = '🍋';
        }
        
        reelDisplay.textContent = finalSymbol;
        
        // Update payout display
        const payoutDisplay = machine.querySelector('.payout-display');
        if (roundedReward >= 0) {
            payoutDisplay.textContent = `+${roundedReward.toFixed(2)}`;
            payoutDisplay.className = 'payout-display win';
        } else {
            payoutDisplay.textContent = roundedReward.toFixed(2);
            payoutDisplay.className = 'payout-display lose';
        }
        
        // Update plays count
        const playsDisplay = machine.querySelector('.machine-plays');
        playsDisplay.textContent = `Plays: ${this.agent.N[index]}`;
        
        // Update balances
        if (this.mode === 'player') {
            this.playerBalance += roundedReward;
            this.playerPlays++;
            this.addLogEntry(`You played Slot ${index + 1}: ${roundedReward >= 0 ? '+' : ''}${roundedReward.toFixed(2)}`, roundedReward >= 0, false);
        } else {
            this.aiBalance += roundedReward;
            this.aiPlays++;
            this.addLogEntry(`Carlos played Slot ${index + 1}: ${roundedReward >= 0 ? '+' : ''}${roundedReward.toFixed(2)}`, roundedReward >= 0, true);
        }
        
        this.totalPlays++;
        this.updateScoreboard();
        
        // Clear payout after a moment
        setTimeout(() => {
            payoutDisplay.textContent = '-';
            payoutDisplay.className = 'payout-display';
        }, 2000);
        
        this.isSpinning = false;
        
        // If in AI mode, let AI continue playing
        if (this.mode === 'ai') {
            setTimeout(() => this.aiPlay(), 800);
        }
    }
    
    async aiPlay() {
        if (this.mode !== 'ai' || this.isSpinning) return;
        
        // AI selects action based on its strategy
        const action = this.agent.selectAction();
        
        // Highlight selected machine briefly
        const machines = document.querySelectorAll('.slot-machine');
        machines.forEach(m => m.classList.remove('selected'));
        machines[action].classList.add('selected');
        
        await new Promise(resolve => setTimeout(resolve, 300));
        
        await this.playMachine(action);
        
        machines[action].classList.remove('selected');
    }
    
    updateScoreboard() {
        document.getElementById('player-balance').textContent = this.playerBalance.toFixed(2);
        document.getElementById('ai-balance').textContent = this.aiBalance.toFixed(2);
        document.getElementById('total-plays').textContent = this.totalPlays;
        
        // Color coding
        const playerBalanceEl = document.getElementById('player-balance');
        const aiBalanceEl = document.getElementById('ai-balance');
        
        playerBalanceEl.className = `score-value ${this.playerBalance >= 0 ? 'positive' : 'negative'}`;
        aiBalanceEl.className = `score-value ${this.aiBalance >= 0 ? 'positive' : 'negative'}`;
    }
    
    addLogEntry(message, isWin, isAI) {
        const entry = { message, isWin, isAI, time: new Date() };
        this.gameLog.unshift(entry);
        
        // Keep only last 50 entries
        if (this.gameLog.length > 50) {
            this.gameLog.pop();
        }
        
        this.renderLog();
    }
    
    renderLog() {
        const container = document.getElementById('log-container');
        container.innerHTML = this.gameLog.map(entry => {
            let className = 'log-entry';
            if (entry.isWin) className += ' win';
            if (entry.isAI) className += ' ai';
            return `<div class="${className}">${entry.message}</div>`;
        }).join('');
    }
    
    resetGame() {
        this.playerBalance = 0;
        this.aiBalance = 0;
        this.totalPlays = 0;
        this.playerPlays = 0;
        this.aiPlays = 0;
        this.gameLog = [];
        
        // Reset agent's play counts but keep Q-values
        this.agent.N = new Array(this.nMachines).fill(0);
        
        // Reset machine displays
        this.createMachines();
        this.updateScoreboard();
        this.renderLog();
    }
    
    showInfoModal() {
        const modal = document.getElementById('info-modal');
        modal.classList.remove('hidden');
        
        const stats = this.agent.getStats();
        
        // Strategy info
        document.getElementById('strategy-type').textContent = 
            stats.strategy === 'epsilon-greedy' ? 'ε-Greedy' : 'Softmax';
        document.getElementById('strategy-param').textContent = 
            stats.strategy === 'epsilon-greedy' 
                ? `ε = ${stats.epsilon}` 
                : `τ = ${stats.temperature}`;
        
        // Machine stats
        const statsContainer = document.getElementById('machine-stats');
        statsContainer.innerHTML = '';
        
        for (let i = 0; i < this.nMachines; i++) {
            const isBest = i === stats.bestEstimated;
            const statItem = document.createElement('div');
            statItem.className = `stat-item ${isBest ? 'best' : ''}`;
            statItem.innerHTML = `
                <div class="stat-machine">${this.symbols[i % this.symbols.length]}</div>
                <div class="stat-value">${stats.Q[i].toFixed(3)}</div>
                <span class="stat-label">Slot ${i + 1}</span>
                <span class="stat-label">Plays: ${stats.N[i]}</span>
            `;
            statsContainer.appendChild(statItem);
        }
        
        // Render chart
        this.renderChart();
    }
    
    renderChart() {
        const canvas = document.getElementById('reward-chart');
        const ctx = canvas.getContext('2d');
        
        // Set canvas size
        canvas.width = canvas.parentElement.clientWidth - 30;
        canvas.height = 120;
        
        const width = canvas.width;
        const height = canvas.height;
        const padding = 20;
        
        // Clear canvas
        ctx.clearRect(0, 0, width, height);
        
        if (!this.rewardHistory || this.rewardHistory.length < 2) {
            ctx.fillStyle = 'rgba(255, 255, 255, 0.3)';
            ctx.font = '12px Roboto';
            ctx.textAlign = 'center';
            ctx.fillText('Not enough data', width / 2, height / 2);
            return;
        }
        
        const data = this.rewardHistory;
        const minY = Math.min(...data.map(d => d.avgReward));
        const maxY = Math.max(...data.map(d => d.avgReward));
        const rangeY = maxY - minY || 1;
        
        // Draw grid lines
        ctx.strokeStyle = 'rgba(255, 255, 255, 0.1)';
        ctx.lineWidth = 1;
        for (let i = 0; i <= 4; i++) {
            const y = padding + (i / 4) * (height - 2 * padding);
            ctx.beginPath();
            ctx.moveTo(padding, y);
            ctx.lineTo(width - padding, y);
            ctx.stroke();
        }
        
        // Draw line
        ctx.beginPath();
        ctx.strokeStyle = '#4ade80';
        ctx.lineWidth = 2;
        
        data.forEach((point, index) => {
            const x = padding + (index / (data.length - 1)) * (width - 2 * padding);
            const y = height - padding - ((point.avgReward - minY) / rangeY) * (height - 2 * padding);
            
            if (index === 0) {
                ctx.moveTo(x, y);
            } else {
                ctx.lineTo(x, y);
            }
        });
        ctx.stroke();
        
        // Draw glow
        ctx.shadowColor = '#4ade80';
        ctx.shadowBlur = 10;
        ctx.stroke();
        ctx.shadowBlur = 0;
        
        // Labels
        ctx.fillStyle = 'rgba(255, 255, 255, 0.5)';
        ctx.font = '10px Roboto';
        ctx.textAlign = 'left';
        ctx.fillText(maxY.toFixed(2), 2, padding + 10);
        ctx.fillText(minY.toFixed(2), 2, height - padding);
        ctx.textAlign = 'center';
        ctx.fillText('Training Rounds', width / 2, height - 2);
    }
    
    hideInfoModal() {
        document.getElementById('info-modal').classList.add('hidden');
    }
}

// Initialize game when DOM is loaded
document.addEventListener('DOMContentLoaded', () => {
    window.game = new SlotMachineGame();
});
