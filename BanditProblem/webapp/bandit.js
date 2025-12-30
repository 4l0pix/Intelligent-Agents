// Bandit Algorithm Implementation
// Based on Multi-Armed Bandit Problem with ε-Greedy and Softmax strategies

class BanditAgent {
    constructor(nMachines, strategy, params) {
        this.nMachines = nMachines;
        this.strategy = strategy; // 'epsilon-greedy' or 'softmax'
        this.epsilon = params.epsilon || 0.1;
        this.temperature = params.temperature || 0.5;
        
        // Q-values (estimated value for each machine)
        this.Q = new Array(nMachines).fill(0);
        // Number of times each machine was played
        this.N = new Array(nMachines).fill(0);
        // Total reward from each machine
        this.totalReward = new Array(nMachines).fill(0);
        
        // True reward distributions (mean and std for each machine)
        // These are the "hidden" probabilities the agent learns about
        this.trueMeans = [];
        this.trueStds = [];
        
        this.initializeMachines();
    }
    
    initializeMachines() {
        // Each machine has a different true reward distribution
        // Mean rewards between -1 and 3
        // Some machines are better than others
        for (let i = 0; i < this.nMachines; i++) {
            // Random mean between -0.5 and 2.5
            this.trueMeans.push(Math.random() * 3 - 0.5);
            // Standard deviation between 0.5 and 1.5
            this.trueStds.push(Math.random() + 0.5);
        }
    }
    
    // Get reward from pulling a machine (based on true distribution)
    getReward(machineIndex) {
        // Sample from normal distribution with machine's true parameters
        const mean = this.trueMeans[machineIndex];
        const std = this.trueStds[machineIndex];
        return this.randomNormal(mean, std);
    }
    
    // Box-Muller transform for normal distribution
    randomNormal(mean = 0, std = 1) {
        const u1 = Math.random();
        const u2 = Math.random();
        const z = Math.sqrt(-2 * Math.log(u1)) * Math.cos(2 * Math.PI * u2);
        return mean + std * z;
    }
    
    // Select action using ε-greedy strategy
    selectActionEpsilonGreedy() {
        if (Math.random() < this.epsilon) {
            // Explore: random action
            return Math.floor(Math.random() * this.nMachines);
        } else {
            // Exploit: choose best known action
            return this.getBestMachine();
        }
    }
    
    // Select action using Softmax strategy
    selectActionSoftmax() {
        // Calculate softmax probabilities
        const probs = this.getSoftmaxProbabilities();
        
        // Sample from probability distribution
        const random = Math.random();
        let cumSum = 0;
        for (let i = 0; i < this.nMachines; i++) {
            cumSum += probs[i];
            if (random < cumSum) {
                return i;
            }
        }
        return this.nMachines - 1;
    }
    
    getSoftmaxProbabilities() {
        // Softmax with temperature
        const expQ = this.Q.map(q => Math.exp(q / this.temperature));
        const sumExpQ = expQ.reduce((a, b) => a + b, 0);
        return expQ.map(e => e / sumExpQ);
    }
    
    // Select action based on current strategy
    selectAction() {
        if (this.strategy === 'epsilon-greedy') {
            return this.selectActionEpsilonGreedy();
        } else {
            return this.selectActionSoftmax();
        }
    }
    
    // Update Q-values after receiving reward
    updateQ(machineIndex, reward) {
        this.N[machineIndex]++;
        this.totalReward[machineIndex] += reward;
        
        // Incremental mean update: Q = Q + (1/n)(r - Q)
        const n = this.N[machineIndex];
        this.Q[machineIndex] += (reward - this.Q[machineIndex]) / n;
    }
    
    // Get the machine with highest estimated value
    getBestMachine() {
        let bestIndex = 0;
        let bestValue = this.Q[0];
        for (let i = 1; i < this.nMachines; i++) {
            if (this.Q[i] > bestValue) {
                bestValue = this.Q[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
    
    // Get the true best machine (for comparison)
    getTrueBestMachine() {
        let bestIndex = 0;
        let bestValue = this.trueMeans[0];
        for (let i = 1; i < this.nMachines; i++) {
            if (this.trueMeans[i] > bestValue) {
                bestValue = this.trueMeans[i];
                bestIndex = i;
            }
        }
        return bestIndex;
    }
    
    // Run one step: select action, get reward, update
    step() {
        const action = this.selectAction();
        const reward = this.getReward(action);
        this.updateQ(action, reward);
        return { action, reward };
    }
    
    // Get statistics for display
    getStats() {
        return {
            Q: [...this.Q],
            N: [...this.N],
            totalReward: [...this.totalReward],
            trueMeans: [...this.trueMeans],
            bestEstimated: this.getBestMachine(),
            bestTrue: this.getTrueBestMachine(),
            strategy: this.strategy,
            epsilon: this.epsilon,
            temperature: this.temperature
        };
    }
}

// Training function with progress callback
async function trainBandit(nMachines, strategy, params, trainingRounds, progressCallback) {
    const agent = new BanditAgent(nMachines, strategy, params);
    
    const batchSize = Math.max(1, Math.floor(trainingRounds / 100));
    let totalReward = 0;
    const rewardHistory = [];
    
    for (let round = 0; round < trainingRounds; round++) {
        const { action, reward } = agent.step();
        totalReward += reward;
        
        // Store reward history for charting (sample every batch)
        if (round % batchSize === 0) {
            rewardHistory.push({
                round: round,
                avgReward: totalReward / (round + 1)
            });
        }
        
        // Update progress
        if (round % batchSize === 0 && progressCallback) {
            const progress = Math.round((round / trainingRounds) * 100);
            await progressCallback(progress, round, trainingRounds);
            // Small delay for UI updates
            await new Promise(resolve => setTimeout(resolve, 0));
        }
    }
    
    if (progressCallback) {
        await progressCallback(100, trainingRounds, trainingRounds);
    }
    
    return { agent, totalReward, rewardHistory };
}

// Export for use in game.js
window.BanditAgent = BanditAgent;
window.trainBandit = trainBandit;
