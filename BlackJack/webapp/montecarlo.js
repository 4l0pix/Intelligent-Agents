/**
 * Monte Carlo Exploring Starts Algorithm for Blackjack
 * JavaScript implementation matching the Python version
 */

// Actions
const HIT = 0;
const STICK = 1;

// Card values: A, 2-10, J, Q, K
const CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10];

class MonteCarloES {
    constructor() {
        this.Q = new Map();
        this.returnsSum = new Map();
        this.returnsCount = new Map();
        this.policy = new Map();
        this.trainedEpisodes = 0;
    }

    // Generate a key for state-action pairs
    stateActionKey(state, action) {
        return `${state.playerSum}-${state.dealerShowing}-${state.usableAce}-${action}`;
    }

    stateKey(state) {
        return `${state.playerSum}-${state.dealerShowing}-${state.usableAce}`;
    }

    // Draw a random card from infinite deck
    drawCard() {
        return CARD_VALUES[Math.floor(Math.random() * CARD_VALUES.length)];
    }

    // Draw initial hand of two cards
    drawHand() {
        return [this.drawCard(), this.drawCard()];
    }

    // Check if hand has a usable ace
    usableAce(hand) {
        return hand.includes(1) && this.rawSum(hand) + 10 <= 21;
    }

    // Raw sum without ace adjustment
    rawSum(hand) {
        return hand.reduce((a, b) => a + b, 0);
    }

    // Sum of hand, treating ace as 11 if beneficial
    sumHand(hand) {
        if (this.usableAce(hand)) {
            return this.rawSum(hand) + 10;
        }
        return this.rawSum(hand);
    }

    // Check if hand is busted
    isBust(hand) {
        return this.sumHand(hand) > 21;
    }

    // Get state representation
    getState(playerHand, dealerShowing) {
        return {
            playerSum: this.sumHand(playerHand),
            dealerShowing: dealerShowing,
            usableAce: this.usableAce(playerHand)
        };
    }

    // Get Q-value, defaulting to 0
    getQ(state, action) {
        const key = this.stateActionKey(state, action);
        return this.Q.get(key) || 0;
    }

    // Set Q-value
    setQ(state, action, value) {
        const key = this.stateActionKey(state, action);
        this.Q.set(key, value);
    }

    // Player policy
    playerPolicy(state, type = 'greedy') {
        if (type === 'initial') {
            return state.playerSum >= 20 ? STICK : HIT;
        }
        // Greedy policy
        return this.getQ(state, HIT) >= this.getQ(state, STICK) ? HIT : STICK;
    }

    // Dealer's fixed strategy
    dealerPolicy(dealerHand) {
        return this.sumHand(dealerHand) < 17 ? HIT : STICK;
    }

    // Play one episode
    playEpisode(initialState = null, initialAction = null) {
        let playerHand, dealerHand, dealerShowing;

        if (initialState) {
            // Create hands matching initial state (Exploring Starts)
            if (initialState.usableAce) {
                playerHand = [1, initialState.playerSum - 11];
            } else {
                if (initialState.playerSum <= 11) {
                    playerHand = [initialState.playerSum];
                } else {
                    playerHand = [10, initialState.playerSum - 10];
                }
            }
            dealerShowing = initialState.dealerShowing;
            dealerHand = [dealerShowing, this.drawCard()];
        } else {
            playerHand = this.drawHand();
            dealerHand = this.drawHand();
            dealerShowing = dealerHand[0];
        }

        const episode = [];

        // Player's turn
        while (true) {
            const playerSum = this.sumHand(playerHand);

            // If sum < 12, always hit
            if (playerSum < 12) {
                playerHand.push(this.drawCard());
                continue;
            }

            const state = this.getState(playerHand, dealerShowing);
            let action;

            if (initialAction !== null) {
                action = initialAction;
                initialAction = null;
            } else {
                action = this.playerPolicy(state, 'greedy');
            }

            episode.push({ state, action });

            if (action === STICK) {
                break;
            } else {
                playerHand.push(this.drawCard());
                if (this.isBust(playerHand)) {
                    // Player busts
                    return episode.map((e, i) => ({
                        ...e,
                        reward: i === episode.length - 1 ? -1 : 0
                    }));
                }
            }
        }

        // Dealer's turn
        while (this.dealerPolicy(dealerHand) === HIT) {
            dealerHand.push(this.drawCard());
        }

        // Determine winner
        const playerSum = this.sumHand(playerHand);
        const dealerSum = this.sumHand(dealerHand);
        let reward;

        if (this.isBust(dealerHand)) {
            reward = 1;
        } else if (dealerSum > playerSum) {
            reward = -1;
        } else if (dealerSum < playerSum) {
            reward = 1;
        } else {
            reward = 0;
        }

        return episode.map((e, i) => ({
            ...e,
            reward: i === episode.length - 1 ? reward : 0
        }));
    }

    // Generate all possible states
    getAllStates() {
        const states = [];
        for (let playerSum = 12; playerSum <= 21; playerSum++) {
            for (let dealerShowing = 1; dealerShowing <= 10; dealerShowing++) {
                for (const usableAce of [true, false]) {
                    states.push({ playerSum, dealerShowing, usableAce });
                }
            }
        }
        return states;
    }

    // Train the agent
    async train(numEpisodes, progressCallback = null) {
        const allStates = this.getAllStates();
        const batchSize = 1000;

        for (let i = 0; i < numEpisodes; i++) {
            // Exploring Starts: random initial state and action
            const initialState = allStates[Math.floor(Math.random() * allStates.length)];
            const initialAction = Math.random() < 0.5 ? HIT : STICK;

            // Generate episode
            const episode = this.playEpisode(initialState, initialAction);

            // First-visit MC update
            const visited = new Set();
            let G = 0;

            for (let t = episode.length - 1; t >= 0; t--) {
                const { state, action, reward } = episode[t];
                G = G + reward;

                const saKey = this.stateActionKey(state, action);
                if (!visited.has(saKey)) {
                    visited.add(saKey);

                    const currentSum = this.returnsSum.get(saKey) || 0;
                    const currentCount = this.returnsCount.get(saKey) || 0;

                    this.returnsSum.set(saKey, currentSum + G);
                    this.returnsCount.set(saKey, currentCount + 1);

                    const newQ = (currentSum + G) / (currentCount + 1);
                    this.setQ(state, action, newQ);
                }
            }

            // Update progress periodically
            if (progressCallback && (i + 1) % batchSize === 0) {
                await new Promise(resolve => setTimeout(resolve, 0)); // Yield to UI
                progressCallback((i + 1) / numEpisodes, i + 1);
            }
        }

        // Extract policy
        for (const state of allStates) {
            const hitQ = this.getQ(state, HIT);
            const stickQ = this.getQ(state, STICK);
            this.policy.set(this.stateKey(state), hitQ >= stickQ ? HIT : STICK);
        }

        this.trainedEpisodes = numEpisodes;
    }

    // Get action from learned policy
    getAction(state) {
        const key = this.stateKey(state);
        if (this.policy.has(key)) {
            return this.policy.get(key);
        }
        // Default: hit if under 17
        return state.playerSum < 17 ? HIT : STICK;
    }

    // Get the policy for display
    getPolicyGrid(usableAce) {
        const grid = [];
        for (let playerSum = 21; playerSum >= 12; playerSum--) {
            const row = [];
            for (let dealerShowing = 1; dealerShowing <= 10; dealerShowing++) {
                const state = { playerSum, dealerShowing, usableAce };
                row.push(this.getAction(state));
            }
            grid.push({ playerSum, actions: row });
        }
        return grid;
    }
}

// Export for use in game.js
window.MonteCarloES = MonteCarloES;
window.HIT = HIT;
window.STICK = STICK;
