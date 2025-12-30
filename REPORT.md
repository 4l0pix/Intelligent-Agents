# Intelligent Agents: A Mathematical Analysis of Reinforcement Learning in Casino Games

## Document Purpose

This report provides comprehensive mathematical documentation for an educational project demonstrating reinforcement learning algorithms through interactive casino game simulations. The project serves as a pedagogical tool illustrating why gambling systems fail mathematically while teaching core RL concepts.

---

## 1. Project Overview

### 1.1 Educational Objectives

- Demonstrate reinforcement learning algorithms in accessible, interactive environments
- Provide rigorous mathematical proofs that all betting systems have negative expected value
- Illustrate the exploration-exploitation tradeoff fundamental to sequential decision making
- Show convergence properties of value-based learning methods

### 1.2 Implemented Games and Algorithms

| Game | Algorithm | Mathematical Foundation |
|------|-----------|------------------------|
| Blackjack | Monte Carlo Exploring Starts | Q-function estimation, policy iteration |
| Bandit Slots | Epsilon-Greedy, Softmax | Multi-armed bandit theory, regret bounds |
| Texas Hold'em | Monte Carlo Simulation | Bayesian inference, pot odds theory |
| Roulette | Monte Carlo Analysis | Martingale theory, Optional Stopping Theorem |

---

## 2. Blackjack: Monte Carlo Exploring Starts

### 2.1 Problem Formulation

Blackjack is modeled as an episodic Markov Decision Process (MDP):

**State Space**: $S = \{(p, d, u) : p \in \{12,...,21\}, d \in \{2,...,11\}, u \in \{0,1\}\}$

Where:
- $p$ = player's hand total (12-21, as below 12 always hit)
- $d$ = dealer's visible card (2-11, where 11 represents Ace)
- $u$ = usable ace indicator (1 if player has usable ace, 0 otherwise)

**State Space Cardinality**: $|S| = 10 \times 10 \times 2 = 200$ states

**Action Space**: $A = \{\text{hit}, \text{stand}\}$

**Reward Structure**:
$$R = \begin{cases} +1 & \text{if player wins} \\ -1 & \text{if player loses} \\ 0 & \text{if draw} \end{cases}$$

### 2.2 Value Function Definition

The action-value function represents expected return from state $s$, taking action $a$, then following policy $\pi$:

$$Q^\pi(s,a) = \mathbb{E}_\pi[G_t | S_t = s, A_t = a]$$

Where the return $G_t$ is:
$$G_t = \sum_{k=0}^{T-t} \gamma^k R_{t+k+1}$$

For episodic tasks with $\gamma = 1$:
$$G_t = R_{T}$$

### 2.3 Monte Carlo Exploring Starts Algorithm

```
Algorithm: Monte Carlo ES for Blackjack
─────────────────────────────────────────
Initialize:
  Q(s,a) ← arbitrary, for all s ∈ S, a ∈ A
  π(s) ← arbitrary, for all s ∈ S
  Returns(s,a) ← empty list, for all s ∈ A

Repeat for each episode:
  Choose S₀ ∈ S, A₀ ∈ A such that all pairs have probability > 0
  Generate episode starting from S₀, A₀ following π
  
  For each pair (s,a) appearing in the episode:
    G ← return following first occurrence of (s,a)
    Append G to Returns(s,a)
    Q(s,a) ← average(Returns(s,a))
    π(s) ← argmax_a Q(s,a)
```

### 2.4 Convergence Guarantee

**Theorem (MC Convergence)**: Under Monte Carlo Exploring Starts, $Q(s,a) \to Q^*(s,a)$ as the number of episodes approaches infinity, provided all state-action pairs are visited infinitely often.

**Proof Sketch**:
1. By the Law of Large Numbers, sample averages converge to expected values
2. Exploring starts ensures all $(s,a)$ pairs are visited
3. Policy improvement theorem guarantees $\pi_{k+1} \geq \pi_k$
4. Finite MDP implies convergence to $\pi^*$ in finite iterations

### 2.5 Optimal Strategy Results

After 500,000 training episodes, the learned policy converges to known optimal basic strategy:

**Hard Totals Decision Matrix** (H=Hit, S=Stand):

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|--------|---|---|---|---|---|---|---|---|----|----|
| 17-21 | S | S | S | S | S | S | S | S | S | S |
| 13-16 | S | S | S | S | S | H | H | H | H | H |
| 12 | H | H | S | S | S | H | H | H | H | H |
| ≤11 | H | H | H | H | H | H | H | H | H | H |

**Soft Totals** (with usable Ace):

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|--------|---|---|---|---|---|---|---|---|----|----|
| 19-21 | S | S | S | S | S | S | S | S | S | S |
| 18 | S | S | S | S | S | S | S | H | H | H |
| ≤17 | H | H | H | H | H | H | H | H | H | H |

### 2.6 House Edge Analysis

Expected value per hand with optimal play:

$$\mathbb{E}[\text{Return}] \approx -0.005$$

This 0.5% house edge arises from:
- Dealer acts last (player can bust first)
- Blackjack pays 3:2, not 2:1
- Dealer must hit on 16, stand on 17

---

## 3. Multi-Armed Bandits: Slot Machines

### 3.1 Problem Formulation

The $k$-armed bandit problem: at each time step $t$, select action $A_t \in \{1, 2, ..., k\}$ and receive reward $R_t$ drawn from unknown distribution associated with $A_t$.

**Objective**: Maximize cumulative reward over $T$ time steps:
$$\max \sum_{t=1}^{T} R_t$$

**True Action Values**:
$$q_*(a) = \mathbb{E}[R_t | A_t = a]$$

**Estimated Values** (sample average):
$$Q_t(a) = \frac{\sum_{i=1}^{t-1} R_i \cdot \mathbb{1}_{A_i=a}}{\sum_{i=1}^{t-1} \mathbb{1}_{A_i=a}}$$

### 3.2 Incremental Update Formula

For computational efficiency, maintain running averages:

$$Q_{n+1}(a) = Q_n(a) + \frac{1}{n}[R_n - Q_n(a)]$$

General form with step size $\alpha$:
$$Q_{n+1}(a) = Q_n(a) + \alpha[R_n - Q_n(a)]$$

### 3.3 Epsilon-Greedy Algorithm

```
Algorithm: ε-Greedy Action Selection
────────────────────────────────────
Parameter: ε ∈ (0, 1)

At each time step t:
  Generate u ~ Uniform(0, 1)
  
  If u < ε:
    A_t ← random action (explore)
  Else:
    A_t ← argmax_a Q_t(a) (exploit)
  
  Execute A_t, observe R_t
  Update Q_t(A_t) using incremental formula
```

**Action Selection Probability**:
$$P(A_t = a) = \begin{cases} 1 - \varepsilon + \frac{\varepsilon}{k} & \text{if } a = \arg\max_{a'} Q_t(a') \\ \frac{\varepsilon}{k} & \text{otherwise} \end{cases}$$

### 3.4 Softmax (Boltzmann) Exploration

```
Algorithm: Softmax Action Selection
───────────────────────────────────
Parameter: τ > 0 (temperature)

At each time step t:
  For each action a, compute:
    P(A_t = a) = exp(Q_t(a)/τ) / Σ_b exp(Q_t(b)/τ)
  
  Sample A_t from this distribution
  Execute A_t, observe R_t
  Update Q_t(A_t)
```

**Temperature Effects**:
- $\tau \to 0$: Greedy selection (pure exploitation)
- $\tau \to \infty$: Uniform random selection (pure exploration)
- Optimal $\tau$: Problem-dependent, often $\tau \in [0.1, 1.0]$

### 3.5 Regret Analysis

**Definition (Regret)**: The expected loss compared to always selecting the optimal action:

$$\text{Regret}_T = T \cdot q_*(a^*) - \mathbb{E}\left[\sum_{t=1}^{T} R_t\right]$$

Where $a^* = \arg\max_a q_*(a)$.

**Theorem (ε-Greedy Regret Bound)**:
For ε-greedy with $\varepsilon = \min\{1, \frac{ck}{d^2 T}\}$:

$$\text{Regret}_T = O\left(\sqrt{kT \log T}\right)$$

**Lower Bound (Lai & Robbins, 1985)**:
For any consistent policy:

$$\liminf_{T \to \infty} \frac{\text{Regret}_T}{\log T} \geq \sum_{a: \Delta_a > 0} \frac{\Delta_a}{D_{KL}(p_a || p_{a^*})}$$

Where $\Delta_a = q_*(a^*) - q_*(a)$ is the suboptimality gap.

### 3.6 Comparison of Algorithms

| Algorithm | Regret Bound | Pros | Cons |
|-----------|--------------|------|------|
| ε-Greedy | $O(\sqrt{kT})$ | Simple, robust | Fixed exploration rate |
| Softmax | $O(\sqrt{kT})$ | Smooth, differentiable | Temperature tuning |
| UCB | $O(\sqrt{kT \log T})$ | No hyperparameters | Assumes bounded rewards |
| Thompson | $O(\sqrt{kT})$ | Optimal empirically | Requires prior |

---

## 4. Texas Hold'em: Monte Carlo Hand Evaluation

### 4.1 Game Complexity

**State Space**: Approximately $10^{18}$ possible game states

**Information Sets**: Imperfect information game with hidden opponent cards

**Decision Points**: Preflop, Flop, Turn, River (4 betting rounds)

### 4.2 Hand Probability Mathematics

From a standard 52-card deck, the number of 5-card hands:

$$\binom{52}{5} = 2,598,960$$

**Hand Distribution**:

| Rank | Hand | Combinations | Probability |
|------|------|--------------|-------------|
| 1 | Royal Flush | 4 | 0.000154% |
| 2 | Straight Flush | 36 | 0.00139% |
| 3 | Four of a Kind | 624 | 0.0240% |
| 4 | Full House | 3,744 | 0.144% |
| 5 | Flush | 5,108 | 0.197% |
| 6 | Straight | 10,200 | 0.392% |
| 7 | Three of a Kind | 54,912 | 2.11% |
| 8 | Two Pair | 123,552 | 4.75% |
| 9 | One Pair | 1,098,240 | 42.3% |
| 10 | High Card | 1,302,540 | 50.1% |

### 4.3 Monte Carlo Equity Estimation

**Definition (Equity)**: Probability of winning given current information:

$$\text{Equity} = P(\text{Win} | \text{Hole Cards}, \text{Community Cards})$$

**Monte Carlo Estimation**:

```
Algorithm: Monte Carlo Hand Equity
──────────────────────────────────
Input: hole_cards, community_cards, num_simulations

wins ← 0
For i = 1 to num_simulations:
    remaining_deck ← deck \ (hole_cards ∪ community_cards)
    Shuffle remaining_deck
    
    # Complete the board
    board ← community_cards ∪ remaining_deck[1:5-|community_cards|]
    
    # Deal opponent hand
    opponent ← remaining_deck[6:7]
    
    # Evaluate hands
    player_hand ← best_5_of_7(hole_cards, board)
    opponent_hand ← best_5_of_7(opponent, board)
    
    If player_hand > opponent_hand:
        wins ← wins + 1
    Else if player_hand = opponent_hand:
        wins ← wins + 0.5

Return wins / num_simulations
```

### 4.4 Convergence by Central Limit Theorem

Let $X_i \in \{0, 0.5, 1\}$ be the outcome of simulation $i$.

**Sample Mean**: $\bar{X}_n = \frac{1}{n}\sum_{i=1}^{n} X_i$

**By CLT**: As $n \to \infty$:
$$\sqrt{n}(\bar{X}_n - \mu) \xrightarrow{d} \mathcal{N}(0, \sigma^2)$$

**95% Confidence Interval**:
$$\text{Equity} \in \left[\bar{X}_n - \frac{1.96\sigma}{\sqrt{n}}, \bar{X}_n + \frac{1.96\sigma}{\sqrt{n}}\right]$$

For $n = 1000$ simulations with $\sigma \approx 0.5$:
$$\text{Margin of Error} \approx \frac{1.96 \times 0.5}{\sqrt{1000}} \approx 3.1\%$$

### 4.5 Pot Odds Theory

**Definition (Pot Odds)**: Ratio of current pot to cost of call:

$$\text{Pot Odds} = \frac{\text{Pot Size}}{\text{Call Amount}}$$

**Decision Rule**:
$$\text{EV}[\text{Call}] = \text{Equity} \times (\text{Pot} + \text{Call}) - (1 - \text{Equity}) \times \text{Call}$$

Call is profitable when:
$$\text{Equity} > \frac{\text{Call}}{\text{Pot} + \text{Call}}$$

**Example**:
- Pot: $100
- Call: $50
- Required Equity: $\frac{50}{150} = 33.3\%$

### 4.6 Game-Theoretic Considerations

**Nash Equilibrium**: Strategy profile where no player benefits from unilateral deviation.

**Bluff-to-Value Ratio** at equilibrium:
$$\frac{\text{Bluffs}}{\text{Value Bets}} = \frac{\text{Bet Size}}{\text{Pot} + \text{Bet Size}}$$

For pot-sized bet:
$$\frac{\text{Bluffs}}{\text{Value}} = \frac{1}{2}$$

---

## 5. Roulette: Betting System Analysis

### 5.1 Mathematical Model

**American Roulette**: 38 pockets (0, 00, 1-36)

**Probability Structure**:
- Red/Black: $P = \frac{18}{38} \approx 0.4737$
- Single number: $P = \frac{1}{38} \approx 0.0263$

**House Edge Calculation**:

For even-money bets (Red/Black):
$$\mathbb{E}[\text{Profit}] = (+1) \times \frac{18}{38} + (-1) \times \frac{20}{38} = -\frac{2}{38} \approx -5.26\%$$

For single-number bets (35:1 payout):
$$\mathbb{E}[\text{Profit}] = (+35) \times \frac{1}{38} + (-1) \times \frac{37}{38} = -\frac{2}{38} \approx -5.26\%$$

**Key Insight**: House edge is identical regardless of bet type.

### 5.2 Martingale System Analysis

**Strategy**: Double bet after each loss, reset after win.

**Bet Sequence**: $1, 2, 4, 8, 16, 32, ..., 2^{n-1}$

**After $n$ consecutive losses**:
$$\text{Total Wagered} = \sum_{i=0}^{n-1} 2^i = 2^n - 1$$

**Expected Value Proof**:

Let $p = \frac{18}{38}$ (win probability), $q = \frac{20}{38}$ (loss probability).

Probability of losing $n$ consecutive times:
$$P(\text{n losses}) = q^n = \left(\frac{20}{38}\right)^n$$

Expected profit per "cycle":
$$\mathbb{E}[\text{Profit}] = \sum_{n=1}^{\infty} \left[(+1) \times q^{n-1}p - (2^n-1) \times q^n \times \mathbb{1}_{\text{table limit}}\right]$$

With infinite bankroll and no table limits (impossible):
$$\mathbb{E}[\text{Profit}] = (+1) \times \sum_{n=1}^{\infty} q^{n-1}p = \frac{p}{1-q} = 1$$

**But** with table limit $L$ (e.g., $2^{10} = 1024$):
$$\mathbb{E}[\text{Profit}] = p \times 1 + q \times p \times 1 + ... + q^{9} \times p \times 1 - q^{10} \times (2^{10}-1)$$
$$= \frac{p(1-q^{10})}{1-q} - q^{10}(1023) < 0$$

### 5.3 D'Alembert System Analysis

**Strategy**: Increase bet by 1 after loss, decrease by 1 after win.

**Bet Sequence Example**: $1, 2, 3, 2, 3, 4, 3, 2, 1$

**Mathematical Analysis**:

Let $W$ = number of wins, $L$ = number of losses, $n = W + L$ total spins.

If $W = L$ (equal wins and losses):
$$\text{Profit} = W \times (\text{average winning bet}) - L \times (\text{average losing bet})$$

Due to asymmetric bet sizing, profit appears positive when $W = L$. However:

$$\mathbb{E}[W - L] = n \times (p - q) = n \times \left(\frac{18}{38} - \frac{20}{38}\right) = -\frac{2n}{38}$$

Expected losses compound over time.

### 5.4 Fibonacci System Analysis

**Strategy**: Bet following Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, ...) after losses.

**Fibonacci Recurrence**: $F_n = F_{n-1} + F_{n-2}$, with $F_1 = F_2 = 1$

**Closed Form (Binet's Formula)**:
$$F_n = \frac{\phi^n - \psi^n}{\sqrt{5}}$$

Where $\phi = \frac{1+\sqrt{5}}{2} \approx 1.618$ (golden ratio), $\psi = \frac{1-\sqrt{5}}{2}$

**Growth Rate**: $F_n = O(\phi^n)$ - exponential growth

**Risk**: After 20 consecutive losses:
$$F_{20} = 6,765 \text{ units}$$
$$\text{Total wagered} = \sum_{i=1}^{20} F_i = F_{22} - 1 = 17,710 \text{ units}$$

### 5.5 The Optional Stopping Theorem

**Theorem**: Let $(M_t)_{t \geq 0}$ be a martingale and $\tau$ a stopping time. Under certain conditions:
$$\mathbb{E}[M_\tau] = \mathbb{E}[M_0]$$

**Application to Gambling**:

Let $X_t$ = cumulative profit at time $t$. For fair game: $(X_t)$ is a martingale.

For biased game (casino): $\mathbb{E}[X_{t+1} | X_t] = X_t - c$ where $c > 0$ is house edge.

**This is a supermartingale**, and for any stopping time $\tau$:
$$\mathbb{E}[X_\tau] \leq \mathbb{E}[X_0] = 0$$

**Implication**: No stopping rule can yield positive expected profit against a house edge.

### 5.6 Formal Proof: All Systems Fail

**Theorem**: For any betting system $\mathcal{S}$ with finite expected stopping time $\mathbb{E}[\tau] < \infty$, the expected profit is negative.

**Proof**:

Let $B_t$ denote the bet at time $t$, and $Y_t \in \{-1, +1\}$ the outcome.

Define $X_t = \sum_{i=1}^{t} B_i Y_i$ (cumulative profit).

For any non-anticipating betting strategy:
$$\mathbb{E}[X_t | X_1, ..., X_{t-1}] = X_{t-1} + B_t \mathbb{E}[Y_t]$$
$$= X_{t-1} + B_t \times (p - q)$$
$$= X_{t-1} - B_t \times \frac{2}{38}$$

Since $B_t > 0$:
$$\mathbb{E}[X_t] < \mathbb{E}[X_{t-1}]$$

By induction:
$$\mathbb{E}[X_\tau] = -\frac{2}{38} \sum_{t=1}^{\tau} \mathbb{E}[B_t] < 0$$

**QED**

---

## 6. Unified Theoretical Framework

### 6.1 Common Mathematical Structures

All games share foundational concepts:

1. **State Spaces**: Finite or countable sets describing game configurations
2. **Stochastic Transitions**: Probability distributions over next states
3. **Reward Signals**: Numerical feedback for learning
4. **Value Functions**: Expected cumulative rewards from states/actions

### 6.2 The Fundamental Theorem of Gambling

**Theorem**: In any game with negative expected value per play, no combination of:
- Bet sizing strategies
- Entry/exit timing rules
- Game selection methods

can produce positive expected value.

**Corollary**: The only winning move is not to play.

### 6.3 Reinforcement Learning Insights

The project demonstrates:

1. **Optimal policies exist** for well-defined MDPs (Blackjack)
2. **Exploration is necessary** to discover optimal actions (Bandits)
3. **Monte Carlo methods** effectively estimate expectations (Poker)
4. **Mathematical laws are inviolable** regardless of betting patterns (Roulette)

---

## 7. Implementation Notes

### 7.1 Technologies Used

- **Frontend**: HTML5, CSS3, JavaScript (ES6+)
- **Visualization**: Chart.js for bankroll tracking
- **Design**: Glass morphism UI with green casino theme

### 7.2 Algorithm Parameters

| Game | Parameter | Default Value | Purpose |
|------|-----------|---------------|---------|
| Blackjack | Episodes | 500,000 | Training iterations |
| Bandits | Epsilon | 0.1 | Exploration rate |
| Bandits | Temperature | 1.0 | Softmax parameter |
| Poker | Simulations | 1,000 | Monte Carlo samples |
| Roulette | Spins | 100 | Session length |

### 7.3 Convergence Verification

All algorithms were verified to converge to known theoretical values:
- Blackjack: Matches published basic strategy tables
- Bandits: Regret grows sublinearly as predicted
- Poker: Equity estimates within 3% of exhaustive calculation
- Roulette: All systems converge to -5.26% expected return

---

## 8. Educational Outcomes

### 8.1 Learning Objectives Achieved

Students interacting with this project will understand:

1. **Why casinos always win**: Mathematical expectation guarantees house profit
2. **How AI learns**: Value estimation through experience
3. **Exploration vs exploitation**: The fundamental RL tradeoff
4. **Convergence properties**: When and why learning algorithms work

### 8.2 Anti-Gambling Message

This project serves as a mathematical intervention against problem gambling by proving:

- **No system beats the house edge**
- **Variance is not profit** - short-term wins are statistical noise
- **Gambler's fallacy is false** - past outcomes don't affect future probabilities
- **Expected value is destiny** - mathematics determines long-run outcomes

---

## 9. References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction* (2nd ed.). MIT Press.

2. Lai, T. L., & Robbins, H. (1985). Asymptotically efficient adaptive allocation rules. *Advances in Applied Mathematics*, 6(1), 4-22.

3. Thorp, E. O. (1966). *Beat the Dealer: A Winning Strategy for the Game of Twenty-One*. Vintage Books.

4. Billingsley, P. (2012). *Probability and Measure* (Anniversary ed.). Wiley.

5. Williams, D. (1991). *Probability with Martingales*. Cambridge University Press.

6. Bowling, M., et al. (2015). Heads-up limit hold'em poker is solved. *Science*, 347(6218), 145-149.

---

## 10. Appendix: Key Formulas Summary

### Bellman Optimality Equation
$$Q^*(s,a) = \mathbb{E}[R_{t+1} + \gamma \max_{a'} Q^*(S_{t+1}, a') | S_t=s, A_t=a]$$

### Monte Carlo Update
$$Q(s,a) \leftarrow Q(s,a) + \alpha[G_t - Q(s,a)]$$

### Epsilon-Greedy Selection
$$\pi(a|s) = \begin{cases} 1-\varepsilon+\frac{\varepsilon}{|A|} & a = \arg\max Q(s,a) \\ \frac{\varepsilon}{|A|} & \text{otherwise} \end{cases}$$

### Softmax Selection
$$\pi(a|s) = \frac{e^{Q(s,a)/\tau}}{\sum_b e^{Q(s,b)/\tau}}$$

### House Edge (American Roulette)
$$\mathbb{E}[\text{Return}] = -\frac{2}{38} \approx -5.26\%$$

### Regret Lower Bound
$$\liminf_{T\to\infty} \frac{\text{Regret}_T}{\log T} \geq \sum_{a:\Delta_a>0} \frac{\Delta_a}{D_{KL}(p_a||p^*)}$$

---

*Document prepared for academic article generation. All mathematical content verified against established literature.*
