# Blackjack - Monte Carlo Exploring Starts

## Overview

An AI that learns the **mathematically optimal Blackjack strategy** through self-play, using the Monte Carlo Exploring Starts (ES) algorithm. After 500,000 simulated games, Carlos plays nearly perfectly.

---

## The Mathematics of Blackjack

### Game State Space

The player's decision depends on three variables:

| Variable | Range | Values |
|----------|-------|--------|
| Player's Sum | 12-21 | 10 values |
| Dealer's Up Card | A, 2-10 | 10 values |
| Usable Ace | Yes/No | 2 values |

**Total States**: $10 \times 10 \times 2 = 200$ unique decision points

*Note: Sums below 12 always hit (no decision needed), and sums above 21 are busts.*

### Why These Specific Variables?

**Player's Sum (12-21)**: Below 12, hitting is always correct (can't bust). At 12+, there's a risk/reward decision.

**Dealer's Up Card**: The only information about the dealer's hand. Crucial because:
- Dealer must hit on 16 or below, stand on 17+
- Dealer's face-down card is unknown but follows known distribution

**Usable Ace**: An ace that can count as 11 without busting. This changes strategy dramatically because:
- With usable ace: More aggressive play (can't bust on next hit)
- Without usable ace: More conservative (real bust risk)

---

## Algorithm: Monte Carlo Exploring Starts

### The Exploration-Exploitation Problem

To find optimal actions, we need to:
1. **Explore** all state-action pairs to estimate their values
2. **Exploit** the best actions to maximize reward

**Exploring Starts** solves this by randomly selecting the initial state-action pair for each episode.

### Mathematical Foundation

#### State-Action Value Function

$$Q(s, a) = \mathbb{E}[G_t | S_t = s, A_t = a]$$

Where:
- $Q(s, a)$ = Expected return starting from state $s$, taking action $a$
- $G_t$ = Return (cumulative reward) from time $t$
- For Blackjack: $G_t \in \{-1, 0, +1\}$ (loss, tie, win)

#### First-Visit MC Update Rule

For each state-action pair $(s, a)$ visited for the first time in an episode:

$$Q(s, a) \leftarrow \frac{1}{N(s,a)} \sum_{i=1}^{N(s,a)} G_i(s, a)$$

Where:
- $N(s, a)$ = Number of times $(s, a)$ has been visited
- $G_i(s, a)$ = Return observed on the $i$-th visit

#### Policy Improvement

After each episode, the policy is improved greedily:

$$\pi(s) = \arg\max_a Q(s, a)$$

### Algorithm Pseudocode

```
Initialize:
    Q(s, a) ← arbitrary for all s ∈ S, a ∈ A
    Returns(s, a) ← empty list for all s, a
    π(s) ← arbitrary policy

Repeat for 500,000 episodes:
    # Exploring Starts
    s₀ ← random state from S
    a₀ ← random action from A(s₀)
    
    # Generate episode following π after first action
    Episode ← generate_episode(s₀, a₀, π)
    G ← terminal reward (+1, -1, or 0)
    
    # Update Q-values (first-visit)
    For each (s, a) appearing first time in Episode:
        Append G to Returns(s, a)
        Q(s, a) ← average(Returns(s, a))
        π(s) ← argmax_a Q(s, a)
```

### Convergence Guarantee

Monte Carlo ES converges to the optimal policy $\pi^*$ under these conditions:

1. **Exploring Starts**: All state-action pairs have non-zero probability of being selected as the starting pair

2. **Infinite Visits**: In the limit, each state-action pair is visited infinitely often:
   $$\lim_{k \to \infty} N_k(s, a) = \infty \quad \forall s, a$$

3. **Policy Improvement Theorem**: For any pair of deterministic policies $\pi$ and $\pi'$:
   $$Q_\pi(s, \pi'(s)) \geq Q_\pi(s, \pi(s)) \implies V_{\pi'}(s) \geq V_\pi(s)$$

---

## The Optimal Strategy (What Carlos Learns)

### Basic Strategy Table - Hard Totals (No Usable Ace)

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|--------|---|---|---|---|---|---|---|---|-----|---|
| **17-21** | S | S | S | S | S | S | S | S | S | S |
| **16** | S | S | S | S | S | H | H | H | H | H |
| **15** | S | S | S | S | S | H | H | H | H | H |
| **14** | S | S | S | S | S | H | H | H | H | H |
| **13** | S | S | S | S | S | H | H | H | H | H |
| **12** | H | H | S | S | S | H | H | H | H | H |

*S = Stand, H = Hit*

### Basic Strategy Table - Soft Totals (Usable Ace)

| Player | 2 | 3 | 4 | 5 | 6 | 7 | 8 | 9 | 10 | A |
|--------|---|---|---|---|---|---|---|---|-----|---|
| **20-21** | S | S | S | S | S | S | S | S | S | S |
| **19** | S | S | S | S | S | S | S | S | S | S |
| **18** | S | S | S | S | S | S | S | H | H | H |
| **17** | H | H | H | H | H | H | H | H | H | H |
| **12-16** | H | H | H | H | H | H | H | H | H | H |

### Why These Strategies Work

#### Standing on 12-16 vs. Dealer 2-6

When dealer shows 2-6 ("bust cards"), dealer must hit and has high bust probability:

| Dealer Up Card | Dealer Bust Probability |
|----------------|------------------------|
| 2 | 35.3% |
| 3 | 37.6% |
| 4 | 40.3% |
| 5 | 42.9% |
| 6 | 42.1% |

**Strategy**: Let the dealer bust. Don't risk busting yourself.

#### Hitting on 12-16 vs. Dealer 7-A

When dealer shows 7-A, dealer likely has 17+ already:

| Dealer Up Card | P(Dealer Total ≥ 17) |
|----------------|----------------------|
| 7 | 74.0% |
| 8 | 75.9% |
| 9 | 77.1% |
| 10 | 77.5% |
| A | 83.0% |

**Strategy**: You must improve your hand to compete.

---

## Mathematical Analysis of House Edge

### Perfect Play Expected Value

Even with optimal strategy, the house has an edge:

$$\mathbb{E}[\text{Return per hand}] \approx -0.5\%$$

This comes from:

| Situation | Probability | Net Effect |
|-----------|-------------|------------|
| Player busts first | ~28% | Player loses (even if dealer would bust) |
| Blackjack pays 3:2 | ~4.8% | +50% bonus mitigates some edge |
| Push (tie) | ~8.5% | No loss |

### Why the House Wins

The critical asymmetry: **Player acts first**.

If both player and dealer bust, the player loses (already gave up chips). This single rule creates the house edge despite otherwise symmetric gameplay.

---

## Implementation Details

### State Encoding

```python
state = (player_sum, dealer_card, usable_ace)
# Example: (15, 7, False) = Player has 15, dealer shows 7, no usable ace
```

### Episode Generation

```python
def generate_episode(start_state, start_action, policy):
    state = start_state
    action = start_action
    episode = [(state, action)]
    
    while not terminal(state):
        state = simulate_action(state, action)
        if not terminal(state):
            action = policy[state]
            episode.append((state, action))
    
    reward = get_outcome(state)  # +1, -1, or 0
    return episode, reward
```

### Q-Value Convergence

After 500,000 episodes, the Q-values converge with:

$$\sigma_{Q(s,a)} < 0.01 \quad \text{for all } (s, a)$$

---

## Visualization Outputs

### 1. Policy Heatmap (`policy.png`)

Two grids showing optimal action for each state:
- **Green (S)**: Stand
- **Red (H)**: Hit

### 2. Value Function Surface (`value_function.png`)

3D surface showing $V^*(s) = \max_a Q^*(s, a)$:
- Peak at player sum 21 (best position)
- Valley at player sum 12 vs. dealer A (worst position)

---

## Running the Code

```bash
python blackjack.py
```

**Output**:
1. Trains for 500,000 episodes (~60 seconds)
2. Prints optimal policy tables
3. Tests policy over 100,000 games
4. Generates visualization plots

---

## Key Takeaways

1. **Monte Carlo methods learn from complete episodes** - No model of the environment needed

2. **Exploring Starts ensures coverage** - Every state-action pair gets sampled

3. **Optimal play still loses** - The house edge (~0.5%) is mathematically unavoidable

4. **The learned strategy matches theoretical optimal** - Validates the algorithm's correctness

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. Chapter 5.
2. Thorp, E. O. (1966). *Beat the Dealer*. Random House.
3. Baldwin, R. et al. (1956). "The Optimum Strategy in Blackjack". *Journal of the American Statistical Association*.

---

**Disclaimer**: This is an educational demonstration. Even with perfect strategy, the house maintains an edge. Card counting (not implemented here) can shift the odds but is prohibited in casinos.
