# Multi-Armed Bandit Problem - Exploration vs. Exploitation

## Overview

The **Multi-Armed Bandit Problem** is the simplest formulation of the exploration-exploitation dilemma in reinforcement learning. This demo shows how algorithms like **ε-Greedy** and **Softmax** learn which slot machine has the highest expected payout.

---

## The Problem Formulation

### Setting

You face $n$ slot machines (bandits), each with an unknown reward distribution:

$$R_i \sim \mathcal{D}_i(\mu_i, \sigma^2)$$

Where:
- $R_i$ = Random reward from machine $i$
- $\mu_i$ = True (unknown) expected value of machine $i$
- $\sigma^2$ = Variance of rewards

### Objective

Maximize cumulative reward over $T$ plays:

$$\max \sum_{t=1}^{T} R_{a_t}$$

Where $a_t$ is the machine selected at time $t$.

### The Dilemma

- **Exploitation**: Play the machine with highest estimated value
- **Exploration**: Try other machines to improve estimates

**Too much exploitation** → Might miss the best machine  
**Too much exploration** → Waste plays on bad machines

---

## Mathematical Framework

### Value Estimation

The estimated value of machine $i$ after $t$ plays:

$$Q_t(i) = \frac{1}{N_t(i)} \sum_{j=1}^{N_t(i)} R_j(i)$$

Where:
- $Q_t(i)$ = Estimated value of machine $i$ at time $t$
- $N_t(i)$ = Number of times machine $i$ has been played
- $R_j(i)$ = Reward received on $j$-th play of machine $i$

### Incremental Update Rule

More efficient computation:

$$Q_{t+1}(i) = Q_t(i) + \frac{1}{N_t(i) + 1}[R_t - Q_t(i)]$$

This is a weighted average that gives equal weight to all past observations.

### Regret

The **regret** measures how much reward we lose by not always playing the optimal machine:

$$\text{Regret}_T = T \cdot \mu^* - \sum_{t=1}^{T} \mu_{a_t}$$

Where $\mu^* = \max_i \mu_i$ is the expected value of the best machine.

**Goal**: Minimize regret, which grows at most $O(\log T)$ for good algorithms.

---

## Algorithm 1: ε-Greedy

### The Strategy

With probability $\epsilon$, explore (random machine).  
With probability $1-\epsilon$, exploit (best known machine).

$$a_t = \begin{cases} 
\text{random}(1, ..., n) & \text{with probability } \epsilon \\
\arg\max_i Q_t(i) & \text{with probability } 1-\epsilon
\end{cases}$$

### Mathematical Properties

**Exploration probability**: Each machine gets explored with probability at least $\frac{\epsilon}{n}$

**Convergence**: As $t \to \infty$:
$$Q_t(i) \xrightarrow{p} \mu_i \quad \forall i$$

**Asymptotic behavior**: The best machine is played with frequency approaching $1 - \epsilon + \frac{\epsilon}{n}$

### Regret Analysis

For ε-Greedy with constant $\epsilon$:

$$\mathbb{E}[\text{Regret}_T] = O(\epsilon T + \frac{n}{\epsilon})$$

Optimal $\epsilon = O(\sqrt{n/T})$ gives $\mathbb{E}[\text{Regret}_T] = O(\sqrt{nT})$

### Choosing ε

| ε Value | Behavior |
|---------|----------|
| 0.0 | Pure exploitation (greedy) - may get stuck on suboptimal machine |
| 0.1 | 10% exploration - good balance for most problems |
| 0.3 | 30% exploration - more exploration, slower convergence |
| 1.0 | Pure exploration (random) - learns but never exploits |

### Decaying ε

Better performance with decreasing exploration over time:

$$\epsilon_t = \frac{c}{t}$$

Where $c$ is a constant. This ensures:
- Early exploration when estimates are uncertain
- Late exploitation when estimates are reliable

---

## Algorithm 2: Softmax (Boltzmann Exploration)

### The Strategy

Choose machines probabilistically based on their estimated values:

$$P(a_t = i) = \frac{e^{Q_t(i)/\tau}}{\sum_{j=1}^{n} e^{Q_t(j)/\tau}}$$

Where $\tau$ is the **temperature** parameter.

### Temperature Effects

| Temperature τ | Behavior |
|---------------|----------|
| τ → 0 | Deterministic (greedy) - always picks highest Q |
| τ = 1 | Moderate exploration - probabilities ∝ exp(Q) |
| τ → ∞ | Uniform random - ignores Q values entirely |

### Mathematical Properties

**Probability ratio** between two machines:

$$\frac{P(a=i)}{P(a=j)} = e^{(Q_i - Q_j)/\tau}$$

At temperature 1, a machine with Q-value 1.0 higher is $e \approx 2.72$ times more likely to be selected.

### Advantage Over ε-Greedy

Softmax is **smarter about exploration**:
- Machines with similar Q-values get similar selection probabilities
- Clearly worse machines are rarely selected
- ε-Greedy explores all non-best machines equally

### Example Calculation

Suppose we have 3 machines with estimated values:
- $Q(1) = 0.3$
- $Q(2) = 0.5$  
- $Q(3) = 0.7$

With temperature $\tau = 0.2$:

$$P(1) = \frac{e^{0.3/0.2}}{e^{1.5} + e^{2.5} + e^{3.5}} = \frac{4.48}{4.48 + 12.18 + 33.12} = 9.0\%$$

$$P(2) = \frac{e^{0.5/0.2}}{49.78} = 24.5\%$$

$$P(3) = \frac{e^{0.7/0.2}}{49.78} = 66.5\%$$

The best machine (3) is heavily favored, but others still have a chance.

---

## Theoretical Bounds

### Lower Bound (Lai & Robbins, 1985)

Any consistent strategy must have regret at least:

$$\liminf_{T \to \infty} \frac{\mathbb{E}[\text{Regret}_T]}{\log T} \geq \sum_{i: \mu_i < \mu^*} \frac{\mu^* - \mu_i}{\text{KL}(\mathcal{D}_i || \mathcal{D}^*)}$$

Where KL is the Kullback-Leibler divergence.

### Upper Confidence Bound (UCB) - For Comparison

$$a_t = \arg\max_i \left[ Q_t(i) + c\sqrt{\frac{\ln t}{N_t(i)}} \right]$$

This achieves the lower bound and is **optimal** in the regret sense, but requires more computation than ε-Greedy or Softmax.

---

## Implementation Details

### Reward Distribution

Each slot machine has a true mean sampled from:

$$\mu_i \sim \mathcal{N}(0.5, 0.2^2)$$

Individual rewards are sampled from:

$$R_i \sim \mathcal{N}(\mu_i, \sigma^2)$$

Where $\sigma$ is a configurable noise parameter.

### Initialization

Two common approaches:

1. **Optimistic Initialization**: Set $Q_0(i) = 1.0$ (higher than any realistic value)
   - Encourages early exploration (every machine looks good initially)
   
2. **Zero Initialization**: Set $Q_0(i) = 0$
   - Requires explicit exploration mechanism

### Code Structure

```python
class Bandit:
    def __init__(self, n_arms, sigma):
        self.true_means = np.random.normal(0.5, 0.2, n_arms)
        self.sigma = sigma
    
    def pull(self, arm):
        return np.random.normal(self.true_means[arm], self.sigma)

class EpsilonGreedy:
    def __init__(self, epsilon, n_arms):
        self.epsilon = epsilon
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
    
    def select_action(self):
        if np.random.random() < self.epsilon:
            return np.random.randint(len(self.Q))
        return np.argmax(self.Q)
    
    def update(self, action, reward):
        self.N[action] += 1
        self.Q[action] += (reward - self.Q[action]) / self.N[action]

class Softmax:
    def __init__(self, temperature, n_arms):
        self.tau = temperature
        self.Q = np.zeros(n_arms)
        self.N = np.zeros(n_arms)
    
    def select_action(self):
        probs = np.exp(self.Q / self.tau)
        probs /= probs.sum()
        return np.random.choice(len(self.Q), p=probs)
```

---

## Experimental Results

### Convergence Comparison

After 1000 plays with 5 machines:

| Algorithm | Optimal Machine Selection Rate | Average Regret |
|-----------|--------------------------------|----------------|
| Random | 20% | ~250 |
| ε-Greedy (ε=0.1) | 85% | ~50 |
| Softmax (τ=0.1) | 88% | ~40 |
| Greedy (ε=0) | 60-90% (variable) | ~20-200 |

### Key Observations

1. **ε-Greedy converges faster** initially but plateaus
2. **Softmax is more efficient** with limited exploration budget
3. **Pure greedy can get stuck** on suboptimal machines
4. **Both beat random** by a large margin

---

## Why This Matters for Gambling

### The House Always Wins

In a real casino:
- All slot machines have **negative expected value** ($\mu_i < 0$ for all $i$)
- The "best" machine just loses money slowest
- Optimal strategy is still: **don't play**

### What the Demo Shows

Even with a **learning algorithm**:
1. It takes many plays to identify the best machine
2. During learning, you lose money to exploration
3. Even after learning, variance means losses continue

The demo uses machines with positive expected value to show the algorithm working, but in reality, no strategy beats a negative-sum game.

---

## Extensions

### Contextual Bandits

State-dependent rewards:

$$R_i(s) \sim \mathcal{D}_i(s)$$

The reward depends on context $s$ (e.g., time of day, user features).

### Non-Stationary Bandits

Reward distributions change over time:

$$\mu_i(t) = f(t)$$

Requires algorithms that "forget" old information (e.g., sliding window, exponential decay).

### Thompson Sampling

Bayesian approach that maintains posterior distributions:

$$P(\mu_i | \text{data}) \propto P(\text{data} | \mu_i) P(\mu_i)$$

Often outperforms ε-Greedy and Softmax in practice.

---

## References

1. Sutton, R. S., & Barto, A. G. (2018). *Reinforcement Learning: An Introduction*. Chapter 2.
2. Lai, T. L., & Robbins, H. (1985). "Asymptotically efficient adaptive allocation rules". *Advances in Applied Mathematics*.
3. Auer, P., Cesa-Bianchi, N., & Fischer, P. (2002). "Finite-time Analysis of the Multiarmed Bandit Problem". *Machine Learning*.

---

**Disclaimer**: This is an educational demonstration. Slot machines in real casinos have negative expected value by design. No algorithm can beat them in the long run.
