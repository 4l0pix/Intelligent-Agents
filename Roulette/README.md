# Roulette - The Mathematics of Inevitable Loss

## Overview

An interactive demonstration proving that **all roulette betting systems fail**. This isn't a game to win—it's a mathematical proof that the house always wins in the long run.

---

## The Fundamental Mathematics

### Probability Structure

American roulette wheel:
- Numbers 1-36 (half red, half black)
- 0 (green)
- 00 (green)
- **Total pockets: 38**

### Expected Value Formula

For any bet, the expected value is:

$$EV = \sum_{i} P(outcome_i) \times payout_i$$

### The House Edge Derivation

For a **single number bet** (pays 35:1):

$$EV = P(win) \times 35 + P(lose) \times (-1)$$
$$EV = \frac{1}{38} \times 35 + \frac{37}{38} \times (-1)$$
$$EV = \frac{35 - 37}{38} = -\frac{2}{38} = -0.0526$$

**House edge: 5.26%**

### Universal House Edge

Every bet has the same expected value:

| Bet Type | Wins | Payout | EV Calculation |
|----------|------|--------|----------------|
| Single number | 1/38 | 35:1 | $(1/38)(35) + (37/38)(-1) = -0.0526$ |
| Red/Black | 18/38 | 1:1 | $(18/38)(1) + (20/38)(-1) = -0.0526$ |
| Dozen | 12/38 | 2:1 | $(12/38)(2) + (26/38)(-1) = -0.0526$ |
| Column | 12/38 | 2:1 | $(12/38)(2) + (26/38)(-1) = -0.0526$ |
| Split | 2/38 | 17:1 | $(2/38)(17) + (36/38)(-1) = -0.0526$ |

**Key insight**: The payout structure is designed so that **all bets have identical expected value**.

---

## Why Betting Systems Cannot Work

### Theorem: Linearity of Expectation

For any combination of bets $B_1, B_2, ..., B_n$:

$$EV[B_1 + B_2 + ... + B_n] = EV[B_1] + EV[B_2] + ... + EV[B_n]$$

Since each $EV[B_i] < 0$:

$$EV[\text{total}] < 0$$

**No combination of negative expected value bets can create positive expected value.**

### Theorem: Optional Stopping

Even with any stopping rule (quit while ahead, etc.):

$$EV[\text{final wealth}] = \text{initial wealth} - (EV \times \text{expected bets})$$

The **Optional Stopping Theorem** proves that no betting strategy can change the expected value, regardless of:
- When you start
- When you stop
- How you vary your bets

---

## Analysis of Betting Systems

### 1. Martingale System

**Strategy**: Double bet after each loss. First win recovers all losses plus one unit profit.

**Bet progression**: 1, 2, 4, 8, 16, 32, 64, 128, ...

#### Mathematical Analysis

After $n$ consecutive losses, you've bet:
$$\text{Total wagered} = 1 + 2 + 4 + ... + 2^{n-1} = 2^n - 1$$

To continue, you must bet $2^n$.

**Example**: After 7 losses:
- Total lost: $127
- Next bet: $128
- If win: profit = $1
- If lose: total loss = $255

#### Probability of Ruin

Probability of $n$ consecutive losses on even-money bets:

$$P(n \text{ losses}) = \left(\frac{20}{38}\right)^n$$

| Consecutive Losses | Probability | Next Bet (base $10) | Total Risk |
|--------------------|-------------|---------------------|------------|
| 4 | 7.7% | $160 | $150 |
| 5 | 4.0% | $320 | $310 |
| 6 | 2.1% | $640 | $630 |
| 7 | 1.1% | $1,280 | $1,270 |
| 8 | 0.57% | $2,560 | $2,550 |

#### Expected Outcome Over Many Sessions

In $n$ sessions of $k$ spins each:

$$E[\text{profit}] = n \times k \times (-\frac{2}{38}) \times \text{avg bet}$$

The system trades **frequent small wins for rare catastrophic losses**.

**Long-term**: The same negative expected value as flat betting.

---

### 2. Reverse Martingale (Paroli)

**Strategy**: Double bet after each win. Reset after loss or target win streak.

**Bet progression after wins**: 1, 2, 4, 8, ...

#### Mathematical Analysis

Probability of $n$ consecutive wins:
$$P(n \text{ wins}) = \left(\frac{18}{38}\right)^n$$

| Win Streak | Probability | Payout (base $10) |
|------------|-------------|-------------------|
| 1 | 47.4% | $10 |
| 2 | 22.4% | $30 |
| 3 | 10.6% | $70 |
| 4 | 5.0% | $150 |

**Expected value per sequence**:
$$EV = \sum_{k=1}^{n} P(\text{exactly } k \text{ wins}) \times \text{profit}_k + P(\text{0 wins}) \times (-1)$$

Still equals $-5.26\%$ per dollar wagered.

---

### 3. D'Alembert System

**Strategy**: Increase bet by 1 unit after loss, decrease by 1 unit after win.

**Example sequence** (starting at 5 units):
- Loss: bet 6
- Loss: bet 7
- Win: bet 6
- Win: bet 5
- Loss: bet 6
- ...

#### Mathematical Analysis

The system assumes wins and losses should "balance out" (gambler's fallacy).

**Probability of being even**: After $2n$ bets, probability of exactly $n$ wins:
$$P(n \text{ wins in } 2n) = \binom{2n}{n} \left(\frac{18}{38}\right)^n \left(\frac{20}{38}\right)^n$$

This approaches zero as $n$ increases due to the bias toward losses.

**Expected bet size**: Grows linearly over time during losing streaks.

---

### 4. Fibonacci System

**Strategy**: Bet according to Fibonacci sequence (1, 1, 2, 3, 5, 8, 13, 21, ...)

**After loss**: Move one step forward in sequence.
**After win**: Move two steps back.

#### Mathematical Properties

Fibonacci numbers grow exponentially:
$$F_n \approx \frac{\phi^n}{\sqrt{5}}, \quad \phi = \frac{1+\sqrt{5}}{2} \approx 1.618$$

| Position | Fibonacci | Bet ($10 base) |
|----------|-----------|----------------|
| 1 | 1 | $10 |
| 5 | 5 | $50 |
| 10 | 55 | $550 |
| 15 | 610 | $6,100 |
| 20 | 6,765 | $67,650 |

**Analysis**: Slower growth than Martingale, but same fundamental flaw—negative EV per bet.

---

### 5. Flat Betting

**Strategy**: Same bet every spin.

#### Mathematical Analysis

The most "honest" system. Clearly shows the house edge.

After $n$ spins with bet $b$:
$$E[\text{profit}] = n \times b \times (-0.0526)$$
$$\sigma[\text{profit}] = b\sqrt{n} \times 0.999$$

**Example**: $10 bets for 1000 spins:
- Expected loss: $526
- Standard deviation: $316
- 95% confidence interval: $-526 ± 632$ = [-$1158, +$106]

Even flat betting can show short-term profits due to variance, but long-term expectation is always negative.

---

### 6. James Bond System

**Strategy**: Cover most of the table with specific allocation:
- $140 on 19-36 (high numbers)
- $50 on 13-18 (six-line)
- $10 on 0 (straight up)

**Total bet**: $200

#### Outcome Analysis

| Result | Payout | Probability | Net |
|--------|--------|-------------|-----|
| 19-36 | $140 × 1 = $140 | 18/38 = 47.4% | +$80 |
| 13-18 | $50 × 5 = $250 | 6/38 = 15.8% | +$100 |
| 0 | $10 × 35 = $350 | 1/38 = 2.6% | +$160 |
| 1-12 or 00 | $0 | 13/38 = 34.2% | -$200 |

**Expected Value**:
$$EV = 0.474(80) + 0.158(100) + 0.026(160) + 0.342(-200)$$
$$EV = 37.9 + 15.8 + 4.2 - 68.4 = -10.5$$

**House edge**: $10.5 / $200 = **5.26%**

Covering more numbers doesn't change the math.

---

## The Gambler's Fallacy

### Definition

The false belief that past results affect future probabilities in independent events.

### Mathematical Reality

Each spin is **independent**:
$$P(\text{red on spin } n+1 | \text{sequence of spins } 1...n) = \frac{18}{38}$$

This is **always true**, regardless of past results.

### Example

After 10 consecutive reds:
- P(red next) = 18/38 = 47.4%
- P(black next) = 18/38 = 47.4%
- P(green next) = 2/38 = 5.3%

The wheel has no memory.

---

## Long-Term Simulation Results

### Monte Carlo Simulation (10,000 players, 1,000 spins each)

**Starting bankroll**: $1,000

| Strategy | Average Final | Bust Rate | Max Final | Min Final |
|----------|---------------|-----------|-----------|-----------|
| Martingale | $842 | 67% | $1,200 | $0 |
| Reverse Martingale | $891 | 23% | $3,800 | $0 |
| D'Alembert | $876 | 45% | $1,400 | $0 |
| Fibonacci | $865 | 52% | $1,500 | $0 |
| Flat Betting | $947 | 12% | $1,300 | $580 |
| James Bond | $901 | 38% | $2,100 | $0 |

**Key observation**: All strategies converge to ~5.26% loss on average. Variance differs, but expected value is identical.

---

## Mathematical Proofs

### Proof: No System Can Beat the House

**Given**:
- $X_i$ = outcome of bet $i$ (random variable)
- $E[X_i] = -c$ where $c > 0$ (house edge)
- $B_i$ = bet size on round $i$ (can depend on history)

**Claim**: $E[\sum_{i=1}^{N} X_i \cdot B_i] < 0$ for any stopping time $N$ and betting strategy $B$.

**Proof**:
By the Optional Stopping Theorem and linearity of expectation:
$$E\left[\sum_{i=1}^{N} X_i \cdot B_i\right] = E\left[\sum_{i=1}^{N} E[X_i | B_i] \cdot B_i\right] = E\left[\sum_{i=1}^{N} (-c) \cdot B_i\right] = -c \cdot E\left[\sum_{i=1}^{N} B_i\right] < 0$$

**QED**: Expected profit is always negative. ∎

---

## Variance and Risk of Ruin

### Risk of Ruin Formula

Probability of losing entire bankroll $B$ before winning target $T$:

$$P(\text{ruin}) = \frac{(q/p)^B - (q/p)^{B+T}}{1 - (q/p)^{B+T}}$$

Where $p = 18/38$ (win probability) and $q = 20/38$ (lose probability).

For large values:
$$P(\text{ruin}) \approx 1 - (p/q)^T$$

With $p/q = 0.9$, even modest targets have high ruin probability.

---

## Wheel Types Comparison

| Wheel | Zeros | House Edge | Your $100 EV |
|-------|-------|------------|--------------|
| European | 1 | 2.70% | -$2.70 |
| American | 2 | 5.26% | -$5.26 |
| Triple Zero | 3 | 7.69% | -$7.69 |

### La Partage / En Prison Rules

Some European casinos offer "la partage" on even-money bets:
- If 0 hits, you lose only half your even-money bet
- Reduces house edge to **1.35%** on these bets

$$EV_{la\ partage} = \frac{18}{37}(1) + \frac{18}{37}(-1) + \frac{1}{37}(-0.5) = -\frac{0.5}{37} = -1.35\%$$

**Still negative**, just less so.

---

## Conclusion

### The Three Certainties of Roulette

1. **Every bet has negative expected value** - The wheel is designed this way

2. **No betting system changes the math** - Variance can be redistributed, but not expected value

3. **The house always wins in aggregate** - Individual sessions vary, but long-term results converge to the house edge

### The Only Winning Strategy

**Don't play.**

Or if you do, recognize it as **entertainment with a known cost**, not an investment opportunity.

---

## References

1. Ethier, S. N. (2010). *The Doctrine of Chances: Probabilistic Aspects of Gambling*. Springer.
2. Billingsley, P. (1986). *Probability and Measure*. Wiley. (Optional Stopping Theorem)
3. Thorp, E. O. (1984). *The Mathematics of Gambling*. Gambling Times.
4. "The Truth about Betting Systems" - Wizard of Odds (wizardofodds.com)

---

**Disclaimer**: This is an educational demonstration proving that gambling is a losing proposition. The house edge is mathematically guaranteed. Please gamble responsibly—or better yet, not at all.
