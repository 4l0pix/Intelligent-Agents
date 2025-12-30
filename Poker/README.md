# Texas Hold'em Poker - Monte Carlo Hand Evaluation

## Overview

A heads-up (1v1) Texas Hold'em poker game against Carlos, an AI that uses **Monte Carlo simulation** to estimate hand strength and make optimal decisions based on pot odds.

---

## The Mathematics of Poker

### Game Complexity

Texas Hold'em is one of the most complex games studied in AI:

| Metric | Value |
|--------|-------|
| **Game States** | ~$10^{18}$ |
| **Information Sets** | ~$10^{14}$ |
| **Decision Points** | ~$10^{71}$ per game tree |
| **Branching Factor** | Variable (bet sizing) |

For comparison:
- Chess: ~$10^{47}$ game states
- Go: ~$10^{170}$ game states
- Poker's challenge: **imperfect information** (hidden cards)

### Why Monte Carlo?

Exhaustive analysis is impossible. Monte Carlo simulation provides:
1. **Statistical estimates** of hand strength
2. **Scalable computation** (adjustable accuracy vs. speed)
3. **No precomputed tables** needed

---

## Hand Strength Calculation

### Equity Definition

**Equity** is your expected share of the pot, given all possible outcomes:

$$\text{Equity} = P(\text{win}) + \frac{1}{2} P(\text{tie})$$

### Monte Carlo Estimation

We estimate equity by random sampling:

$$\hat{\text{Equity}} = \frac{1}{N} \sum_{i=1}^{N} \text{outcome}_i$$

Where $\text{outcome}_i \in \{0, 0.5, 1\}$ (loss, tie, win).

### Algorithm

```
function MonteCarloEquity(holeCards, communityCards, N):
    wins = 0
    ties = 0
    deck = remove(fullDeck, holeCards ∪ communityCards)
    
    for i = 1 to N:
        shuffle(deck)
        
        # Complete community cards if needed
        remainingCommunity = deal(deck, 5 - |communityCards|)
        fullCommunity = communityCards ∪ remainingCommunity
        
        # Deal opponent's hole cards
        opponentHole = deal(deck, 2)
        
        # Evaluate hands
        myHand = bestFiveCardHand(holeCards, fullCommunity)
        oppHand = bestFiveCardHand(opponentHole, fullCommunity)
        
        if myHand > oppHand: wins += 1
        elif myHand == oppHand: ties += 1
    
    return (wins + 0.5 * ties) / N
```

### Convergence

By the Central Limit Theorem, the error decreases as:

$$\sigma_{\hat{\text{Equity}}} = \frac{\sigma}{\sqrt{N}} \approx \frac{0.5}{\sqrt{N}}$$

| Simulations | Standard Error | 95% Confidence Interval |
|-------------|----------------|-------------------------|
| 100 | 5.0% | ±10% |
| 400 | 2.5% | ±5% |
| 1000 | 1.6% | ±3.2% |
| 10000 | 0.5% | ±1% |

---

## Hand Evaluation Mathematics

### Hand Rankings

Hands are ranked by this hierarchy (high to low):

| Rank | Hand | Example | Combinations | Probability |
|------|------|---------|--------------|-------------|
| 1 | Royal Flush | A♠ K♠ Q♠ J♠ 10♠ | 4 | 0.00015% |
| 2 | Straight Flush | 9♦ 8♦ 7♦ 6♦ 5♦ | 36 | 0.00139% |
| 3 | Four of a Kind | K♣ K♠ K♦ K♥ 2♣ | 624 | 0.024% |
| 4 | Full House | Q♠ Q♥ Q♦ 7♣ 7♠ | 3,744 | 0.144% |
| 5 | Flush | A♥ J♥ 8♥ 4♥ 2♥ | 5,108 | 0.197% |
| 6 | Straight | 10♣ 9♠ 8♦ 7♥ 6♣ | 10,200 | 0.393% |
| 7 | Three of a Kind | 8♠ 8♦ 8♣ K♥ 3♠ | 54,912 | 2.11% |
| 8 | Two Pair | J♠ J♥ 5♦ 5♣ A♠ | 123,552 | 4.75% |
| 9 | One Pair | 10♠ 10♣ A♦ 7♥ 4♣ | 1,098,240 | 42.3% |
| 10 | High Card | A♠ J♦ 8♣ 5♥ 2♠ | 1,302,540 | 50.1% |

**Total**: 2,598,960 possible 5-card hands

### Combination Selection (7 Choose 5)

In Hold'em, you pick the best 5 from 7 cards:

$$\binom{7}{5} = 21 \text{ possible combinations}$$

Each combination is evaluated and the best is kept.

### Tie-Breaking with Kickers

When hand types match, compare kickers (remaining cards) in order:

```
Hand A: A♠ A♥ K♦ 7♣ 2♠  (Pair of Aces, K-7-2 kickers)
Hand B: A♦ A♣ K♠ 6♥ 5♦  (Pair of Aces, K-6-5 kickers)

Compare: A=A, A=A, K=K, 7>6 → Hand A wins
```

---

## Decision Making: Pot Odds Theory

### What Are Pot Odds?

**Pot odds** = ratio of the cost to call vs. the total pot after calling:

$$\text{Pot Odds} = \frac{\text{Call Amount}}{\text{Pot} + \text{Call Amount}}$$

### Example

- Pot: $100
- Opponent bets: $50
- Your call cost: $50
- Pot after call: $200

$$\text{Pot Odds} = \frac{50}{200} = 25\%$$

You need at least 25% equity to profitably call.

### The Decision Rule

$$\text{If } \text{Equity} > \text{Pot Odds} \implies \text{Call is profitable}$$

**Mathematical proof**:

Expected value of calling:
$$EV_{call} = \text{Equity} \times (\text{Pot} + \text{Call}) - \text{Call}$$

Call is profitable when $EV_{call} > 0$:
$$\text{Equity} > \frac{\text{Call}}{\text{Pot} + \text{Call}} = \text{Pot Odds}$$

### Extended Decision Framework

Carlos uses:

```python
def make_decision(equity, pot_odds, pot_size, hand_strength):
    
    # Clear fold: equity too low
    if equity < pot_odds - 0.1:
        if random() < bluff_frequency:
            return raise(bluff_amount)  # Bluff
        return fold()
    
    # Marginal: equity slightly above pot odds
    if pot_odds - 0.1 <= equity < pot_odds + 0.1:
        return call()
    
    # Strong: equity well above pot odds
    if equity >= pot_odds + 0.1:
        if equity > raise_threshold:
            return raise(value_bet_size)  # Value bet
        return call()
```

---

## Implied Odds and Reverse Implied Odds

### Implied Odds

Future money you might win if you hit your draw:

$$\text{Implied Odds} = \frac{\text{Call}}{\text{Pot} + \text{Call} + \text{Expected Future Winnings}}$$

**Example**: You have a flush draw (9 outs, ~19% to hit). Pot is $100, call is $50.
- Direct pot odds: 33% (need more equity)
- But if opponent will bet another $100 when you hit: Implied odds = $50/($200 + $100) = 17%
- Now 19% > 17%, so calling is profitable

### Reverse Implied Odds

Money you might lose when you make your hand but opponent has better:

**Example**: You have a small flush draw. If you hit, opponent might have a bigger flush.

---

## Game Theory: Bluffing Mathematics

### Optimal Bluff Frequency

In game theory, the optimal bluff frequency makes the opponent **indifferent** between calling and folding.

For a pot-sized bet ($P$ into $P$):
- Opponent must call $P$ to win $2P$
- They need 33% equity to call profitably
- You should bluff 33% as often as you value bet

### Bluff-to-Value Ratio

$$\frac{\text{Bluffs}}{\text{Value Bets}} = \frac{\text{Pot Odds Opponent Gets}}{1 - \text{Pot Odds}}$$

For pot-sized bet: $\frac{0.33}{0.67} = 0.5$ (bluff half as often as value bet)

### Carlos's Bluffing

By difficulty level:

| Difficulty | Bluff Frequency | Explanation |
|------------|-----------------|-------------|
| Easy | 20% | Over-bluffs, exploitable |
| Medium | 15% | Slightly unbalanced |
| Hard | 10% | Near-optimal, hard to exploit |

---

## Position and Information

### Position Advantage

Acting last provides:
1. **More information** - See opponent's action first
2. **Pot control** - Easier to manage pot size
3. **Bluffing opportunities** - Can bluff when opponent shows weakness

**Mathematical impact**: Position is worth approximately 10-15% equity in close decisions.

### Information Set Complexity

At any decision point, the player has an **information set** containing:
- Their hole cards (known)
- Community cards (known)
- Betting history (known)
- Opponent's hole cards (unknown - 1081 possible holdings)

---

## Expected Value Calculations

### Preflop Hand Strength

Starting hand equities vs. random hand (heads-up):

| Hand | Equity vs. Random |
|------|-------------------|
| AA | 85.2% |
| KK | 82.4% |
| AKs | 67.0% |
| QQ | 79.9% |
| JJ | 77.5% |
| AKo | 65.3% |
| 72o | 34.6% |

### Outs and Probabilities

**Outs** = cards that improve your hand to likely winner

| Draw | Outs | P(Hit Turn) | P(Hit River) | P(Hit Either) |
|------|------|-------------|--------------|---------------|
| Flush draw | 9 | 19.1% | 19.6% | 35.0% |
| Open-ended straight | 8 | 17.0% | 17.4% | 31.5% |
| Gutshot straight | 4 | 8.5% | 8.7% | 16.5% |
| Two overcards | 6 | 12.8% | 13.0% | 24.1% |
| Set to full house | 7 | 14.9% | 15.2% | 27.8% |

**Rule of 2 and 4**:
- One card to come: Outs × 2 ≈ Probability
- Two cards to come: Outs × 4 ≈ Probability

---

## Implementation Details

### Card Representation

```javascript
const card = {
    suit: 'hearts' | 'diamonds' | 'clubs' | 'spades',
    rank: 2-14  // 2-10, 11=J, 12=Q, 13=K, 14=A
};
```

### Hand Comparison

```javascript
function compareHands(hand1, hand2) {
    // Compare rank types first
    if (hand1.rankType !== hand2.rankType) {
        return hand1.rankType - hand2.rankType;
    }
    
    // Same rank type: compare kickers
    for (let i = 0; i < hand1.kickers.length; i++) {
        if (hand1.kickers[i] !== hand2.kickers[i]) {
            return hand1.kickers[i] - hand2.kickers[i];
        }
    }
    
    return 0;  // Exact tie
}
```

### Performance

| Operation | Time |
|-----------|------|
| Single hand evaluation | ~0.01ms |
| 400 Monte Carlo simulations | ~5ms |
| Full decision calculation | ~10ms |

---

## Difficulty Levels

| Setting | Easy | Medium | Hard |
|---------|------|--------|------|
| MC Simulations | 200 | 400 | 600 |
| Bluff Frequency | 20% | 15% | 10% |
| Call Threshold | Pot Odds - 5% | Pot Odds | Pot Odds + 5% |
| Raise Threshold | 60% | 70% | 75% |

---

## Why Poker Is Hard for AI

1. **Imperfect Information**: Hidden cards create uncertainty
2. **Game-Theoretic Complexity**: Optimal play requires mixed strategies
3. **Opponent Modeling**: Exploiting suboptimal opponents vs. being unexploitable
4. **Deep Strategy**: Multi-level thinking ("I know that he knows that I know...")

### State of the Art

- **Libratus (2017)**: Beat top humans at heads-up no-limit
- **Pluribus (2019)**: Beat 6-player no-limit poker
- Both use **Counterfactual Regret Minimization (CFR)** - too computationally intensive for browser

Monte Carlo provides a **practical approximation** suitable for real-time play.

---

## References

1. Billings, D. et al. (2002). "The Challenge of Poker". *Artificial Intelligence*.
2. Bowling, M. et al. (2015). "Heads-up limit hold'em poker is solved". *Science*.
3. Brown, N. & Sandholm, T. (2019). "Superhuman AI for multiplayer poker". *Science*.
4. Sklansky, D. (1999). *The Theory of Poker*. Two Plus Two Publishing.

---

**Disclaimer**: This is an educational demonstration. Poker involves skill but also significant variance. In casino poker, the house takes a rake from every pot, making it a negative-sum game for players collectively.
