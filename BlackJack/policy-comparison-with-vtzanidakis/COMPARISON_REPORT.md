# Blackjack Monte Carlo ES - Algorithm Comparison Report

## Overview

This document explains the comparison between two Monte Carlo Exploring Starts (MC-ES) implementations for learning optimal Blackjack policies:

1. **Koukosias Implementation** - Athanasios Koukosias (2025-2026)
2. **Tzanidakis Implementation** - V. Tzanidakis

Both algorithms are trained on **identical episodes** using a shared random seed, yet they produce **different policies**. This document explains why.

---

## The Critical Difference

### Why Do They Produce Different Policies?

Even though both algorithms receive the same:
- Initial player/dealer hands
- Exploring start (random initial state + action)
- Sequence of card draws

**They diverge after the first action because they use different behavior policies.**

### Koukosias Implementation (Proper MC-ES)

```
After the exploring start action:
    action = self.player_policy(state)  # Uses LEARNED Q-values
```

The Koukosias implementation follows the **on-policy** nature of Monte Carlo ES correctly:
- The first action is random (exploring start)
- All subsequent actions use the **current greedy policy** based on learned Q-values
- As Q-values improve, the behavior policy improves
- This creates a feedback loop where better Q-values → better episodes → better Q-values

### Tzanidakis Implementation (Fixed Behavior Policy)

```
After the exploring start action:
    action = self.initial_policy(state)  # Uses FIXED policy (hit if <20)
```

The Tzanidakis implementation uses a **fixed behavior policy**:
- The first action is random (exploring start)
- All subsequent actions follow a **fixed rule**: "hit if sum < 20, else stick"
- The learned Q-values are **never used** during episode generation
- Q-values are computed under a different policy than the one being evaluated

### Consequence

| Aspect | Koukosias | Tzanidakis |
|--------|-----------|------------|
| Behavior Policy | Improves over time | Fixed forever |
| Episode Quality | Gets better as learning progresses | Constant throughout |
| Q-value Accuracy | Estimates optimal action values | Estimates "hit until 20" action values |
| Final Policy | Closer to true optimal | Biased toward conservative play |

---

## How the Code Works

### Training Phase

1. **SharedEpisodeGenerator** creates identical random data for each episode:
   - Pre-generates 30 cards per episode
   - Randomly selects exploring start (state + action)
   - Both algorithms receive copies of this data

2. **Both algorithms play the episode**:
   - They start from the same state with the same first action
   - They draw the same cards in sequence
   - BUT they may take different subsequent actions (due to different behavior policies)

3. **Q-value updates**:
   - Both use first-visit Monte Carlo
   - Both update Q(state, action) based on episode returns
   - Koukosias processes in reverse order; Tzanidakis processes forward (equivalent result)

### Statistics Tracking

We track detailed statistics for each (state, action) pair:
- Total episodes encountered
- Player bust count
- Dealer bust count
- Win/loss/draw counts
- Final hand sums

This allows us to explain **why** each policy decision was made.

---

## What to Expect When Running

### Console Output

1. **Training progress** (takes ~30-60 seconds):
   ```
   Episode 100,000/500,000
   Episode 200,000/500,000
   ...
   ```

2. **Policy comparison tables** showing both policies side-by-side

3. **List of policy differences** with exact states where algorithms disagree

### Generated Plots (11 figures total)

**Warning: Many windows will open simultaneously. We recommend closing them in order to examine each properly.**

| Plot # | Description | Subplots | What We Show |
|--------|-------------|----------|--------------|
| 1 | Koukosias Stats (Usable Ace) | 12 | Policy, Q-values, bust/win rates, rewards |
| 2 | Koukosias Stats (No Usable Ace) | 12 | Same as above |
| 3 | Tzanidakis Stats (Usable Ace) | 12 | Same as above |
| 4 | Tzanidakis Stats (No Usable Ace) | 12 | Same as above |
| 5 | Comparison (Usable Ace) | 16 | Side-by-side statistics |
| 6 | Comparison (No Usable Ace) | 16 | Side-by-side statistics |
| 7 | Policy Differences | 6 | Highlights disagreements |
| 8 | Interactive Koukosias (Usable Ace) | 1 | Clickable policy grid |
| 9 | Interactive Koukosias (No Usable Ace) | 1 | Clickable policy grid |
| 10 | Interactive Tzanidakis (Usable Ace) | 1 | Clickable policy grid |
| 11 | Interactive Tzanidakis (No Usable Ace) | 1 | Clickable policy grid |

### Saved PNG Files

We automatically save the following to the working directory:
- `blackjack_koukosias_usable_ace.png`
- `blackjack_koukosias_no_usable_ace.png`
- `blackjack_tzanidakis_usable_ace.png`
- `blackjack_tzanidakis_no_usable_ace.png`
- `blackjack_comparison_usable_ace.png`
- `blackjack_comparison_no_usable_ace.png`
- `blackjack_policy_differences.png`

### Interactive Features

The last 4 plots are **clickable**:
- Click any cell in the policy grid
- A new figure opens showing:
  - Q-value bar chart comparison
  - Outcome distribution pie chart
  - Bust rate bar chart
  - Player/Dealer final sum histograms
  - Decision analysis text

---

## Interpreting the Results

### Expected Policy Differences

We expect **21-32 states** where the algorithms disagree (out of 200 total states).

Most differences occur where:
- Player has a low sum (12-17) with usable ace
- Tzanidakis tends to STICK more (conservative)
- Koukosias tends to HIT more (aggressive but correct)

### Why Koukosias is More Accurate

The Koukosias policy more closely matches the **known optimal Blackjack policy**:
- Hit soft 17 or less
- Stand on hard 17+
- Hit soft 18 vs dealer 9, 10, A

Tzanidakis over-sticks because its fixed "hit until 20" behavior policy doesn't explore the consequences of standing on lower totals properly.

---

## Code Structure Summary

```
combined_comparison.py
│
├── SharedEpisodeGenerator    # We generate identical episodes for both algorithms
├── StatisticsTracker         # We track outcomes per state-action pair
│
├── KoukosiasMCES            # We use learned Q-values for action selection
├── TzanidakisMCES           # We use fixed policy for action selection
│
├── train_both_algorithms()   # We train on same episodes
├── create_all_plots()        # We generate all visualizations
│   ├── create_algorithm_stats_multiplot()    # 12-subplot stats
│   ├── create_comparison_multiplot()          # Side-by-side comparison
│   ├── create_policy_difference_plot()        # Difference highlighting
│   └── InteractivePolicyPlot                  # Clickable analysis
│
└── main()                    # We orchestrate everything
```

---

## Conclusion

The key takeaway is that **identical training data does not guarantee identical policies** when the algorithms differ in their behavior policies during episode generation.

Monte Carlo Exploring Starts requires that we follow the **current estimate of the optimal policy** after the random exploring start. The Tzanidakis implementation's use of a fixed behavior policy technically violates this requirement, leading to suboptimal Q-value estimates and consequently a different (less optimal) extracted policy.

This comparison demonstrates the importance of correctly implementing the on-policy nature of Monte Carlo methods.
