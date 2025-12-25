
# Reinforce Jack to Learn
### Description
The object of the popular casino card game Blackjack is to get cards that add up to the highest possible total, without exceeding 21. All face cards count as 10, while an ace can count as either 1 or 11. The game starts with two cards dealt to both the dealer and the player. One of the dealer's cards is face up and the other is face down. If the player has 21 straight away (ace and 10), it is called a natural. Therefore, the player wins, unless the dealer also has a natural, in which case the game is a tie. If the player does not have a natural, he can ask for additional cards, one by one (hits), until he stops (sticks) or goes over 21 (busts and loses the game). If he sticks, it is the dealer's turn. The dealer hits or sticks according to a fixed
strategy: he sticks when he has accumulated 17 or more, otherwise he hits. If the dealer busts, then the player wins. Otherwise, the outcome — win, lose, or tie — is determined by the final cumulative total that is closest to
21.

# Our Case
Each Blackjack game is an episode. The payoffs +1, -1, and 0 are given for a win,
loss, and tie, respectively. All payoffs within a game are zero
and we do not discount (γ = 1). Therefore, these terminal payoffs are also
the returns. The player's actions are either hit or stick. The situations depend on
the player's cards and the dealer's card. We assume that the cards are dealt from an infinite deck (i.e., with replacement), so that there is no advantage in keeping track of cards that have already been dealt. If the player holds an ace that could count as 11 without being busted, then the ace is said to be usable. In this case it always counts as 11, because counting it as 1 will make the sum 11 or less, in which case there is no decision to make, because, obviously, the player must always hit. Thus, the player makes decisions based on three variables: his current total (12 – 21), the dealer’s first card (1 – 10), and whether or not he holds a usable ace. This makes a total of 200 situations. We want to find the optimal policy by applying the Monte Carlo method Exploring Starts (ES). • The player's initial policy is to stick only when the sum is
20 or 21.

# Solution

## How the Code Works

The solution is implemented in `blackjack.py` using the **Monte Carlo Exploring Starts (ES)** algorithm to find the optimal Blackjack policy.

### Algorithm Overview

1. **State Representation**: Each state is defined by three variables:
   - Player's current sum (12-21)
   - Dealer's face-up card (1-10, where 1 = Ace)
   - Whether the player has a usable ace (True/False)
   
   This creates 200 unique states (10 × 10 × 2).

2. **Exploring Starts**: To ensure all state-action pairs are visited, each episode begins from a randomly selected state with a randomly chosen action (Hit or Stick). This guarantees exploration of the entire state-action space.

3. **Episode Generation**: After the initial state-action pair, the player follows the current greedy policy (choosing the action with the highest Q-value). The dealer follows the fixed strategy: hit on sum < 17, stick otherwise.

4. **First-Visit Monte Carlo Update**: For each state-action pair encountered for the first time in an episode:
   - The return (final reward) is recorded
   - The Q-value is updated as the average of all observed returns
   - Since γ = 1 (no discounting), the return equals the terminal reward (+1 win, -1 loss, 0 tie)

5. **Policy Improvement**: After updating Q-values, the policy is implicitly improved by always selecting the action with the highest Q-value (greedy policy).

### Key Implementation Details

- **Infinite Deck**: Cards are drawn with replacement, so card counting provides no advantage
- **Initial Policy**: Player sticks only on 20 or 21 (used before Q-values are learned)
- **Training Episodes**: 500,000 episodes ensure convergence to the optimal policy

## Generated Plots

### 1. Optimal Policy Plot (`policy.png`)

This plot shows two grids representing the optimal action for each state:

- **Left Grid**: States with a usable ace
- **Right Grid**: States without a usable ace

**How to read**:
- **X-axis**: Dealer's face-up card (A, 2-10)
- **Y-axis**: Player's sum (12-21)
- **Green cells (S)**: Stick is optimal
- **Red cells (H)**: Hit is optimal

**Key insights from the optimal policy**:
- Without a usable ace: Stick on 17+ always; for 12-16, stick against dealer's weak cards (2-6), hit otherwise
- With a usable ace: Be more aggressive—hit on most hands since busting is impossible; stick only on 19-21 and on 18 against dealer's 2-8

### 2. State-Value Function Plot (`value_function.png`)

This 3D surface plot shows the expected return (value) from each state when following the optimal policy:

- **X-axis**: Dealer's showing card
- **Y-axis**: Player's sum
- **Z-axis**: State value (expected return, ranging roughly from -0.5 to +1.0)

**Key insights**:
- Higher player sums (19-21) have positive values (favorable states)
- Lower player sums against high dealer cards have negative values (unfavorable)
- The surface is smoother for states without a usable ace
- States with a usable ace show more variation due to the flexibility the ace provides

## Running the Code

```bash
python blackjack.py
```

The program will:
1. Train for 500,000 episodes (~1 minute)
2. Print the optimal policy as a text table
3. Evaluate the policy over 100,000 test games
4. Generate and save the two visualization plots