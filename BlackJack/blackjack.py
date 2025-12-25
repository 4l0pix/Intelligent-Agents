"""
Blackjack - Monte Carlo Exploring Starts (ES)
Finds the optimal policy for playing Blackjack using Monte Carlo method.
"""

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import random


# Actions
HIT = 0
STICK = 1
ACTIONS = [HIT, STICK]

# Card values
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  # A, 2-10, J, Q, K


def draw_card():
    """Draw a card from an infinite deck."""
    return random.choice(CARD_VALUES)


def draw_hand():
    """Draw initial two cards."""
    return [draw_card(), draw_card()]


def usable_ace(hand):
    """Check if hand has a usable ace (can count as 11 without busting)."""
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    """Return the sum of the hand, treating ace as 11 if beneficial."""
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    """Check if hand is busted (over 21)."""
    return sum_hand(hand) > 21


def get_state(player_hand, dealer_showing):
    """
    Get state representation.
    State: (player_sum, dealer_showing, usable_ace)
    - player_sum: 12-21 (below 12, player always hits)
    - dealer_showing: 1-10 (ace=1)
    - usable_ace: True/False
    """
    return (sum_hand(player_hand), dealer_showing, usable_ace(player_hand))


def player_policy(state, Q, policy_type='greedy'):
    """
    Player policy.
    - 'initial': stick only on 20 or 21
    - 'greedy': follow the learned Q-values
    """
    player_sum, dealer_showing, has_usable_ace = state
    
    if policy_type == 'initial':
        # Initial policy: stick only on 20 or 21
        return STICK if player_sum >= 20 else HIT
    else:
        # Greedy policy based on Q-values
        if Q[(state, HIT)] >= Q[(state, STICK)]:
            return HIT
        else:
            return STICK


def dealer_policy(dealer_hand):
    """
    Dealer's fixed strategy: hit if sum < 17, otherwise stick.
    """
    return HIT if sum_hand(dealer_hand) < 17 else STICK


def play_game(initial_state=None, initial_action=None, Q=None):
    """
    Play one episode of Blackjack.
    Returns: list of (state, action, reward) tuples
    
    If initial_state and initial_action are provided, use them (Exploring Starts).
    """
    # Initialize player's hand
    player_hand = draw_hand()
    
    # Initialize dealer's hand
    dealer_hand = draw_hand()
    dealer_showing = dealer_hand[0]  # First card is face up
    
    # For Exploring Starts: set up initial state
    if initial_state is not None:
        # We need to create a hand that matches the initial state
        player_sum, dealer_showing, has_usable_ace = initial_state
        
        # Create player hand matching the state
        if has_usable_ace:
            # Hand with usable ace: ace + (sum - 11)
            player_hand = [1, player_sum - 11]
        else:
            # Hand without usable ace
            if player_sum <= 11:
                player_hand = [player_sum]
            else:
                # Split into two cards
                player_hand = [10, player_sum - 10]
        
        # Create dealer hand with the showing card
        dealer_hand = [dealer_showing, draw_card()]
    
    # Episode history
    episode = []
    
    # Check for naturals (21 with initial two cards)
    player_sum = sum_hand(player_hand)
    
    # Player's turn
    while True:
        player_sum = sum_hand(player_hand)
        
        # If player sum < 12, always hit (no decision needed)
        if player_sum < 12:
            player_hand.append(draw_card())
            continue
        
        state = get_state(player_hand, dealer_showing)
        
        # Choose action
        if initial_action is not None:
            action = initial_action
            initial_action = None  # Only use initial action once
        elif Q is not None:
            action = player_policy(state, Q, 'greedy')
        else:
            action = player_policy(state, None, 'initial')
        
        episode.append((state, action))
        
        if action == STICK:
            break
        else:  # HIT
            player_hand.append(draw_card())
            if is_bust(player_hand):
                # Player busts, loses
                return [(s, a, 0) for s, a in episode[:-1]] + [(episode[-1][0], episode[-1][1], -1)]
    
    # Dealer's turn (only if player didn't bust)
    while dealer_policy(dealer_hand) == HIT:
        dealer_hand.append(draw_card())
    
    # Determine winner
    player_sum = sum_hand(player_hand)
    dealer_sum = sum_hand(dealer_hand)
    
    if is_bust(dealer_hand):
        reward = 1  # Dealer busts, player wins
    elif dealer_sum > player_sum:
        reward = -1  # Dealer wins
    elif dealer_sum < player_sum:
        reward = 1  # Player wins
    else:
        reward = 0  # Tie
    
    # Assign reward to all state-action pairs in episode
    return [(s, a, 0) for s, a in episode[:-1]] + [(episode[-1][0], episode[-1][1], reward)]


def monte_carlo_es(num_episodes=500000):
    """
    Monte Carlo Exploring Starts algorithm to find optimal policy.
    """
    # Initialize Q-values and returns
    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    # All possible states for exploring starts
    # player_sum: 12-21, dealer_showing: 1-10, usable_ace: True/False
    all_states = []
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            for has_usable_ace in [True, False]:
                all_states.append((player_sum, dealer_showing, has_usable_ace))
    
    for episode_num in range(num_episodes):
        if (episode_num + 1) % 100000 == 0:
            print(f"Episode {episode_num + 1}/{num_episodes}")
        
        # Exploring Starts: random initial state and action
        initial_state = random.choice(all_states)
        initial_action = random.choice(ACTIONS)
        
        # Generate episode
        episode = play_game(initial_state, initial_action, Q)
        
        # First-visit MC: update Q-values
        visited = set()
        G = 0  # Return (gamma = 1, so no discounting)
        
        # Process episode in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = G + reward  # gamma = 1
            
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                returns_sum[state_action] += G
                returns_count[state_action] += 1
                Q[state_action] = returns_sum[state_action] / returns_count[state_action]
    
    # Extract optimal policy
    policy = {}
    for state in all_states:
        if Q[(state, HIT)] >= Q[(state, STICK)]:
            policy[state] = HIT
        else:
            policy[state] = STICK
    
    return Q, policy


def plot_policy(policy, title_suffix=""):
    """Plot the optimal policy for usable and non-usable ace cases."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, usable_ace in enumerate([True, False]):
        ax = axes[idx]
        
        # Create policy grid
        policy_grid = np.zeros((10, 10))  # player_sum (12-21) x dealer_showing (1-10)
        
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                state = (player_sum, dealer_showing, usable_ace)
                action = policy.get(state, HIT)
                policy_grid[player_sum - 12, dealer_showing - 1] = action
        
        # Plot
        im = ax.imshow(policy_grid, cmap='RdYlGn', aspect='auto', 
                       origin='lower', vmin=0, vmax=1)
        
        ax.set_xticks(range(10))
        ax.set_xticklabels(['A'] + list(range(2, 11)))
        ax.set_yticks(range(10))
        ax.set_yticklabels(range(12, 22))
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        ax.set_title(f"Optimal Policy ({ace_str})")
        
        # Add legend
        for i in range(10):
            for j in range(10):
                text = "S" if policy_grid[i, j] == STICK else "H"
                color = 'white' if policy_grid[i, j] == STICK else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.suptitle(f"Blackjack Optimal Policy (Monte Carlo ES){title_suffix}")
    plt.tight_layout()
    plt.savefig('c:/Users/akoukosias/Documents/GitHub/Intelligent-Agents/BlackJack/policy.png', dpi=150)
    plt.show()


def plot_value_function(Q, title_suffix=""):
    """Plot the state-value function as 3D surface plots."""
    fig = plt.figure(figsize=(14, 5))
    
    for idx, usable_ace in enumerate([True, False]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        # Create value grid
        X = np.arange(1, 11)  # dealer showing
        Y = np.arange(12, 22)  # player sum
        X, Y = np.meshgrid(X, Y)
        
        Z = np.zeros_like(X, dtype=float)
        
        for i, player_sum in enumerate(range(12, 22)):
            for j, dealer_showing in enumerate(range(1, 11)):
                state = (player_sum, dealer_showing, usable_ace)
                # V(s) = max_a Q(s, a)
                Z[i, j] = max(Q[(state, HIT)], Q[(state, STICK)])
        
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        ax.set_title(f"State-Value Function ({ace_str})")
    
    plt.suptitle(f"Blackjack State-Value Function (Monte Carlo ES){title_suffix}")
    plt.tight_layout()
    plt.savefig('c:/Users/akoukosias/Documents/GitHub/Intelligent-Agents/BlackJack/value_function.png', dpi=150)
    plt.show()


def evaluate_policy(policy, num_episodes=100000):
    """Evaluate a policy by playing many episodes and computing win rate."""
    wins = 0
    losses = 0
    ties = 0
    
    for _ in range(num_episodes):
        # Play without exploring starts
        player_hand = draw_hand()
        dealer_hand = draw_hand()
        dealer_showing = dealer_hand[0]
        
        # Player's turn
        while True:
            player_sum = sum_hand(player_hand)
            
            if player_sum < 12:
                player_hand.append(draw_card())
                continue
            
            if is_bust(player_hand):
                losses += 1
                break
            
            state = get_state(player_hand, dealer_showing)
            action = policy.get(state, HIT)
            
            if action == STICK:
                # Dealer's turn
                while dealer_policy(dealer_hand) == HIT:
                    dealer_hand.append(draw_card())
                
                player_sum = sum_hand(player_hand)
                dealer_sum = sum_hand(dealer_hand)
                
                if is_bust(dealer_hand):
                    wins += 1
                elif dealer_sum > player_sum:
                    losses += 1
                elif dealer_sum < player_sum:
                    wins += 1
                else:
                    ties += 1
                break
            else:
                player_hand.append(draw_card())
    
    total = wins + losses + ties
    print(f"\nPolicy Evaluation ({num_episodes} episodes):")
    print(f"  Wins:   {wins} ({100*wins/total:.2f}%)")
    print(f"  Losses: {losses} ({100*losses/total:.2f}%)")
    print(f"  Ties:   {ties} ({100*ties/total:.2f}%)")
    print(f"  Expected Return: {(wins - losses) / total:.4f}")
    
    return wins, losses, ties


def print_policy(policy):
    """Print the policy in a readable format."""
    print("\n" + "="*60)
    print("OPTIMAL POLICY")
    print("="*60)
    
    for usable_ace in [False, True]:
        ace_str = "WITH USABLE ACE" if usable_ace else "WITHOUT USABLE ACE"
        print(f"\n{ace_str}")
        print("-" * 40)
        
        # Header
        print("Player\\Dealer", end="")
        for d in range(1, 11):
            label = "A" if d == 1 else str(d)
            print(f" {label:>2}", end="")
        print()
        
        # Policy grid
        for p in range(21, 11, -1):
            print(f"    {p:>2}       ", end="")
            for d in range(1, 11):
                state = (p, d, usable_ace)
                action = policy.get(state, HIT)
                symbol = "S" if action == STICK else "H"
                print(f" {symbol:>2}", end="")
            print()
    
    print("\nLegend: H = Hit, S = Stick")


def main():
    print("="*60)
    print("BLACKJACK - Monte Carlo Exploring Starts")
    print("="*60)
    
    # Run Monte Carlo ES
    print("\nTraining with Monte Carlo Exploring Starts...")
    print("This may take a minute...\n")
    
    Q, policy = monte_carlo_es(num_episodes=500000)
    
    # Print the optimal policy
    print_policy(policy)
    
    # Evaluate the learned policy
    evaluate_policy(policy, num_episodes=100000)
    
    # Plot results
    print("\nGenerating plots...")
    plot_policy(policy)
    plot_value_function(Q)
    
    print("\nDone! Plots saved to BlackJack folder.")


if __name__ == "__main__":
    main()
