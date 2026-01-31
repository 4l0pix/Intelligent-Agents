import numpy as np
from collections import defaultdict
import random
import matplotlib.pyplot as plt

def plot_policy(policy, usable_ace=True, filename=None):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    fig, ax = plt.subplots()
    # Collect points for each action for better color separation
    hit_x, hit_y = [], []
    stick_x, stick_y = [], []
    other_x, other_y = [], []
    for ps in player_sums:
        for dc in dealer_cards:
            state = (ps, dc, usable_ace)
            action = policy.get(state, None)
            if action == "stick":
                stick_x.append(dc)
                stick_y.append(ps)
            elif action == "hit":
                hit_x.append(dc)
                hit_y.append(ps)
            else:
                other_x.append(dc)
                other_y.append(ps)
    # Plot each action with a distinct color
    ax.scatter(hit_x, hit_y, marker='o', color='red', s=80, label='Hit')
    ax.scatter(stick_x, stick_y, marker='s', color='green', s=80, label='Stick')
    if other_x:
        ax.scatter(other_x, other_y, marker='x', color='gray', s=80, label='Other')
    # Add action text on top of each point
    for x, y in zip(hit_x, hit_y):
        ax.text(x, y, 'H', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    for x, y in zip(stick_x, stick_y):
        ax.text(x, y, 'S', ha='center', va='center', color='white', fontsize=8, fontweight='bold')
    for x, y in zip(other_x, other_y):
        ax.text(x, y, 'X', ha='center', va='center', color='black', fontsize=8, fontweight='bold')
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title(f'Optimal Policy (Usable Ace: {usable_ace})')
    ax.set_xticks(dealer_cards)
    ax.set_yticks(player_sums)
    plt.gca().invert_yaxis()
    ax.legend()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()

def draw_card():
    """Infinite deck, Ace = 1, face cards = 10"""
    card = random.randint(1, 13)
    return min(card, 10)

def draw_hand():
    return [draw_card(), draw_card()]

def usable_ace(hand):
    """Check if the hand has a usable ace (counted as 11 without busting)."""
    return 1 in hand and sum(hand) + 10 <= 21

def hand_value(hand):
    """Calculate the total value of a hand."""
    val = sum(hand)
    if usable_ace(hand):
        val += 10
    return val

def is_bust(hand):
    """Check if the hand value exceeds 21."""
    return hand_value(hand) > 21

def dealer_policy(hand):
    """Dealer hits on less than 17, otherwise sticks."""
    return hand_value(hand) < 17  # True = hit, False = stick


class Blackjack:
    def __init__(self):
        self.player = []
        self.dealer = []
        self.done = False
        self.reward = 0
    
    def reset(self):
        """Start a new game episode."""
        while True:
            self.player = draw_hand()
            self.dealer = draw_hand()
            player_sum = hand_value(self.player)
            if 12 <= player_sum <= 21:  # valid starting state
                break
        self.done = False
        self.reward = 0
        return self.get_state()
    
    def get_state(self):
        """Return the current game state."""
        return (hand_value(self.player),
                self.dealer[0],
                usable_ace(self.player))
    
    def step(self, action):
        """Execute one step in the environment (player's action)."""
        if action == "hit":
            self.player.append(draw_card())
            if is_bust(self.player):
                self.done = True
                self.reward = -1
        else:  # stick
            while dealer_policy(self.dealer):
                self.dealer.append(draw_card())
            player_val = hand_value(self.player)
            dealer_val = hand_value(self.dealer)
            self.done = True
            if is_bust(self.dealer):
                self.reward = 1
            elif player_val > dealer_val:
                self.reward = 1
            elif player_val < dealer_val:
                self.reward = -1
            else:
                self.reward = 0
        return self.get_state(), self.reward, self.done


actions = ["stick", "hit"]
Q = defaultdict(lambda: {a: 0 for a in actions})
returns_sum = defaultdict(lambda: {a: 0 for a in actions})
returns_count = defaultdict(lambda: {a: 0 for a in actions})

# Initial policy: stick only at 20 or 21
def initial_policy(state):
    player_sum, dealer_card, usable = state
    if player_sum >= 20:
        return "stick"
    else:
        return "hit"

# Training using Exploring Starts
def mc_es(num_episodes=500000):
    for _ in range(num_episodes):
        env = Blackjack()
        # Exploring start: random initial state and random action
        state = env.reset()
        action = random.choice(actions)
        episode = [(state, action)]
        
        # Play the rest of the episode following the policy
        done = False
        while not done:
            state, reward, done = env.step(action)
            if done:
                break
            action = initial_policy(state)
            episode.append((state, action))
        
        G = reward  # Terminal reward
        visited = set()
        for state, action in episode:
            if (state, action) not in visited:
                returns_sum[state][action] += G
                returns_count[state][action] += 1
                Q[state][action] = returns_sum[state][action] / returns_count[state][action]
                visited.add((state, action))
    
    # Derive the optimal policy
    policy = {}
    for state, actions_dict in Q.items():
        best_action = max(actions_dict, key=actions_dict.get)
        policy[state] = best_action
    return policy, Q

def plot_state_values(Q_values, usable_ace=True, filename=None):
    player_sums = np.arange(12, 22)
    dealer_cards = np.arange(1, 11)
    Z = np.zeros((len(player_sums), len(dealer_cards)))
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            state = (ps, dc, usable_ace)
            if state in Q_values:
                Z[i, j] = max(Q_values[state].values())
            else:
                Z[i, j] = 0
    fig, ax = plt.subplots()
    for i, ps in enumerate(player_sums):
        for j, dc in enumerate(dealer_cards):
            ax.scatter(dc, ps, c='b', s=60)
            ax.text(dc, ps, f"{Z[i, j]:.2f}", ha='center', va='center', color='red', fontsize=8)
    ax.set_xlabel('Dealer Showing')
    ax.set_ylabel('Player Sum')
    ax.set_title(f'Optimal State Values (Usable Ace: {usable_ace})')
    ax.set_xticks(dealer_cards)
    ax.set_yticks(player_sums)
    plt.gca().invert_yaxis()
    plt.tight_layout()
    if filename:
        plt.savefig(filename)
    plt.show()


policy, Q_values = mc_es(500000)  
# Display the full learned policy
print("=== Optimal Policy (Monte Carlo Exploring Starts) ===")
for k in sorted(policy.keys()):
    print(k, policy[k])

# Plot for usable ace and non-usable ace (state values)
plot_state_values(Q_values, usable_ace=True, filename='state_values_usable_ace.png')
plot_state_values(Q_values, usable_ace=False, filename='state_values_no_usable_ace.png')
# Plot the policy with colored actions
plot_policy(policy, usable_ace=True, filename='policy_usable_ace.png')
plot_policy(policy, usable_ace=False, filename='policy_no_usable_ace.png')