"""
================================================================================
COMBINED BLACKJACK MONTE CARLO EXPLORING STARTS COMPARISON
================================================================================
This script combines two implementations of Monte Carlo ES for Blackjack:
    1. Koukosias Athanasios (akoukosias) - 2025-2026
    2. Tzanidakis V. (vtzanidakis)

CRITICAL: Both algorithms are trained on the EXACT SAME episodes.
A shared episode generator creates identical sequences of:
    - Initial player hands
    - Initial dealer hands  
    - Exploring starts (random initial state + action)
    - Card draws during gameplay

This ensures a fair comparison of the two implementations.

OUTPUT: 4 Interactive Policy Plots (2 per algorithm x 2 ace conditions)
Each cell is clickable to show detailed statistics explaining the policy decision.
================================================================================
"""

import numpy as np
import matplotlib.pyplot as plt
from matplotlib.widgets import Button
from collections import defaultdict
import random

# ================================================================================
# SHARED CONSTANTS AND UTILITIES
# ================================================================================

# Actions
HIT = 0
STICK = 1
ACTIONS = [HIT, STICK]
ACTION_NAMES = {HIT: "hit", STICK: "stick", "hit": HIT, "stick": STICK}

# Card values: A, 2-10, J, Q, K
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]


# ================================================================================
# SHARED EPISODE GENERATOR
# ================================================================================
# This class generates episodes that BOTH algorithms will use for training.
# By using the same random seed and sequence, both agents see identical games.
# ================================================================================

class SharedEpisodeGenerator:
    """
    Generates shared episodes for both algorithms.
    Ensures both agents train on the EXACT SAME sequence of games.
    """
    
    def __init__(self, seed=42):
        """Initialize with a fixed seed for reproducibility."""
        self.seed = seed
        self.rng = random.Random(seed)
        self.episode_count = 0
        
        # Pre-generate all possible states for exploring starts
        self.all_states = []
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                for has_usable_ace in [True, False]:
                    self.all_states.append((player_sum, dealer_showing, has_usable_ace))
    
    def reset(self):
        """Reset the generator to replay the same sequence."""
        self.rng = random.Random(self.seed)
        self.episode_count = 0
    
    def draw_card(self):
        """Draw a card from infinite deck."""
        return self.rng.choice(CARD_VALUES)
    
    def draw_hand(self):
        """Draw initial two cards."""
        return [self.draw_card(), self.draw_card()]
    
    def get_exploring_start(self):
        """Get random initial state and action for exploring starts."""
        initial_state = self.rng.choice(self.all_states)
        initial_action = self.rng.choice(ACTIONS)
        return initial_state, initial_action
    
    def generate_episode_data(self):
        """
        Generate all random data needed for one episode.
        Returns a dictionary with all pre-determined random choices.
        Both algorithms will use this EXACT data.
        """
        self.episode_count += 1
        
        # Pre-generate enough cards for any episode (max ~20 cards should be plenty)
        cards = [self.draw_card() for _ in range(30)]
        
        # Get exploring start
        initial_state, initial_action = self.get_exploring_start()
        
        return {
            'episode_num': self.episode_count,
            'cards': cards,
            'card_index': 0,
            'initial_state': initial_state,
            'initial_action': initial_action
        }


# ================================================================================
# SHARED HELPER FUNCTIONS
# ================================================================================

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
        - player_sum: 12-21
        - dealer_showing: 1-10 (ace=1)
        - usable_ace: True/False
    """
    return (sum_hand(player_hand), dealer_showing, usable_ace(player_hand))


def dealer_policy(dealer_hand):
    """Dealer's fixed strategy: hit if sum < 17, otherwise stick."""
    return sum_hand(dealer_hand) < 17


def create_hand_from_state(state, episode_data):
    """Create a hand that matches the given state using episode's random data."""
    player_sum, dealer_showing, has_usable_ace = state
    
    if has_usable_ace:
        # Hand with usable ace: ace + (sum - 11)
        player_hand = [1, player_sum - 11]
    else:
        # Hand without usable ace
        if player_sum <= 11:
            player_hand = [player_sum]
        else:
            player_hand = [10, player_sum - 10]
    
    return player_hand


def get_next_card(episode_data):
    """Get the next card from the pre-generated sequence."""
    card = episode_data['cards'][episode_data['card_index']]
    episode_data['card_index'] += 1
    return card


# ================================================================================
# STATISTICS TRACKER
# ================================================================================
# Tracks detailed statistics for each state-action pair to explain policy decisions
# ================================================================================

class StatisticsTracker:
    """
    Tracks detailed statistics for each state-action pair.
    Used to explain why a particular policy was chosen.
    """
    
    def __init__(self, name):
        self.name = name
        # For each (state, action): track outcomes
        self.stats = defaultdict(lambda: {
            'total_episodes': 0,
            'player_busted': 0,
            'dealer_busted': 0,
            'player_won': 0,
            'dealer_won': 0,
            'draw': 0,
            'total_reward': 0,
            'final_player_sums': [],
            'final_dealer_sums': []
        })
    
    def record_outcome(self, state, action, outcome_info):
        """
        Record the outcome of an episode for a state-action pair.
        
        outcome_info contains:
            - reward: -1, 0, or 1
            - player_busted: bool
            - dealer_busted: bool
            - final_player_sum: int
            - final_dealer_sum: int (or None if player busted)
        """
        key = (state, action)
        stats = self.stats[key]
        
        stats['total_episodes'] += 1
        stats['total_reward'] += outcome_info['reward']
        
        if outcome_info['player_busted']:
            stats['player_busted'] += 1
        elif outcome_info['dealer_busted']:
            stats['dealer_busted'] += 1
            stats['player_won'] += 1
        elif outcome_info['reward'] == 1:
            stats['player_won'] += 1
        elif outcome_info['reward'] == -1:
            stats['dealer_won'] += 1
        else:
            stats['draw'] += 1
        
        if outcome_info['final_player_sum'] is not None:
            stats['final_player_sums'].append(outcome_info['final_player_sum'])
        if outcome_info['final_dealer_sum'] is not None:
            stats['final_dealer_sums'].append(outcome_info['final_dealer_sum'])
    
    def get_stats(self, state, action):
        """Get statistics for a specific state-action pair."""
        return self.stats[(state, action)]


# ================================================================================
# ALGORITHM 1: KOUKOSIAS ATHANASIOS IMPLEMENTATION
# ================================================================================
# Original implementation by Koukosias Athanasios (2025-2026)
# Modified to use shared episode data for fair comparison
# ================================================================================

class KoukosiasMCES:
    """
    Monte Carlo Exploring Starts - Koukosias Implementation
    
    Key characteristics:
    - Uses first-visit MC updates
    - Processes episodes in reverse order for return calculation
    - Greedy policy improvement after each episode
    """
    
    def __init__(self):
        self.Q = defaultdict(float)
        self.returns_sum = defaultdict(float)
        self.returns_count = defaultdict(int)
        self.policy = {}
        self.stats_tracker = StatisticsTracker("Koukosias")
        
        # Initialize all states
        self.all_states = []
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                for has_usable_ace in [True, False]:
                    self.all_states.append((player_sum, dealer_showing, has_usable_ace))
    
    def player_policy(self, state):
        """Greedy policy based on Q-values."""
        if self.Q[(state, HIT)] >= self.Q[(state, STICK)]:
            return HIT
        else:
            return STICK
    
    def play_episode(self, episode_data):
        """
        Play one episode using shared episode data.
        Returns episode history and outcome information.
        """
        initial_state = episode_data['initial_state']
        initial_action = episode_data['initial_action']
        
        # Create hands matching the initial state
        player_hand = create_hand_from_state(initial_state, episode_data)
        dealer_showing = initial_state[1]
        dealer_hand = [dealer_showing, get_next_card(episode_data)]
        
        # Episode history: list of (state, action)
        episode = []
        
        # Outcome tracking
        outcome_info = {
            'reward': 0,
            'player_busted': False,
            'dealer_busted': False,
            'final_player_sum': None,
            'final_dealer_sum': None
        }
        
        # Player's turn
        first_action = True
        while True:
            player_sum = sum_hand(player_hand)
            
            # If player sum < 12, always hit (no decision needed)
            if player_sum < 12:
                player_hand.append(get_next_card(episode_data))
                continue
            
            state = get_state(player_hand, dealer_showing)
            
            # Choose action
            if first_action:
                action = initial_action
                first_action = False
            else:
                action = self.player_policy(state)
            
            episode.append((state, action))
            
            if action == STICK:
                break
            else:  # HIT
                player_hand.append(get_next_card(episode_data))
                if is_bust(player_hand):
                    outcome_info['player_busted'] = True
                    outcome_info['reward'] = -1
                    outcome_info['final_player_sum'] = sum_hand(player_hand)
                    return episode, outcome_info
        
        # Dealer's turn (only if player didn't bust)
        while dealer_policy(dealer_hand):
            dealer_hand.append(get_next_card(episode_data))
        
        # Determine winner
        player_sum = sum_hand(player_hand)
        dealer_sum = sum_hand(dealer_hand)
        
        outcome_info['final_player_sum'] = player_sum
        outcome_info['final_dealer_sum'] = dealer_sum
        
        if is_bust(dealer_hand):
            outcome_info['dealer_busted'] = True
            outcome_info['reward'] = 1
        elif dealer_sum > player_sum:
            outcome_info['reward'] = -1
        elif dealer_sum < player_sum:
            outcome_info['reward'] = 1
        else:
            outcome_info['reward'] = 0
        
        return episode, outcome_info
    
    def update(self, episode, outcome_info):
        """Update Q-values using first-visit MC (Koukosias style)."""
        reward = outcome_info['reward']
        
        # Track statistics for the first state-action pair
        if episode:
            first_state, first_action = episode[0]
            self.stats_tracker.record_outcome(first_state, first_action, outcome_info)
        
        # First-visit MC: process in reverse order
        visited = set()
        G = 0  # Return (gamma = 1, no discounting)
        
        for t in range(len(episode) - 1, -1, -1):
            state, action = episode[t]
            # Only the last state gets the actual reward
            if t == len(episode) - 1:
                G = reward
            else:
                G = G + 0  # gamma = 1, intermediate rewards = 0
            
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                self.returns_sum[state_action] += G
                self.returns_count[state_action] += 1
                self.Q[state_action] = self.returns_sum[state_action] / self.returns_count[state_action]
    
    def extract_policy(self):
        """Extract optimal policy from Q-values."""
        for state in self.all_states:
            if self.Q[(state, HIT)] >= self.Q[(state, STICK)]:
                self.policy[state] = HIT
            else:
                self.policy[state] = STICK
        return self.policy


# ================================================================================
# ALGORITHM 2: TZANIDAKIS IMPLEMENTATION
# ================================================================================
# Original implementation by Tzanidakis V.
# Modified to use shared episode data for fair comparison
# ================================================================================

class TzanidakisMCES:
    """
    Monte Carlo Exploring Starts - Tzanidakis Implementation
    
    Key characteristics:
    - Uses dictionary-based Q-values with action keys
    - Processes all state-actions in episode (not reverse order)
    - Different data structure organization
    """
    
    def __init__(self):
        self.Q = defaultdict(lambda: {HIT: 0.0, STICK: 0.0})
        self.returns_sum = defaultdict(lambda: {HIT: 0.0, STICK: 0.0})
        self.returns_count = defaultdict(lambda: {HIT: 0, STICK: 0})
        self.policy = {}
        self.stats_tracker = StatisticsTracker("Tzanidakis")
    
    def initial_policy(self, state):
        """Initial policy: stick only at 20 or 21."""
        player_sum, dealer_card, usable = state
        if player_sum >= 20:
            return STICK
        else:
            return HIT
    
    def play_episode(self, episode_data):
        """
        Play one episode using shared episode data.
        Returns episode history and outcome information.
        """
        initial_state = episode_data['initial_state']
        initial_action = episode_data['initial_action']
        
        # Create hands matching the initial state
        player_hand = create_hand_from_state(initial_state, episode_data)
        dealer_showing = initial_state[1]
        dealer_hand = [dealer_showing, get_next_card(episode_data)]
        
        # Episode history
        episode = [(initial_state, initial_action)]
        
        # Outcome tracking
        outcome_info = {
            'reward': 0,
            'player_busted': False,
            'dealer_busted': False,
            'final_player_sum': None,
            'final_dealer_sum': None
        }
        
        # Execute first action
        action = initial_action
        
        while True:
            if action == STICK:
                # Dealer's turn
                while dealer_policy(dealer_hand):
                    dealer_hand.append(get_next_card(episode_data))
                
                player_sum = sum_hand(player_hand)
                dealer_sum = sum_hand(dealer_hand)
                
                outcome_info['final_player_sum'] = player_sum
                outcome_info['final_dealer_sum'] = dealer_sum
                
                if is_bust(dealer_hand):
                    outcome_info['dealer_busted'] = True
                    outcome_info['reward'] = 1
                elif player_sum > dealer_sum:
                    outcome_info['reward'] = 1
                elif player_sum < dealer_sum:
                    outcome_info['reward'] = -1
                else:
                    outcome_info['reward'] = 0
                break
            
            else:  # HIT
                player_hand.append(get_next_card(episode_data))
                
                if is_bust(player_hand):
                    outcome_info['player_busted'] = True
                    outcome_info['reward'] = -1
                    outcome_info['final_player_sum'] = sum_hand(player_hand)
                    break
                
                # Get new state and action
                state = get_state(player_hand, dealer_showing)
                action = self.initial_policy(state)
                episode.append((state, action))
        
        return episode, outcome_info
    
    def update(self, episode, outcome_info):
        """Update Q-values (Tzanidakis style)."""
        reward = outcome_info['reward']
        
        # Track statistics for the first state-action pair
        if episode:
            first_state, first_action = episode[0]
            self.stats_tracker.record_outcome(first_state, first_action, outcome_info)
        
        # First-visit MC update
        visited = set()
        G = reward  # Terminal reward
        
        for state, action in episode:
            if (state, action) not in visited:
                self.returns_sum[state][action] += G
                self.returns_count[state][action] += 1
                self.Q[state][action] = self.returns_sum[state][action] / self.returns_count[state][action]
                visited.add((state, action))
    
    def extract_policy(self):
        """Extract optimal policy from Q-values."""
        for state, actions_dict in self.Q.items():
            if actions_dict[HIT] >= actions_dict[STICK]:
                self.policy[state] = HIT
            else:
                self.policy[state] = STICK
        return self.policy


# ================================================================================
# TRAINING FUNCTION
# ================================================================================
# Trains BOTH algorithms on the EXACT SAME episodes
# ================================================================================

def train_both_algorithms(num_episodes=500000, seed=42):
    """
    Train both algorithms on the EXACT SAME episodes.
    
    This ensures a fair comparison by:
    1. Using a shared random seed
    2. Generating identical episode data for both algorithms
    3. Both agents see the same cards, same exploring starts
    
    Returns both trained algorithm instances.
    """
    print("=" * 70)
    print("TRAINING BOTH ALGORITHMS ON IDENTICAL EPISODES")
    print("=" * 70)
    print(f"Number of episodes: {num_episodes:,}")
    print(f"Random seed: {seed}")
    print("-" * 70)
    
    # Initialize algorithms
    koukosias = KoukosiasMCES()
    tzanidakis = TzanidakisMCES()
    
    # Initialize shared episode generator
    generator = SharedEpisodeGenerator(seed=seed)
    
    for episode_num in range(num_episodes):
        if (episode_num + 1) % 100000 == 0:
            print(f"Episode {episode_num + 1:,}/{num_episodes:,}")
        
        # Generate shared episode data
        episode_data = generator.generate_episode_data()
        
        # ============================================================
        # ALGORITHM 1: Koukosias plays this episode
        # ============================================================
        episode_data_copy1 = {
            'episode_num': episode_data['episode_num'],
            'cards': episode_data['cards'].copy(),
            'card_index': 0,
            'initial_state': episode_data['initial_state'],
            'initial_action': episode_data['initial_action']
        }
        episode1, outcome1 = koukosias.play_episode(episode_data_copy1)
        koukosias.update(episode1, outcome1)
        
        # ============================================================
        # ALGORITHM 2: Tzanidakis plays this episode
        # ============================================================
        episode_data_copy2 = {
            'episode_num': episode_data['episode_num'],
            'cards': episode_data['cards'].copy(),
            'card_index': 0,
            'initial_state': episode_data['initial_state'],
            'initial_action': episode_data['initial_action']
        }
        episode2, outcome2 = tzanidakis.play_episode(episode_data_copy2)
        tzanidakis.update(episode2, outcome2)
    
    # Extract final policies
    koukosias.extract_policy()
    tzanidakis.extract_policy()
    
    print("-" * 70)
    print("Training complete!")
    print("=" * 70)
    
    return koukosias, tzanidakis


# ================================================================================
# COMPREHENSIVE STATISTICAL VISUALIZATION
# ================================================================================
# Creates multi-plot figures showing all statistics as heatmaps
# Plus interactive click functionality for detailed analysis
# ================================================================================

def get_stats_grids(algorithm, usable_ace):
    """
    Extract all statistics grids for visualization.
    Returns dictionary of 10x10 numpy arrays for each statistic.
    """
    grids = {
        'policy': np.zeros((10, 10)),
        'q_hit': np.zeros((10, 10)),
        'q_stick': np.zeros((10, 10)),
        'q_diff': np.zeros((10, 10)),
        'player_bust_pct': np.zeros((10, 10)),
        'dealer_bust_pct': np.zeros((10, 10)),
        'player_win_pct': np.zeros((10, 10)),
        'dealer_win_pct': np.zeros((10, 10)),
        'draw_pct': np.zeros((10, 10)),
        'avg_reward': np.zeros((10, 10)),
        'total_episodes': np.zeros((10, 10)),
    }
    
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            i = player_sum - 12
            j = dealer_showing - 1
            state = (player_sum, dealer_showing, usable_ace)
            
            # Policy
            action = algorithm.policy.get(state, HIT)
            grids['policy'][i, j] = action
            
            # Q-values
            if isinstance(algorithm.Q[state], dict):
                q_hit = algorithm.Q[state][HIT]
                q_stick = algorithm.Q[state][STICK]
            else:
                q_hit = algorithm.Q[(state, HIT)]
                q_stick = algorithm.Q[(state, STICK)]
            
            grids['q_hit'][i, j] = q_hit
            grids['q_stick'][i, j] = q_stick
            grids['q_diff'][i, j] = q_stick - q_hit  # Positive = STICK better
            
            # Statistics for the chosen action
            stats = algorithm.stats_tracker.get_stats(state, action)
            total = stats['total_episodes']
            
            grids['total_episodes'][i, j] = total
            
            if total > 0:
                grids['player_bust_pct'][i, j] = 100 * stats['player_busted'] / total
                grids['dealer_bust_pct'][i, j] = 100 * stats['dealer_busted'] / total
                grids['player_win_pct'][i, j] = 100 * stats['player_won'] / total
                grids['dealer_win_pct'][i, j] = 100 * stats['dealer_won'] / total
                grids['draw_pct'][i, j] = 100 * stats['draw'] / total
                grids['avg_reward'][i, j] = stats['total_reward'] / total
    
    return grids


def create_heatmap(ax, data, title, cmap='viridis', vmin=None, vmax=None, 
                   show_policy_labels=False, policy_grid=None, fmt='.1f', annot=True):
    """Create a single heatmap subplot."""
    if vmin is None:
        vmin = np.min(data)
    if vmax is None:
        vmax = np.max(data)
    
    im = ax.imshow(data, cmap=cmap, aspect='auto', origin='lower', vmin=vmin, vmax=vmax)
    
    ax.set_xticks(range(10))
    ax.set_xticklabels(['A'] + list(range(2, 11)), fontsize=8)
    ax.set_yticks(range(10))
    ax.set_yticklabels(range(12, 22), fontsize=8)
    ax.set_xlabel("Dealer Showing", fontsize=9)
    ax.set_ylabel("Player Sum", fontsize=9)
    ax.set_title(title, fontsize=10, fontweight='bold')
    
    # Add annotations
    if annot:
        for i in range(10):
            for j in range(10):
                if show_policy_labels and policy_grid is not None:
                    text = "S" if policy_grid[i, j] == STICK else "H"
                    color = 'white' if policy_grid[i, j] == STICK else 'black'
                else:
                    val = data[i, j]
                    if fmt == 'd':
                        text = f"{int(val)}"
                    else:
                        text = f"{val:{fmt}}"
                    # Choose text color based on background brightness
                    norm_val = (val - vmin) / (vmax - vmin + 1e-10)
                    color = 'white' if norm_val < 0.5 else 'black'
                ax.text(j, i, text, ha='center', va='center', 
                       color=color, fontsize=6, fontweight='bold')
    
    return im


def create_algorithm_stats_multiplot(algorithm, usable_ace, save_prefix=None):
    """
    Create a comprehensive multiplot figure showing all statistics for one algorithm.
    
    Layout: 3 rows x 4 columns = 12 subplots
    Row 1: Policy, Q(HIT), Q(STICK), Q-Difference
    Row 2: Player Bust%, Dealer Bust%, Player Win%, Dealer Win%
    Row 3: Draw%, Average Reward, Total Episodes, Summary
    """
    ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
    
    grids = get_stats_grids(algorithm, usable_ace)
    
    fig, axes = plt.subplots(3, 4, figsize=(20, 15))
    fig.suptitle(f"{algorithm.stats_tracker.name} Algorithm - Complete Statistics Analysis\n{ace_str}",
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Row 1: Policy and Q-values
    # Policy
    im1 = create_heatmap(axes[0, 0], grids['policy'], "Optimal Policy",
                         cmap='RdYlGn', vmin=0, vmax=1,
                         show_policy_labels=True, policy_grid=grids['policy'])
    
    # Q(HIT)
    im2 = create_heatmap(axes[0, 1], grids['q_hit'], "Q(state, HIT)",
                         cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.colorbar(im2, ax=axes[0, 1], shrink=0.8)
    
    # Q(STICK)
    im3 = create_heatmap(axes[0, 2], grids['q_stick'], "Q(state, STICK)",
                         cmap='RdBu_r', vmin=-1, vmax=1, fmt='.2f')
    plt.colorbar(im3, ax=axes[0, 2], shrink=0.8)
    
    # Q-Difference (STICK - HIT)
    im4 = create_heatmap(axes[0, 3], grids['q_diff'], "Q(STICK) - Q(HIT)\n(Positive = STICK better)",
                         cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im4, ax=axes[0, 3], shrink=0.8)
    
    # Row 2: Outcome percentages
    # Player Bust %
    im5 = create_heatmap(axes[1, 0], grids['player_bust_pct'], "Player Bust %",
                         cmap='Reds', vmin=0, vmax=100, fmt='.0f')
    plt.colorbar(im5, ax=axes[1, 0], shrink=0.8)
    
    # Dealer Bust %
    im6 = create_heatmap(axes[1, 1], grids['dealer_bust_pct'], "Dealer Bust %",
                         cmap='Blues', vmin=0, vmax=50, fmt='.0f')
    plt.colorbar(im6, ax=axes[1, 1], shrink=0.8)
    
    # Player Win %
    im7 = create_heatmap(axes[1, 2], grids['player_win_pct'], "Player Win %",
                         cmap='Greens', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im7, ax=axes[1, 2], shrink=0.8)
    
    # Dealer Win %
    im8 = create_heatmap(axes[1, 3], grids['dealer_win_pct'], "Dealer Win %",
                         cmap='Oranges', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im8, ax=axes[1, 3], shrink=0.8)
    
    # Row 3: Draw, Average Reward, Episodes, Summary
    # Draw %
    im9 = create_heatmap(axes[2, 0], grids['draw_pct'], "Draw %",
                         cmap='Purples', vmin=0, vmax=20, fmt='.0f')
    plt.colorbar(im9, ax=axes[2, 0], shrink=0.8)
    
    # Average Reward
    im10 = create_heatmap(axes[2, 1], grids['avg_reward'], "Average Reward",
                          cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im10, ax=axes[2, 1], shrink=0.8)
    
    # Total Episodes (log scale for better visualization)
    episodes_log = np.log10(grids['total_episodes'] + 1)
    im11 = create_heatmap(axes[2, 2], grids['total_episodes'], "Total Episodes",
                          cmap='YlOrRd', fmt='d', annot=False)
    plt.colorbar(im11, ax=axes[2, 2], shrink=0.8)
    # Add episode counts as text
    for i in range(10):
        for j in range(10):
            val = int(grids['total_episodes'][i, j])
            text = f"{val//1000}k" if val >= 1000 else str(val)
            ax = axes[2, 2]
            ax.text(j, i, text, ha='center', va='center', 
                   color='black', fontsize=5, fontweight='bold')
    
    # Summary statistics text box
    axes[2, 3].axis('off')
    
    # Calculate summary statistics
    total_states = 100
    stick_states = np.sum(grids['policy'] == STICK)
    hit_states = total_states - stick_states
    avg_player_win = np.mean(grids['player_win_pct'])
    avg_dealer_win = np.mean(grids['dealer_win_pct'])
    avg_draw = np.mean(grids['draw_pct'])
    avg_player_bust = np.mean(grids['player_bust_pct'])
    avg_dealer_bust = np.mean(grids['dealer_bust_pct'])
    overall_avg_reward = np.mean(grids['avg_reward'])
    
    summary_text = f"""
    SUMMARY STATISTICS
    ══════════════════════════════
    
    POLICY DISTRIBUTION:
      • STICK actions: {stick_states} states ({stick_states}%)
      • HIT actions: {hit_states} states ({hit_states}%)
    
    AVERAGE OUTCOMES:
      • Player Win Rate: {avg_player_win:.1f}%
      • Dealer Win Rate: {avg_dealer_win:.1f}%
      • Draw Rate: {avg_draw:.1f}%
    
    BUST RATES:
      • Player Bust: {avg_player_bust:.1f}%
      • Dealer Bust: {avg_dealer_bust:.1f}%
    
    EXPECTED VALUE:
      • Average Reward: {overall_avg_reward:+.4f}
    
    ══════════════════════════════
    Click on any plot cell for
    detailed state-action analysis
    """
    
    axes[2, 3].text(0.05, 0.95, summary_text, transform=axes[2, 3].transAxes,
                   fontsize=10, verticalalignment='top', fontfamily='monospace',
                   bbox=dict(boxstyle='round', facecolor='wheat', alpha=0.5))
    
    plt.tight_layout(rect=[0, 0.03, 1, 0.95])
    
    if save_prefix:
        algo_name = algorithm.stats_tracker.name.lower()
        filename = f"{save_prefix}_{algo_name}_{ace_str.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig, axes


def create_comparison_multiplot(koukosias, tzanidakis, usable_ace, save_prefix=None):
    """
    Create a side-by-side comparison of both algorithms.
    
    Layout: 4 rows x 4 columns
    Columns 1-2: Koukosias
    Columns 3-4: Tzanidakis
    """
    ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
    
    k_grids = get_stats_grids(koukosias, usable_ace)
    t_grids = get_stats_grids(tzanidakis, usable_ace)
    
    fig, axes = plt.subplots(4, 4, figsize=(22, 18))
    fig.suptitle(f"Algorithm Comparison: Koukosias vs Tzanidakis\n{ace_str}",
                 fontsize=16, fontweight='bold', y=0.98)
    
    # Add algorithm labels
    fig.text(0.25, 0.94, "KOUKOSIAS", fontsize=14, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightblue', alpha=0.8))
    fig.text(0.75, 0.94, "TZANIDAKIS", fontsize=14, fontweight='bold', ha='center',
             bbox=dict(boxstyle='round', facecolor='lightgreen', alpha=0.8))
    
    # Row 1: Policies and Q-differences
    create_heatmap(axes[0, 0], k_grids['policy'], "Policy",
                   cmap='RdYlGn', vmin=0, vmax=1,
                   show_policy_labels=True, policy_grid=k_grids['policy'])
    
    im1 = create_heatmap(axes[0, 1], k_grids['q_diff'], "Q(STICK)-Q(HIT)",
                         cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im1, ax=axes[0, 1], shrink=0.7)
    
    create_heatmap(axes[0, 2], t_grids['policy'], "Policy",
                   cmap='RdYlGn', vmin=0, vmax=1,
                   show_policy_labels=True, policy_grid=t_grids['policy'])
    
    im2 = create_heatmap(axes[0, 3], t_grids['q_diff'], "Q(STICK)-Q(HIT)",
                         cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im2, ax=axes[0, 3], shrink=0.7)
    
    # Row 2: Win rates
    im3 = create_heatmap(axes[1, 0], k_grids['player_win_pct'], "Player Win %",
                         cmap='Greens', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im3, ax=axes[1, 0], shrink=0.7)
    
    im4 = create_heatmap(axes[1, 1], k_grids['dealer_win_pct'], "Dealer Win %",
                         cmap='Oranges', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im4, ax=axes[1, 1], shrink=0.7)
    
    im5 = create_heatmap(axes[1, 2], t_grids['player_win_pct'], "Player Win %",
                         cmap='Greens', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im5, ax=axes[1, 2], shrink=0.7)
    
    im6 = create_heatmap(axes[1, 3], t_grids['dealer_win_pct'], "Dealer Win %",
                         cmap='Oranges', vmin=0, vmax=60, fmt='.0f')
    plt.colorbar(im6, ax=axes[1, 3], shrink=0.7)
    
    # Row 3: Bust rates
    im7 = create_heatmap(axes[2, 0], k_grids['player_bust_pct'], "Player Bust %",
                         cmap='Reds', vmin=0, vmax=100, fmt='.0f')
    plt.colorbar(im7, ax=axes[2, 0], shrink=0.7)
    
    im8 = create_heatmap(axes[2, 1], k_grids['dealer_bust_pct'], "Dealer Bust %",
                         cmap='Blues', vmin=0, vmax=50, fmt='.0f')
    plt.colorbar(im8, ax=axes[2, 1], shrink=0.7)
    
    im9 = create_heatmap(axes[2, 2], t_grids['player_bust_pct'], "Player Bust %",
                         cmap='Reds', vmin=0, vmax=100, fmt='.0f')
    plt.colorbar(im9, ax=axes[2, 2], shrink=0.7)
    
    im10 = create_heatmap(axes[2, 3], t_grids['dealer_bust_pct'], "Dealer Bust %",
                          cmap='Blues', vmin=0, vmax=50, fmt='.0f')
    plt.colorbar(im10, ax=axes[2, 3], shrink=0.7)
    
    # Row 4: Average reward and draws
    im11 = create_heatmap(axes[3, 0], k_grids['avg_reward'], "Avg Reward",
                          cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im11, ax=axes[3, 0], shrink=0.7)
    
    im12 = create_heatmap(axes[3, 1], k_grids['draw_pct'], "Draw %",
                          cmap='Purples', vmin=0, vmax=20, fmt='.0f')
    plt.colorbar(im12, ax=axes[3, 1], shrink=0.7)
    
    im13 = create_heatmap(axes[3, 2], t_grids['avg_reward'], "Avg Reward",
                          cmap='RdYlGn', vmin=-0.5, vmax=0.5, fmt='.2f')
    plt.colorbar(im13, ax=axes[3, 2], shrink=0.7)
    
    im14 = create_heatmap(axes[3, 3], t_grids['draw_pct'], "Draw %",
                          cmap='Purples', vmin=0, vmax=20, fmt='.0f')
    plt.colorbar(im14, ax=axes[3, 3], shrink=0.7)
    
    plt.tight_layout(rect=[0, 0.02, 1, 0.92])
    
    if save_prefix:
        filename = f"{save_prefix}_comparison_{ace_str.replace(' ', '_').lower()}.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig, axes


def create_policy_difference_plot(koukosias, tzanidakis, save_prefix=None):
    """
    Create a plot highlighting differences between the two policies.
    """
    fig, axes = plt.subplots(2, 3, figsize=(18, 12))
    fig.suptitle("Policy Comparison: Differences Between Algorithms",
                 fontsize=16, fontweight='bold')
    
    for idx, usable_ace in enumerate([False, True]):
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        row = idx
        
        k_grids = get_stats_grids(koukosias, usable_ace)
        t_grids = get_stats_grids(tzanidakis, usable_ace)
        
        # Column 0: Koukosias policy
        create_heatmap(axes[row, 0], k_grids['policy'], f"Koukosias - {ace_str}",
                       cmap='RdYlGn', vmin=0, vmax=1,
                       show_policy_labels=True, policy_grid=k_grids['policy'])
        
        # Column 1: Tzanidakis policy
        create_heatmap(axes[row, 1], t_grids['policy'], f"Tzanidakis - {ace_str}",
                       cmap='RdYlGn', vmin=0, vmax=1,
                       show_policy_labels=True, policy_grid=t_grids['policy'])
        
        # Column 2: Difference (highlight where they disagree)
        diff = np.abs(k_grids['policy'] - t_grids['policy'])
        axes[row, 2].imshow(diff, cmap='Reds', aspect='auto', origin='lower', vmin=0, vmax=1)
        
        axes[row, 2].set_xticks(range(10))
        axes[row, 2].set_xticklabels(['A'] + list(range(2, 11)), fontsize=8)
        axes[row, 2].set_yticks(range(10))
        axes[row, 2].set_yticklabels(range(12, 22), fontsize=8)
        axes[row, 2].set_xlabel("Dealer Showing", fontsize=9)
        axes[row, 2].set_ylabel("Player Sum", fontsize=9)
        
        diff_count = np.sum(diff > 0)
        axes[row, 2].set_title(f"Differences - {ace_str}\n({diff_count} disagreements)", 
                               fontsize=10, fontweight='bold')
        
        # Annotate with both actions where different
        for i in range(10):
            for j in range(10):
                if diff[i, j] > 0:
                    k_act = "S" if k_grids['policy'][i, j] == STICK else "H"
                    t_act = "S" if t_grids['policy'][i, j] == STICK else "H"
                    axes[row, 2].text(j, i, f"K:{k_act}\nT:{t_act}", ha='center', va='center',
                                     color='white', fontsize=6, fontweight='bold')
                else:
                    act = "S" if k_grids['policy'][i, j] == STICK else "H"
                    axes[row, 2].text(j, i, act, ha='center', va='center',
                                     color='green', fontsize=7, fontweight='bold')
    
    plt.tight_layout()
    
    if save_prefix:
        filename = f"{save_prefix}_policy_differences.png"
        plt.savefig(filename, dpi=150, bbox_inches='tight')
        print(f"Saved: {filename}")
    
    return fig, axes


class InteractivePolicyPlot:
    """
    Creates an interactive policy plot where each cell can be clicked
    to show detailed statistics in a separate popup figure.
    """
    
    def __init__(self, algorithm, usable_ace, title):
        self.algorithm = algorithm
        self.usable_ace = usable_ace
        self.title = title
        self.fig = None
        self.ax = None
    
    def get_stats_data(self, player_sum, dealer_showing):
        """Get statistics data for a state."""
        state = (player_sum, dealer_showing, self.usable_ace)
        action = self.algorithm.policy.get(state, HIT)
        
        # Get stats for both actions
        stats_hit = self.algorithm.stats_tracker.get_stats(state, HIT)
        stats_stick = self.algorithm.stats_tracker.get_stats(state, STICK)
        
        # Q-values
        if isinstance(self.algorithm.Q[state], dict):
            q_hit = self.algorithm.Q[state][HIT]
            q_stick = self.algorithm.Q[state][STICK]
        else:
            q_hit = self.algorithm.Q[(state, HIT)]
            q_stick = self.algorithm.Q[(state, STICK)]
        
        return {
            'state': state,
            'action': action,
            'q_hit': q_hit,
            'q_stick': q_stick,
            'stats_hit': stats_hit,
            'stats_stick': stats_stick
        }
    
    def show_detailed_stats_figure(self, player_sum, dealer_showing):
        """Show a detailed statistics figure for the clicked cell."""
        data = self.get_stats_data(player_sum, dealer_showing)
        state = data['state']
        action = data['action']
        action_name = "STICK" if action == STICK else "HIT"
        
        # Create a new figure with multiple subplots
        fig, axes = plt.subplots(2, 3, figsize=(15, 10))
        ace_str = "Usable Ace" if self.usable_ace else "No Usable Ace"
        fig.suptitle(f"State Analysis: Player {player_sum} vs Dealer {dealer_showing} ({ace_str})\n"
                    f"Algorithm: {self.algorithm.stats_tracker.name} | Optimal Action: {action_name}",
                    fontsize=14, fontweight='bold')
        
        # Get stats for chosen action
        stats = data['stats_hit'] if action == HIT else data['stats_stick']
        total = stats['total_episodes']
        
        if total == 0:
            axes[0, 0].text(0.5, 0.5, "No data available", ha='center', va='center', fontsize=14)
            plt.show()
            return
        
        # Subplot 1: Q-values comparison bar chart
        ax1 = axes[0, 0]
        q_values = [data['q_hit'], data['q_stick']]
        colors = ['red' if action == HIT else 'lightcoral', 
                  'green' if action == STICK else 'lightgreen']
        bars = ax1.bar(['HIT', 'STICK'], q_values, color=colors, edgecolor='black', linewidth=2)
        ax1.axhline(y=0, color='black', linestyle='-', linewidth=0.5)
        ax1.set_ylabel('Q-Value')
        ax1.set_title('Q-Values Comparison')
        for bar, val in zip(bars, q_values):
            ax1.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.02,
                    f'{val:.4f}', ha='center', fontsize=10, fontweight='bold')
        
        # Subplot 2: Outcome distribution pie chart
        ax2 = axes[0, 1]
        outcomes = [stats['player_won'], stats['dealer_won'], stats['draw']]
        labels = [f"Player Won\n{stats['player_won']} ({100*stats['player_won']/total:.1f}%)",
                  f"Dealer Won\n{stats['dealer_won']} ({100*stats['dealer_won']/total:.1f}%)",
                  f"Draw\n{stats['draw']} ({100*stats['draw']/total:.1f}%)"]
        colors_pie = ['#2ecc71', '#e74c3c', '#3498db']
        if sum(outcomes) > 0:
            ax2.pie(outcomes, labels=labels, colors=colors_pie, autopct='',
                   startangle=90, explode=(0.05, 0.05, 0.05))
        ax2.set_title(f'Outcome Distribution\n(Total: {total:,} episodes)')
        
        # Subplot 3: Bust rates bar chart
        ax3 = axes[0, 2]
        bust_data = [stats['player_busted'], stats['dealer_busted']]
        bust_pct = [100 * stats['player_busted'] / total, 
                    100 * stats['dealer_busted'] / total]
        bars3 = ax3.bar(['Player Bust', 'Dealer Bust'], bust_pct, 
                       color=['#c0392b', '#2980b9'], edgecolor='black')
        ax3.set_ylabel('Bust Rate (%)')
        ax3.set_title('Bust Rates')
        ax3.set_ylim(0, 100)
        for bar, val, count in zip(bars3, bust_pct, bust_data):
            ax3.text(bar.get_x() + bar.get_width()/2, bar.get_height() + 2,
                    f'{val:.1f}%\n({count:,})', ha='center', fontsize=9)
        
        # Subplot 4: Final sum distributions (if available)
        ax4 = axes[1, 0]
        if stats['final_player_sums']:
            ax4.hist(stats['final_player_sums'], bins=range(12, 32), 
                    color='#3498db', edgecolor='black', alpha=0.7, label='Player')
            ax4.axvline(x=21, color='green', linestyle='--', linewidth=2, label='21')
            ax4.axvline(x=np.mean(stats['final_player_sums']), color='red', 
                       linestyle=':', linewidth=2, label=f'Mean: {np.mean(stats["final_player_sums"]):.1f}')
            ax4.set_xlabel('Final Sum')
            ax4.set_ylabel('Frequency')
            ax4.set_title('Player Final Sum Distribution')
            ax4.legend(fontsize=8)
        else:
            ax4.text(0.5, 0.5, "No player sum data", ha='center', va='center')
        
        # Subplot 5: Dealer final sum distribution
        ax5 = axes[1, 1]
        if stats['final_dealer_sums']:
            ax5.hist(stats['final_dealer_sums'], bins=range(17, 32), 
                    color='#e74c3c', edgecolor='black', alpha=0.7, label='Dealer')
            ax5.axvline(x=21, color='green', linestyle='--', linewidth=2, label='21')
            ax5.axvline(x=np.mean(stats['final_dealer_sums']), color='blue', 
                       linestyle=':', linewidth=2, label=f'Mean: {np.mean(stats["final_dealer_sums"]):.1f}')
            ax5.set_xlabel('Final Sum')
            ax5.set_ylabel('Frequency')
            ax5.set_title('Dealer Final Sum Distribution')
            ax5.legend(fontsize=8)
        else:
            ax5.text(0.5, 0.5, "No dealer sum data", ha='center', va='center')
        
        # Subplot 6: Summary text
        ax6 = axes[1, 2]
        ax6.axis('off')
        
        avg_reward = stats['total_reward'] / total
        
        summary = f"""
        ═══════════════════════════════════════
        DECISION ANALYSIS
        ═══════════════════════════════════════
        
        State: ({player_sum}, {dealer_showing}, {self.usable_ace})
        
        Q-VALUES:
          Q(HIT):   {data['q_hit']:+.4f}
          Q(STICK): {data['q_stick']:+.4f}
          
        WHY {action_name}?
        The agent chose {action_name} because
        Q({action_name}) is {'higher' if action == STICK else 'higher or equal'}.
        
        EXPECTED OUTCOME:
          Average Reward: {avg_reward:+.4f}
          
        INTERPRETATION:
        """
        
        if action == HIT:
            summary += f"""
          With player sum {player_sum}:
          - Bust risk: {100*stats['player_busted']/total:.1f}%
          - But standing would likely lose
          - Expected improvement outweighs bust risk
        """
        else:
            summary += f"""
          With player sum {player_sum}:
          - Standing has {100*stats['player_won']/total:.1f}% win rate
          - Dealer busts {100*stats['dealer_busted']/total:.1f}% of time
          - Hitting would risk busting
        """
        
        ax6.text(0.05, 0.95, summary, transform=ax6.transAxes,
                fontsize=9, verticalalignment='top', fontfamily='monospace',
                bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        plt.tight_layout(rect=[0, 0, 1, 0.93])
        plt.show()
    
    def on_click(self, event):
        """Handle click events on the plot."""
        if event.inaxes != self.ax:
            return
        
        x = int(round(event.xdata))
        y = int(round(event.ydata))
        
        if 0 <= x < 10 and 0 <= y < 10:
            dealer_showing = x + 1
            player_sum = y + 12
            self.show_detailed_stats_figure(player_sum, dealer_showing)
    
    def create_plot(self):
        """Create the interactive policy plot."""
        self.fig, self.ax = plt.subplots(figsize=(12, 10))
        
        grids = get_stats_grids(self.algorithm, self.usable_ace)
        
        # Main policy heatmap
        im = self.ax.imshow(grids['policy'], cmap='RdYlGn', aspect='auto',
                           origin='lower', vmin=0, vmax=1)
        
        self.ax.set_xticks(range(10))
        self.ax.set_xticklabels(['A'] + list(range(2, 11)))
        self.ax.set_yticks(range(10))
        self.ax.set_yticklabels(range(12, 22))
        self.ax.set_xlabel("Dealer Showing", fontsize=12)
        self.ax.set_ylabel("Player Sum", fontsize=12)
        
        ace_str = "Usable Ace" if self.usable_ace else "No Usable Ace"
        self.ax.set_title(f"{self.title}\n{ace_str}\n(Click any cell for detailed statistics)", fontsize=12)
        
        # Add action labels with additional info
        for i in range(10):
            for j in range(10):
                policy_val = grids['policy'][i, j]
                text = "S" if policy_val == STICK else "H"
                color = 'white' if policy_val == STICK else 'black'
                self.ax.text(j, i, text, ha='center', va='center',
                           color=color, fontsize=11, fontweight='bold')
        
        # Legend
        self.ax.text(0.5, -0.08, "H = Hit (Red) | S = Stick (Green) | CLICK any cell for detailed analysis",
                    ha='center', transform=self.ax.transAxes, fontsize=11, fontweight='bold',
                    bbox=dict(boxstyle='round', facecolor='lightyellow', alpha=0.8))
        
        # Connect click event
        self.fig.canvas.mpl_connect('button_press_event', self.on_click)
        
        return self.fig, self.ax


def create_all_plots(koukosias, tzanidakis, save_plots=True):
    """Create all visualization plots."""
    print("\n" + "=" * 70)
    print("CREATING COMPREHENSIVE STATISTICAL PLOTS")
    print("=" * 70)
    
    save_prefix = "blackjack" if save_plots else None
    
    # 1. Individual algorithm statistics (4 plots: 2 algorithms x 2 ace conditions)
    print("\n1. Creating individual algorithm statistics plots...")
    for usable_ace in [True, False]:
        create_algorithm_stats_multiplot(koukosias, usable_ace, save_prefix)
        create_algorithm_stats_multiplot(tzanidakis, usable_ace, save_prefix)
    
    # 2. Side-by-side comparison plots (2 plots: 1 per ace condition)
    print("\n2. Creating comparison multiplots...")
    for usable_ace in [True, False]:
        create_comparison_multiplot(koukosias, tzanidakis, usable_ace, save_prefix)
    
    # 3. Policy difference plot
    print("\n3. Creating policy difference plot...")
    create_policy_difference_plot(koukosias, tzanidakis, save_prefix)
    
    # 4. Interactive plots (4 plots: clickable for detailed stats)
    print("\n4. Creating interactive policy plots...")
    print("   Click on any cell to see detailed statistics!")
    
    plots = []
    for usable_ace in [True, False]:
        plot_k = InteractivePolicyPlot(koukosias, usable_ace, "Koukosias Algorithm - Optimal Policy")
        plot_k.create_plot()
        plots.append(plot_k)
        
        plot_t = InteractivePolicyPlot(tzanidakis, usable_ace, "Tzanidakis Algorithm - Optimal Policy")
        plot_t.create_plot()
        plots.append(plot_t)
    
    print("\n" + "-" * 70)
    print("All plots created! Click on cells in the policy plots for detailed analysis.")
    print("=" * 70)
    
    plt.show()
    
    return plots


# ================================================================================
# POLICY COMPARISON FUNCTIONS
# ================================================================================

def compare_policies(koukosias, tzanidakis):
    """Compare the policies learned by both algorithms."""
    print("\n" + "=" * 70)
    print("POLICY COMPARISON")
    print("=" * 70)
    
    differences = []
    
    for usable_ace in [True, False]:
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        print(f"\n{ace_str}:")
        print("-" * 50)
        
        diff_count = 0
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                state = (player_sum, dealer_showing, usable_ace)
                
                k_action = koukosias.policy.get(state, HIT)
                t_action = tzanidakis.policy.get(state, HIT)
                
                if k_action != t_action:
                    diff_count += 1
                    k_str = "STICK" if k_action == STICK else "HIT"
                    t_str = "STICK" if t_action == STICK else "HIT"
                    differences.append((state, k_str, t_str))
                    print(f"  State {state}: Koukosias={k_str}, Tzanidakis={t_str}")
        
        if diff_count == 0:
            print("  No differences found - policies are identical!")
        else:
            print(f"  Total differences: {diff_count}")
    
    print("\n" + "=" * 70)
    return differences


def print_policies(koukosias, tzanidakis):
    """Print both policies side by side."""
    print("\n" + "=" * 70)
    print("OPTIMAL POLICIES")
    print("=" * 70)
    
    for usable_ace in [False, True]:
        ace_str = "WITH USABLE ACE" if usable_ace else "WITHOUT USABLE ACE"
        
        print(f"\n{ace_str}")
        print("-" * 60)
        
        # Header
        print(" " * 20 + "KOUKOSIAS" + " " * 20 + "TZANIDAKIS")
        print("Player\\Dealer", end="")
        for d in range(1, 11):
            label = "A" if d == 1 else str(d)
            print(f" {label:>2}", end="")
        print("    ", end="")
        print("Player\\Dealer", end="")
        for d in range(1, 11):
            label = "A" if d == 1 else str(d)
            print(f" {label:>2}", end="")
        print()
        
        # Policy grids
        for p in range(21, 11, -1):
            # Koukosias
            print(f"    {p:>2}       ", end="")
            for d in range(1, 11):
                state = (p, d, usable_ace)
                action = koukosias.policy.get(state, HIT)
                symbol = "S" if action == STICK else "H"
                print(f" {symbol:>2}", end="")
            
            print("    ", end="")
            
            # Tzanidakis
            print(f"    {p:>2}       ", end="")
            for d in range(1, 11):
                state = (p, d, usable_ace)
                action = tzanidakis.policy.get(state, HIT)
                symbol = "S" if action == STICK else "H"
                print(f" {symbol:>2}", end="")
            print()
    
    print("\nLegend: H = Hit, S = Stick")
    print("=" * 70)


# ================================================================================
# MAIN FUNCTION
# ================================================================================

def main():
    """
    Main function to run the combined comparison.
    
    This script:
    1. Trains both algorithms (Koukosias & Tzanidakis) on IDENTICAL episodes
    2. Compares the learned policies
    3. Creates comprehensive statistical visualizations:
       - Individual algorithm statistics (12 heatmaps per algorithm per ace condition)
       - Side-by-side comparison multiplots
       - Policy difference visualization
       - Interactive clickable policy plots with detailed statistics popups
    
    Total plots generated:
    - 4 individual algorithm stats (2 algorithms x 2 ace conditions) - 12 subplots each
    - 2 comparison multiplots (1 per ace condition) - 16 subplots each  
    - 1 policy difference plot - 6 subplots
    - 4 interactive policy plots (clickable for detailed analysis)
    """
    print("\n" + "=" * 70)
    print("BLACKJACK MONTE CARLO EXPLORING STARTS - ALGORITHM COMPARISON")
    print("=" * 70)
    print("Comparing: Koukosias vs Tzanidakis implementations")
    print("Both algorithms trained on IDENTICAL episodes for fair comparison")
    print("=" * 70)
    
    # Train both algorithms on the same episodes
    koukosias, tzanidakis = train_both_algorithms(num_episodes=500000, seed=42)
    
    # Print policies to console
    print_policies(koukosias, tzanidakis)
    
    # Compare policies and show differences
    compare_policies(koukosias, tzanidakis)
    
    # Create all visualization plots
    # This includes:
    # - Heatmaps for: Policy, Q-values, Bust rates, Win rates, Draw rates, Average reward
    # - Interactive plots where clicking shows detailed statistics
    create_all_plots(koukosias, tzanidakis, save_plots=True)


if __name__ == "__main__":
    main()
