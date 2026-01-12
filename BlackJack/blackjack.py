#koukosias athanasios 2025-2026

import numpy as np
import matplotlib.pyplot as plt
from mpl_toolkits.mplot3d import Axes3D
from collections import defaultdict
import random


#actions
HIT = 0
STICK = 1
ACTIONS = [HIT, STICK]

#card values
CARD_VALUES = [1, 2, 3, 4, 5, 6, 7, 8, 9, 10, 10, 10, 10]  #a, 2-10, j, q, k


def draw_card():
    #draw a card from an infinite deck.
    return random.choice(CARD_VALUES)


def draw_hand():
    #draw initial two cards.
    return [draw_card(), draw_card()]


def usable_ace(hand):
    #check if hand has a usable ace (can count as 11 without busting).
    return 1 in hand and sum(hand) + 10 <= 21


def sum_hand(hand):
    #return the sum of the hand, treating ace as 11 if beneficial.
    if usable_ace(hand):
        return sum(hand) + 10
    return sum(hand)


def is_bust(hand):
    #check if hand is busted (over 21).
    return sum_hand(hand) > 21


def get_state(player_hand, dealer_showing):
    #get state representation.
    #state: (player_sum, dealer_showing, usable_ace)
    #- player_sum: 12-21 (below 12, player always hits)
    #- dealer_showing: 1-10 (ace=1)
    #- usable_ace: true/false
    return (sum_hand(player_hand), dealer_showing, usable_ace(player_hand))


def player_policy(state, Q, policy_type='greedy'):
    #player policy.
    #- 'initial': stick only on 20 or 21
    #- 'greedy': follow the learned q-values
    player_sum, dealer_showing, has_usable_ace = state
    
    if policy_type == 'initial':
        #initial policy: stick only on 20 or 21
        return STICK if player_sum >= 20 else HIT
    else:
        #greedy policy based on q-values
        if Q[(state, HIT)] >= Q[(state, STICK)]:
            return HIT
        else:
            return STICK


def dealer_policy(dealer_hand):
    #dealer's fixed strategy: hit if sum < 17, otherwise stick.
    return HIT if sum_hand(dealer_hand) < 17 else STICK


def play_game(initial_state=None, initial_action=None, Q=None):
    #play one episode of blackjack.
    #returns: list of (state, action, reward) tuples
    #if initial_state and initial_action are provided, use them (exploring starts).
    #initialize player's hand
    player_hand = draw_hand()
    
    #initialize dealer's hand
    dealer_hand = draw_hand()
    dealer_showing = dealer_hand[0]  #first card is face up
    
    #for exploring starts: set up initial state
    if initial_state is not None:
        #we need to create a hand that matches the initial state
        player_sum, dealer_showing, has_usable_ace = initial_state
        
        #create player hand matching the state
        if has_usable_ace:
            #hand with usable ace: ace + (sum - 11)
            player_hand = [1, player_sum - 11]
        else:
            #hand without usable ace
            if player_sum <= 11:
                player_hand = [player_sum]
            else:
                #split into two cards
                player_hand = [10, player_sum - 10]
        
        #create dealer hand with the showing card
        dealer_hand = [dealer_showing, draw_card()]
    
    #episode history
    episode = []
    
    #check for naturals (21 with initial two cards)
    player_sum = sum_hand(player_hand)
    
    #player's turn
    while True:
        player_sum = sum_hand(player_hand)
        
        #if player sum < 12, always hit (no decision needed)
        if player_sum < 12:
            player_hand.append(draw_card())
            continue
        
        state = get_state(player_hand, dealer_showing)
        
        #choose action
        if initial_action is not None:
            action = initial_action
            initial_action = None  #only use initial action once
        elif Q is not None:
            action = player_policy(state, Q, 'greedy')
        else:
            action = player_policy(state, None, 'initial')
        
        episode.append((state, action))
        
        if action == STICK:
            break
        else:  #hit
            player_hand.append(draw_card())
            if is_bust(player_hand):
                #player busts, loses
                return [(s, a, 0) for s, a in episode[:-1]] + [(episode[-1][0], episode[-1][1], -1)]
    
    #dealer's turn (only if player didn't bust)
    while dealer_policy(dealer_hand) == HIT:
        dealer_hand.append(draw_card())
    
    #determine winner
    player_sum = sum_hand(player_hand)
    dealer_sum = sum_hand(dealer_hand)
    
    if is_bust(dealer_hand):
        reward = 1  #dealer busts, player wins
    elif dealer_sum > player_sum:
        reward = -1  #dealer wins
    elif dealer_sum < player_sum:
        reward = 1  #player wins
    else:
        reward = 0  #tie
    
    #assign reward to all state-action pairs in episode
    return [(s, a, 0) for s, a in episode[:-1]] + [(episode[-1][0], episode[-1][1], reward)]


def monte_carlo_es(num_episodes=500000):
    #monte carlo exploring starts algorithm to find optimal policy.
    #initialize q-values and returns
    Q = defaultdict(float)
    returns_sum = defaultdict(float)
    returns_count = defaultdict(int)
    
    #all possible states for exploring starts
    #player_sum: 12-21, dealer_showing: 1-10, usable_ace: true/false
    all_states = []
    for player_sum in range(12, 22):
        for dealer_showing in range(1, 11):
            for has_usable_ace in [True, False]:
                all_states.append((player_sum, dealer_showing, has_usable_ace))
    
    for episode_num in range(num_episodes):
        if (episode_num + 1) % 100000 == 0:
            print(f"Episode {episode_num + 1}/{num_episodes}")
        
        #exploring starts: random initial state and action
        initial_state = random.choice(all_states)
        initial_action = random.choice(ACTIONS)
        
        #generate episode
        episode = play_game(initial_state, initial_action, Q)
        
        #first-visit mc: update q-values
        visited = set()
        G = 0  #return (gamma = 1, so no discounting)
        
        #process episode in reverse order
        for t in range(len(episode) - 1, -1, -1):
            state, action, reward = episode[t]
            G = G + reward  #gamma = 1
            
            state_action = (state, action)
            if state_action not in visited:
                visited.add(state_action)
                returns_sum[state_action] += G
                returns_count[state_action] += 1
                Q[state_action] = returns_sum[state_action] / returns_count[state_action]
    
    #extract optimal policy
    policy = {}
    for state in all_states:
        if Q[(state, HIT)] >= Q[(state, STICK)]:
            policy[state] = HIT
        else:
            policy[state] = STICK
    
    return Q, policy


def plot_policy(policy, title_suffix=""):
    #plot the optimal policy for usable and non-usable ace cases.
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for idx, usable_ace in enumerate([True, False]):
        ax = axes[idx]
        
        #create policy grid
        policy_grid = np.zeros((10, 10))  #player_sum (12-21) x dealer_showing (1-10)
        
        for player_sum in range(12, 22):
            for dealer_showing in range(1, 11):
                state = (player_sum, dealer_showing, usable_ace)
                action = policy.get(state, HIT)
                policy_grid[player_sum - 12, dealer_showing - 1] = action
        
        #plot
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
        
        #add legend
        for i in range(10):
            for j in range(10):
                text = "S" if policy_grid[i, j] == STICK else "H"
                color = 'white' if policy_grid[i, j] == STICK else 'black'
                ax.text(j, i, text, ha='center', va='center', color=color, fontsize=8)
    
    plt.suptitle(f"Blackjack Optimal Policy (Monte Carlo ES){title_suffix}")
    plt.tight_layout()
    plt.savefig('policy.png', dpi=150)
    plt.show()


def plot_value_function(Q, title_suffix=""):
    #plot the state-value function as 3d surface plots.
    fig = plt.figure(figsize=(14, 5))
    
    for idx, usable_ace in enumerate([True, False]):
        ax = fig.add_subplot(1, 2, idx + 1, projection='3d')
        
        #create value grid
        X = np.arange(1, 11)  #dealer showing
        Y = np.arange(12, 22)  #player sum
        X, Y = np.meshgrid(X, Y)
        
        Z = np.zeros_like(X, dtype=float)
        
        for i, player_sum in enumerate(range(12, 22)):
            for j, dealer_showing in enumerate(range(1, 11)):
                state = (player_sum, dealer_showing, usable_ace)
                #v(s) = max_a q(s, a)
                Z[i, j] = max(Q[(state, HIT)], Q[(state, STICK)])
        
        ax.plot_surface(X, Y, Z, cmap='viridis', edgecolor='none', alpha=0.8)
        
        ax.set_xlabel("Dealer Showing")
        ax.set_ylabel("Player Sum")
        ax.set_zlabel("Value")
        
        ace_str = "Usable Ace" if usable_ace else "No Usable Ace"
        ax.set_title(f"State-Value Function ({ace_str})")
    
    plt.suptitle(f"Blackjack State-Value Function (Monte Carlo ES){title_suffix}")
    plt.tight_layout()
    plt.savefig('value_function.png', dpi=150)
    plt.show()


def evaluate_policy(policy, num_episodes=100000):
    #evaluate a policy by playing many episodes and computing win rate.
    wins = 0
    losses = 0
    ties = 0
    
    for _ in range(num_episodes):
        #play without exploring starts
        player_hand = draw_hand()
        dealer_hand = draw_hand()
        dealer_showing = dealer_hand[0]
        
        #player's turn
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
                #dealer's turn
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
    print(f"\npolicy evaluation ({num_episodes} episodes):")
    print(f"  wins:   {wins} ({100*wins/total:.2f}%)")
    print(f"  losses: {losses} ({100*losses/total:.2f}%)")
    print(f"  ties:   {ties} ({100*ties/total:.2f}%)")
    print(f"  expected return: {(wins - losses) / total:.4f}")
    
    return wins, losses, ties


def print_policy(policy):
    #print the policy in a readable format.
    print("OPTIMAL POLICY")

    
    for usable_ace in [False, True]:
        ace_str = "WITH USABLE ACE" if usable_ace else "WITHOUT USABLE ACE"
        print(f"\n{ace_str}")
        print("-" * 40)
        
        #header
        print("Player\\Dealer", end="")
        for d in range(1, 11):
            label = "A" if d == 1 else str(d)
            print(f" {label:>2}", end="")
        print()
        
        #policy grid
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
    #run monte carlo es
    print("\ntraining with Monte Carlo Exploring Starts")
    
    Q, policy = monte_carlo_es(num_episodes=500000)
    
    #print the optimal policy
    print_policy(policy)
    
    #evaluate the learned policy
    evaluate_policy(policy, num_episodes=100000)
    
    #plot results
    plot_policy(policy)
    plot_value_function(Q)
 


if __name__ == "__main__":
    main()
