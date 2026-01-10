import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches

#grid: 4x4, states 0-15. Terminals: 0 and 15. non-terminals: 1-14
GRID_SIZE = 4
TERMINAL_STATES = {0, 15}
ACTIONS = {'up': -4, 'down': 4, 'left': -1, 'right': 1}
GAMMA = 1
THETA = 1e-4

def get_next_state(s, action):
    #get next state given current state and action
    row, col = s // GRID_SIZE, s % GRID_SIZE
    if action == 'up':
        row = max(0, row - 1)
    elif action == 'down':
        row = min(GRID_SIZE - 1, row + 1)
    elif action == 'left':
        col = max(0, col - 1)
    elif action == 'right':
        col = min(GRID_SIZE - 1, col + 1)
    return row * GRID_SIZE + col

def iterative_policy_evaluation_two_tables():
    #task 1: two table iterative policy evaluation
    V_old = np.zeros(16)
    V_new = np.zeros(16)
    
    while True:
        for s in range(16):
            if s in TERMINAL_STATES:
                continue
            value = 0
            for a in ACTIONS:
                s_next = get_next_state(s, a)
                value += 0.25 * (-1 + GAMMA * V_old[s_next])
            V_new[s] = value
        
        delta = np.max(np.abs(V_new - V_old))
        V_old = V_new.copy()
        if delta < THETA:
            break
    return V_new

def iterative_policy_evaluation_single_table():
    #task 1b: single table (in place) iterative policy evaluation
    V = np.zeros(16)
    
    while True:
        delta = 0
        for s in range(16):
            if s in TERMINAL_STATES:
                continue
            v_old = V[s]
            value = 0
            for a in ACTIONS:
                s_next = get_next_state(s, a)
                value += 0.25 * (-1 + GAMMA * V[s_next])
            V[s] = value
            delta = max(delta, abs(V[s] - v_old))
        if delta < THETA:
            break
    return V

def plot_values(V, title, ax=None):
    #plot state value function as a heatmap
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    grid = V.reshape(GRID_SIZE, GRID_SIZE)
    im = ax.imshow(grid, cmap='RdYlGn', aspect='equal')
    
    #add text annotations
    for i in range(GRID_SIZE):
        for j in range(GRID_SIZE):
            state = i * GRID_SIZE + j
            value = V[state]
            color = 'white' if abs(value) > 10 else 'black'
            if state in TERMINAL_STATES:
                ax.text(j, i, f'T\n{value:.1f}', ha='center', va='center', 
                       fontsize=12, fontweight='bold', color=color)
            else:
                ax.text(j, i, f'{state}\n{value:.1f}', ha='center', va='center', 
                       fontsize=10, color=color)
    
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE))
    ax.set_yticklabels(range(GRID_SIZE))
    ax.set_title(title, fontsize=12, fontweight='bold')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    
    return im

def plot_optimal_policy(V, ax=None):
    #plot optimal policy with arrows
    if ax is None:
        fig, ax = plt.subplots(figsize=(6, 6))
    
    #create grid background
    grid = V.reshape(GRID_SIZE, GRID_SIZE)
    ax.imshow(grid, cmap='RdYlGn', aspect='equal', alpha=0.3)
    
    #arrow offsets for each action
    arrow_delta = {
        'up': (0, -0.3),
        'down': (0, 0.3),
        'left': (-0.3, 0),
        'right': (0.3, 0)
    }
    
    for s in range(16):
        row, col = s // GRID_SIZE, s % GRID_SIZE
        
        if s in TERMINAL_STATES:
            ax.text(col, row, 'T', ha='center', va='center', 
                   fontsize=16, fontweight='bold', color='darkgreen')
            continue
        
        #compute q-values for all actions
        Q = {}
        for a in ACTIONS:
            s_next = get_next_state(s, a)
            Q[a] = -1 + GAMMA * V[s_next]
        
        max_Q = max(Q.values())
        optimal = [a for a, q in Q.items() if abs(q - max_Q) < 1e-6]
        
        #draw arrows for optimal actions
        for a in optimal:
            dx, dy = arrow_delta[a]
            ax.annotate('', xy=(col + dx, row + dy), xytext=(col, row),
                       arrowprops=dict(arrowstyle='->', color='darkblue', lw=2))
    
    ax.set_xticks(range(GRID_SIZE))
    ax.set_yticks(range(GRID_SIZE))
    ax.set_xticklabels(range(GRID_SIZE))
    ax.set_yticklabels(range(GRID_SIZE))
    ax.set_title('optimal policy', fontsize=12, fontweight='bold')
    ax.set_xlabel('column')
    ax.set_ylabel('row')
    ax.set_xlim(-0.5, GRID_SIZE - 0.5)
    ax.set_ylim(GRID_SIZE - 0.5, -0.5)

if __name__ == "__main__":
    #task 1: two-table method
    V_two = iterative_policy_evaluation_two_tables()
    
    #task 1b: single-table method
    V_single = iterative_policy_evaluation_single_table()
    
    #create figure with 3 subplots
    fig, axes = plt.subplots(1, 3, figsize=(15, 5))
    
    #plot value functions
    im1 = plot_values(V_two, 'value function (two tables)', axes[0])
    im2 = plot_values(V_single, 'value function (single table)', axes[1])
    
    #plot optimal policy
    plot_optimal_policy(V_two, axes[2])
    
    #add colorbar
    
    plt.suptitle('gridworld policy evaluation results', fontsize=14, fontweight='bold')
    plt.tight_layout()
    plt.savefig('grid_policy_results.png', dpi=150, bbox_inches='tight')
    plt.close()
