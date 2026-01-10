import numpy as np

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

def print_values(V, title):
    #task 2: print state value function in grid format
    print(f"\n{title}")
    print("-" * 40)
    for i in range(GRID_SIZE):
        row = [f"{V[i*GRID_SIZE + j]:7.2f}" for j in range(GRID_SIZE)]
        print(" ".join(row))

def compute_optimal_actions(V):
    #task 3: Compute optimal actions for each non-terminal state
    print("\npptimal actions per state:")
    print("-" * 40)
    
    optimal_grid = [['T' if s in TERMINAL_STATES else '' for s in range(i*4, i*4+4)] for i in range(4)]
    
    for s in range(16):
        if s in TERMINAL_STATES:
            continue
        
        Q = {}
        for a in ACTIONS:
            s_next = get_next_state(s, a)
            Q[a] = -1 + GAMMA * V[s_next]
        
        max_Q = max(Q.values())
        optimal = [a for a, q in Q.items() if abs(q - max_Q) < 1e-6]
        
        row, col = s // GRID_SIZE, s % GRID_SIZE
        symbols = {'up': '↑', 'down': '↓', 'left': '←', 'right': '→'} #here are some ascii arrows for better visualization
        optimal_grid[row][col] = ''.join(symbols[a] for a in optimal)
        
        print(f"State {s:2d}: {', '.join(optimal):20s} Q-values: {Q}")
    
    print("\npptimal policy grid:")
    print("-" * 40)
    for row in optimal_grid:
        print(" | ".join(f"{cell:^6}" for cell in row))

if __name__ == "__main__":
    #task 1: two-table method
    V_two = iterative_policy_evaluation_two_tables()
    print_values(V_two, "task 1: value function (Two Tables)")
    
    #task 1b: single-table method
    V_single = iterative_policy_evaluation_single_table()
    print_values(V_single, "task 1b: value function (Single Table)")
    
    #task 3: optimal actions using the converged value function
    compute_optimal_actions(V_two)
