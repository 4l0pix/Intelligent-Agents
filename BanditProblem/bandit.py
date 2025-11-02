import numpy as np
import matplotlib.pyplot as plt
from scipy.special import softmax

#configuration
n_arms = 5  #arm num
n_plays = 1000  #games num
n_experiments = 100  #experiments num for means

#true average arm values
np.random.seed(42)
true_means = np.random.uniform(0, 2, n_arms)
optimal_arm = np.argmax(true_means)

print(f"True Mean Values: {true_means}")
print(f"Optimal Arm: {optimal_arm} (mean value: {true_means[optimal_arm]:.2f})\n")


# ========== ε-GREEDY ==========
def epsilon_greedy(epsilon=0.1):
    total_rewards = []
    optimal_actions = []

    for _ in range(n_experiments):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        rewards = []
        optimal = []

        for _ in range(n_plays):
            #arm selection
            if np.random.random() < epsilon:
                arm = np.random.randint(n_arms)  #search
            else:
                arm = np.argmax(values)  #arm exploitation

            #reward
            reward = np.random.normal(true_means[arm], 1.0)

            #update
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

            rewards.append(reward)
            optimal.append(1 if arm == optimal_arm else 0)

        total_rewards.append(rewards)
        optimal_actions.append(optimal)

    return np.mean(total_rewards, axis=0), np.mean(optimal_actions, axis=0)


# ========== SOFTMAX ==========
def softmax_bandit(temperature=0.5):
    total_rewards = []
    optimal_actions = []

    for _ in range(n_experiments):
        counts = np.zeros(n_arms)
        values = np.zeros(n_arms)
        rewards = []
        optimal = []

        for _ in range(n_plays):
            #arm selection with softmax
            probs = softmax(values / temperature)
            arm = np.random.choice(n_arms, p=probs)

            #reward
            reward = np.random.normal(true_means[arm], 1.0)

            #update
            counts[arm] += 1
            values[arm] += (reward - values[arm]) / counts[arm]

            rewards.append(reward)
            optimal.append(1 if arm == optimal_arm else 0)

        total_rewards.append(rewards)
        optimal_actions.append(optimal)

    return np.mean(total_rewards, axis=0), np.mean(optimal_actions, axis=0)


#algorithm execution
eg_rewards, eg_optimal = epsilon_greedy(epsilon=0.1)
sm_rewards, sm_optimal = softmax_bandit(temperature=0.5)

#results
print("=== RESULTS ===")
print(f"ε-greedy (ε=0.1):")
print(f"  Total Reward: {np.sum(eg_rewards):.2f}")
print(f"  Mean Reward: {np.mean(eg_rewards):.2f}")
print(f" Optimal Decision: {np.mean(eg_optimal)*100:.1f}%")

print(f"\nSoftmax (τ=0.5):")
print(f"  Total Reward: {np.sum(sm_rewards):.2f}")
print(f"  Mean Reward: {np.mean(sm_rewards):.2f}")
print(f"  Optimal Decision: {np.mean(sm_optimal)*100:.1f}%")

#vi\sualization
plt.figure(figsize=(15, 5))

#1st Graph-->Mean Reward
plt.subplot(1, 3, 1)
plt.plot(eg_rewards, label='ε-greedy', alpha=0.8)
plt.plot(sm_rewards, label='Softmax', alpha=0.8)
plt.xlabel('Games')
plt.ylabel('Mean Reward')
plt.title('Per-Game-Reward')
plt.legend()
plt.grid(alpha=0.3)

#2nd Graph --> Total Mean Reward
plt.subplot(1, 3, 2)
plt.plot(np.cumsum(eg_rewards) / np.arange(1, n_plays + 1), label='ε-greedy', alpha=0.8)
plt.plot(np.cumsum(sm_rewards) / np.arange(1, n_plays + 1), label='Softmax', alpha=0.8)
plt.xlabel('Games')
plt.ylabel('Total Mean Reward')
plt.title('Total Mean Reward')
plt.legend()
plt.grid(alpha=0.3)

#3rd Graph--> Percentage of optimal Decisions
plt.subplot(1, 3, 3)
window = 50
eg_smooth = np.convolve(eg_optimal, np.ones(window)/window, mode='valid')
sm_smooth = np.convolve(sm_optimal, np.ones(window)/window, mode='valid')
plt.plot(eg_smooth, label='ε-greedy', alpha=0.8)
plt.plot(sm_smooth, label='Softmax', alpha=0.8)
plt.xlabel('Games')
plt.ylabel('Percentage of Optimal Actions')
plt.title('Optimal Action %')
plt.legend()
plt.grid(alpha=0.3)

plt.tight_layout()
plt.savefig('bandit_results.png', dpi=150, bbox_inches='tight')
print("\nsaved results as 'bandit_results.png'")
# plt.show()  #this can also be used if run in GUI
