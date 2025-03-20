import numpy as np
import matplotlib.pyplot as plt


class BernoulliBandit:
    def __init__(self, p1, p2):
        """
        Initialize the 2-armed Bernoulli bandit.

        Parameters:
        p1 (float): Probability of success for arm 1.
        p2 (float): Probability of success for
         arm 2.
        """
        self.p = [p1, p2]

    def pull(self, arm):
        """
        Simulate pulling an arm.

        Parameters:
        arm (int): The arm to pull (0 or 1).

        Returns:
        int: 1 if the pull is successful, 0 otherwise.
        """
        return np.random.binomial(1, self.p[arm])


class UCB:
    def __init__(self, num_arms):
        """
        Initialize the UCB algorithm.

        Parameters:
        num_arms (int): Number of arms.
        """
        self.num_arms = num_arms
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)

    def select_arm(self):
        """
        Select an arm using the UCB algorithm.

        Returns:
        int: The selected arm.
        """
        total_counts = np.sum(self.counts)
        if total_counts == 0:
            return np.random.choice(self.num_arms)

        ucb_values = self.values + np.sqrt(2 * np.log(total_counts) / (self.counts + 1e-5))
        return np.argmax(ucb_values)

    def update(self, arm, reward):
        """
        Update the algorithm's estimates.

        Parameters:
        arm (int): The arm that was pulled.
        reward (int): The reward received.
        """
        self.counts[arm] += 1
        n = self.counts[arm]
        value = self.values[arm]
        self.values[arm] = ((n - 1) / n) * value + (1 / n) * reward


class ThompsonSampling:
    def __init__(self, num_arms):
        """
        Initialize the Thompson Sampling algorithm.

        Parameters:
        num_arms (int): Number of arms.
        """
        self.num_arms = num_arms
        self.alpha = np.ones(num_arms)
        self.beta = np.ones(num_arms)

    def select_arm(self):
        """
        Select an arm using Thompson Sampling.

        Returns:
        int: The selected arm.
        """
        samples = [np.random.beta(self.alpha[i], self.beta[i]) for i in range(self.num_arms)]
        return np.argmax(samples)

    def update(self, arm, reward):
        """
        Update the algorithm's estimates.

        Parameters:
        arm (int): The arm that was pulled.
        reward (int): The reward received.
        """
        self.alpha[arm] += reward
        self.beta[arm] += 1 - reward


def simulate(bandit, algorithm, num_steps):
    """
    Simulate the bandit problem using the specified algorithm.

    Parameters:
    bandit (BernoulliBandit): The bandit environment.
    algorithm: The bandit algorithm (UCB or ThompsonSampling).
    num_steps (int): Number of steps to simulate.

    Returns:
    list: Cumulative regret over time.
    """
    rewards = []
    optimal_rewards = []
    cumulative_regret = []

    for step in range(num_steps):
        arm = algorithm.select_arm()
        reward = bandit.pull(arm)
        algorithm.update(arm, reward)

        rewards.append(reward)
        optimal_reward = max(bandit.p)
        optimal_rewards.append(optimal_reward)

        cumulative_regret.append(np.sum(optimal_rewards) - np.sum(rewards))

    return cumulative_regret


# Parameters
p1, p2 = 0.6, 0.4  # Probabilities for arm 1 and arm 2
num_steps = 1000
num_runs = 100

# Initialize bandit
bandit = BernoulliBandit(p1, p2)

# Simulate UCB
ucb_regret = np.zeros(num_steps)
for _ in range(num_runs):
    ucb = UCB(num_arms=2)
    ucb_regret += np.array(simulate(bandit, ucb, num_steps))
ucb_regret /= num_runs

# Simulate Thompson Sampling
ts_regret = np.zeros(num_steps)
for _ in range(num_runs):
    ts = ThompsonSampling(num_arms=2)
    ts_regret += np.array(simulate(bandit, ts, num_steps))
ts_regret /= num_runs

# Plot regret curves
plt.plot(ucb_regret, label='UCB')
plt.plot(ts_regret, label='Thompson Sampling')
plt.xlabel('Steps')
plt.ylabel('Cumulative Regret')
plt.title('Regret Curves for UCB and Thompson Sampling')
plt.legend()
plt.show()