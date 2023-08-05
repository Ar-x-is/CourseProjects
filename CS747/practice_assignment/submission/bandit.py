import numpy as np
import matplotlib.pyplot as plt


# Create an Arm class
class Arm:
    def __init__(self, mean):
        self.mean = mean
        pass

    def pull(self):
        return np.random.choice([0, 1], p=[1 - self.mean, self.mean])


# Creating Bandit Instances
class Bandit:
    def __init__(self, means, seed):
        np.random.seed(seed)
        self.arms = np.array([Arm(mean) for mean in means], dtype=object)

    def pull_arm(self, arm_id):
        return self.arms[arm_id].pull()


# epsilon-greedy algorithm
def epsilon_greedy(bandit, horizon, epsilon):
    num_arms = len(bandit.arms)
    arm_means = np.zeros(num_arms)
    arm_pulls = np.zeros(num_arms)
    cum_reward = 0
    for _ in range(horizon):
        p = np.random.uniform()
        if p < epsilon:
            arm_id = np.random.randint(num_arms)
        else:
            arm_id = np.argmax(arm_means)
        reward = bandit.pull_arm(arm_id)
        cum_reward += reward
        arm_means[arm_id] = (arm_means[arm_id] * arm_pulls[arm_id] + reward) / (
            arm_pulls[arm_id] + 1
        )
        arm_pulls[arm_id] += 1

    return cum_reward, arm_means, arm_pulls


# UCB algorithm
def ucb(bandit, horizon):
    num_arms = len(bandit.arms)
    arm_means = np.zeros(num_arms)
    arm_pulls = np.ones(num_arms)
    confidence_bounds = np.zeros(num_arms)
    for i in range(num_arms):
        arm_means[i] = bandit.pull_arm(i)
        confidence_bounds[i] = np.sqrt((2 * np.log(horizon)) / 1)

    cum_reward = 0
    ucb = arm_means + confidence_bounds
    for _ in range(horizon-num_arms):
        arm_id = np.argmax(ucb)
        reward = bandit.pull_arm(arm_id)
        cum_reward += reward
        arm_means[arm_id] = (arm_means[arm_id] * arm_pulls[arm_id] + reward) / (
            arm_pulls[arm_id] + 1
        )
        arm_pulls[arm_id] += 1
        ucb[arm_id] = arm_means[arm_id] + np.sqrt(
            (2 * np.log(horizon)) / arm_pulls[arm_id]
        )

    return cum_reward, arm_means, arm_pulls


# KL-UCB algorithm
def kl_ucb(bandit, horizon):

    def KL(p, q):
        if p == 0:
            return 0
        elif q == 1:
            return np.inf
        elif p == 1:
            return p * np.log(p / q)
        return p * np.log(p / q) + (1 - p) * np.log((1 - p) / (1 - q))

    def find_q(p, u, horizon):

        c = 3
        a = p
        b = 1
        q_max = p

        while b - a > 10e-4:
            q = (a + b)/2
            if u * KL(p, q) <= np.log(horizon) + c * np.log(np.log(horizon)):
                q_max = q
                a = q
            else:
                b = q

        return q_max

    num_arms = len(bandit.arms)
    arm_means = np.zeros(num_arms)
    arm_pulls = np.ones(num_arms)
    for i in range(num_arms):
        arm_means[i] = bandit.pull_arm(i)

    cum_reward = 0
    for _ in range(horizon-num_arms):

        kl_ucb = [find_q(arm_means[i], arm_pulls[i], horizon) for i in range(num_arms)]
        arm_id = np.argmax(kl_ucb)
        reward = bandit.pull_arm(arm_id)
        cum_reward += reward
        arm_means[arm_id] = (arm_means[arm_id] * arm_pulls[arm_id] + reward) / (arm_pulls[arm_id] + 1)
        arm_pulls[arm_id] += 1

    return cum_reward, arm_means, arm_pulls


# Thompson Sampling algorithm
def thompson_sampling(bandit, horizon):
    num_arms = len(bandit.arms)
    arm_means = np.zeros(num_arms)
    arm_fails = np.zeros(num_arms)
    arm_successes = np.zeros(num_arms)
    for i in range(num_arms):
        arm_means[i] = bandit.pull_arm(i)
        if arm_means[i] == 1:
            arm_successes[i] += 1
        elif arm_means[i] == 0:
            arm_fails[i] += 1

    cum_reward = 0
    for _ in range(horizon):
        samples = np.random.beta(np.array(arm_successes)+1, np.array(arm_fails)+1, len(arm_successes))
        arm_id = np.argmax(samples)
        reward = bandit.pull_arm(arm_id)
        cum_reward += reward
        arm_pulls = arm_fails[arm_id] + arm_successes[arm_id]
        arm_means[arm_id] = (arm_means[arm_id] * arm_pulls + reward) / (arm_pulls + 1)
        if reward == 0:
            arm_fails[arm_id] += 1
        else:
            arm_successes[arm_id] += 1

    arm_pulls = [arm_fails[i] + arm_successes[i] for i in range(num_arms)]
    return cum_reward, arm_means, arm_pulls


def thompson_sampling_with_hint(bandit, horizon, p):

    def minimise_loss(emp_arm_means, p):
        diff = np.zeros((num_arms, num_arms))
        for i in range(num_arms):
            for j in range(num_arms):
                diff[i][j] = np.abs(emp_arm_means[i] - p[j])

        loss_indices = np.zeros(num_arms)
        rows = []
        cols = []
        min_loss = np.inf
        min_loss_idx = 0
        for _ in range(num_arms):
            for i in range(num_arms):
                if i in rows:
                    continue
                for j in range(num_arms):
                    if j in cols:
                        continue
                    if diff[i][j] < min_loss:
                        min_loss = diff[i][j]
                        min_loss_idx = [i, j]
            rows.append(min_loss_idx[0])
            cols.append(min_loss_idx[1])
            loss_indices[min_loss_idx[0]] = min_loss_idx[1]
        
        return loss_indices.astype(int)


    # p: the real means hint is to be passed as an ascending list of the means
    num_arms = len(bandit.arms)
    emp_arm_means = np.zeros(num_arms)
    # failures(0): f, successes(1): s
    f = np.zeros(num_arms)
    s = np.zeros(num_arms)
    cum_reward = 0
    
    for i in range(horizon):
        emp_arm_means = np.array([(s[i]+1) / (s[i] + f[i] + 2) for i in range(num_arms)])
        # indices = minimise_loss(emp_arm_means, p)
        indices = np.argsort(emp_arm_means)
        bandit.arms = np.take_along_axis(bandit.arms, indices, axis=0)
        f = np.take_along_axis(f, indices, axis=0)
        s = np.take_along_axis(s, indices, axis=0)
        emp_arm_means = np.take_along_axis(emp_arm_means, indices, axis=0)
        variances = np.array([((s[i]+1) * (f[i]+1)) / ((s[i]+f[i]+2)**2 * (s[i]+f[i]+3)) for i in range(num_arms)])
        samples = np.zeros(num_arms)
        for i in range(num_arms):
            alpha = ((p[i]**2 * (1-p[i])) / variances[i]) - p[i]
            beta = ((p[i] * (1-p[i])**2) / variances[i]) - (1-p[i])
            samples[i] = np.random.beta(alpha, beta)
        arm_id = np.argmax(samples)
        reward = bandit.pull_arm(arm_id)
        cum_reward += reward
        if reward == 0:
            f[arm_id] += 1
        else:
            s[arm_id] += 1

    arm_pulls = [f[i] + s[i] for i in range(num_arms)]
    return cum_reward, emp_arm_means, arm_pulls


iterations = 100
epsilon = 0.333
horizon = 100
real_means = np.array([0.2, 0.5, 0.75])
regret = np.zeros(iterations)
for i in range(iterations):
    
    bandit = Bandit(real_means, i)
    max_reward = np.max(real_means) * horizon
    cum_reward, arm_means, arm_pulls = epsilon_greedy(bandit, horizon, i/iterations)
    # cum_reward, arm_means, arm_pulls = ucb(bandit, horizon)
    # cum_reward, arm_means, arm_pulls = kl_ucb(bandit, horizon)
    # cum_reward, arm_means, arm_pulls = thompson_sampling(bandit, horizon)
    # cum_reward, arm_means, arm_pulls = thompson_sampling_with_hint(bandit, horizon, np.sort(real_means))
    # print("Cumulative Reward: ", cum_reward)
    # print("Maximum Reward: ", max_reward)
    # print("Arm Means: ", arm_means)
    # print("Arm Pulls: ", arm_pulls)
    regret[i] = max_reward - cum_reward

print("Regret: ", regret)
# print("Average Regret: ", np.mean(regret))
