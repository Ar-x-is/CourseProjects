"""
NOTE: You are only allowed to edit this file between the lines that say:
    # START EDITING HERE
    # END EDITING HERE

This file contains the base Algorithm class that all algorithms should inherit
from. Here are the method details:
    - __init__(self, num_arms, horizon): This method is called when the class
        is instantiated. Here, you can add any other member variables that you
        need in your algorithm.
    
    - give_pull(self): This method is called when the algorithm needs to
        select an arm to pull. The method should return the index of the arm
        that it wants to pull (0-indexed).
    
    - get_reward(self, arm_index, reward): This method is called just after the 
        give_pull method. The method should update the algorithm's internal
        state based on the arm that was pulled and the reward that was received.
        (The value of arm_index is the same as the one returned by give_pull.)

We have implemented the epsilon-greedy algorithm for you. You can use it as a
reference for implementing your own algorithms.
"""

import numpy as np
import math
# Hint: math.log is much faster than np.log for scalars

class Algorithm:
    def __init__(self, num_arms, horizon):
        self.num_arms = num_arms
        self.horizon = horizon
    
    def give_pull(self):
        raise NotImplementedError
    
    def get_reward(self, arm_index, reward):
        raise NotImplementedError

# Example implementation of Epsilon Greedy algorithm
class Eps_Greedy(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # Extra member variables to keep track of the state
        self.eps = 0.1
        self.counts = np.zeros(num_arms)
        self.values = np.zeros(num_arms)
    
    def give_pull(self):
        if np.random.random() < self.eps:
            return np.random.randint(self.num_arms)
        else:
            return np.argmax(self.values)
    
    def get_reward(self, arm_index, reward):
        self.counts[arm_index] += 1
        n = self.counts[arm_index]
        value = self.values[arm_index]
        new_value = ((n - 1) / n) * value + (1 / n) * reward
        self.values[arm_index] = new_value

# START EDITING HERE
# You can use this space to define any helper functions that you need

# KL divergence
def kl(p, q):
    if q < 1 and p > 0:
        return p*math.log(p/q) + (1-p)*math.log((1-p)/(1-q))
    elif p == 0 and q < 1:
        return -math.log(1-q)
    else:
        return np.inf

# binary search to find q in the KL-UCB algorithm
def bin_search(t, p, u):
    # p : empirical mean
    # u : number of pulls
    # t : total number of pulls

    # we want to find max q such that u*kl(p, q) <= ln(t) + cln(ln(t))
    a = p # lower bound for binary search
    b = 1 # upper bound for binary search

    c = 0 # parameter for the bound

    if t > 1:
        bound = math.log(t) + c*math.log(math.log(t)) 
    else:
        bound = -np.inf

    x = (a+b)/2 # midpoint
    while b-a > 1e-3:
        x = (a+b)/2 # midpoint
        
        if u*kl(p, x) > bound:
            b = x
        else:
            a = x

    return x

# END EDITING HERE

class UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # START EDITING HERE
        self.counts = np.zeros(num_arms)
        self.total_pulls = 0

        self.emp_means = np.zeros(num_arms)
        self.bounds = np.inf * np.ones(num_arms)

        self.ucb = np.zeros(num_arms)
        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.ucb)
        # END EDITING HERE  
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE

        # calculate the empirical mean
        num_pulls = self.counts[arm_index]
        new_mean = (num_pulls * self.emp_means[arm_index] + reward)/(num_pulls + 1)
        self.emp_means[arm_index] = new_mean
        self.counts[arm_index] += 1
        self.total_pulls += 1
        
        # calculate the confidence bounds (all of them change as sqrt(log(t)))
        # make sure not to divide by 0
        if self.total_pulls < self.num_arms: # this condition makes us pull all arms once before using ucb
            self.bounds[arm_index] = math.sqrt(2 * math.log(self.num_arms) / 1)
        else:
            self.bounds = np.array([math.sqrt(2 * math.log(self.total_pulls)/self.counts[arm_id]) for arm_id in range(self.num_arms)])

        # update the values
        self.ucb = self.emp_means + self.bounds
        
        # END EDITING HERE


class KL_UCB(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.counts = np.zeros(num_arms)
        self.total_pulls = 0

        self.emp_means = np.zeros(num_arms)
        self.values = np.inf*np.ones(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        return np.argmax(self.values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        # calculate the empirical mean
        num_pulls = self.counts[arm_index]
        new_mean = (num_pulls * self.emp_means[arm_index] + reward)/(num_pulls + 1)
        self.emp_means[arm_index] = new_mean
        self.counts[arm_index] += 1
        self.total_pulls += 1

        # update the values
        if self.total_pulls < self.num_arms: # this condition makes us pull all arms once before using ucb
            self.values[arm_index] = bin_search(self.num_arms, self.emp_means[arm_index], 1)
        else:
            self.values = np.array([bin_search(self.total_pulls, self.emp_means[arm_id], self.counts[arm_id]) for arm_id in range(self.num_arms)])

        # END EDITING HERE

class Thompson_Sampling(Algorithm):
    def __init__(self, num_arms, horizon):
        super().__init__(num_arms, horizon)
        # You can add any other variables you need here
        # START EDITING HERE

        self.successes = np.zeros(num_arms)
        self.failures = np.zeros(num_arms)

        self.emp_means = np.zeros(num_arms)

        # END EDITING HERE
    
    def give_pull(self):
        # START EDITING HERE
        self.values = np.random.beta(self.successes + 1, self.failures + 1, size=self.num_arms)
        return np.argmax(self.values)
        # END EDITING HERE
    
    def get_reward(self, arm_index, reward):
        # START EDITING HERE
        
        # calculate the empirical mean
        num_pulls = self.successes[arm_index] + self.failures[arm_index]
        new_mean = (num_pulls * self.emp_means[arm_index] + reward)/(num_pulls + 1)
        self.emp_means[arm_index] = new_mean
        if reward == 1:
            self.successes[arm_index] += 1
        else:
            self.failures[arm_index] += 1

        # END EDITING HERE
