#! /usr/bin/python
import random,argparse,sys
parser = argparse.ArgumentParser()

import pulp 
import numpy as np
import math

class MDP():
    def __init__(self, path):
        # get MDP from filepath

        self.lines= []
        with open(path) as f:
            self.lines = f.readlines()
            self.lines = [line.strip() for line in self.lines]
            self.lines = [line.split() for line in self.lines]

        S = int(self.lines[0][1])
        A = int(self.lines[1][1])
        T = np.zeros((S,A,S))
        R = np.zeros((S,A,S))
        gamma = float(self.lines[-1][1])
        if self.lines[-2][1] == "continuing":
            mdptype = "continuing"
            end = []
        else:
            mdptype = "episodic"
            end = [int(x) for x in self.lines[2][1:]]

        for line in self.lines[3:-2]:
            s = int(line[1])
            a = int(line[2])
            s1 = int(line[3])
            r = float(line[4])
            t = float(line[5])
            T[s][a][s1] = t
            R[s][a][s1] = r

        self.S = S
        self.T = T
        self.A = A
        self.R = R
        self.gamma = gamma
        self.mdptype = mdptype
        self.end = end 


class Policy():
    def __init__(self, path=None, mdp=None, actions=None):
        if path != None:
            with open(path) as f:
                self.lines = f.readlines()
                self.lines = [line.strip() for line in self.lines]

            self.S = len(self.lines)
            self.A = [int(i) for i in self.lines]
        else:
            self.S = mdp.S
            self.A = actions


def Evaluate_Policy(mdp, policy):
    values = np.zeros(mdp.S)
    actions = np.zeros(mdp.S)
    old_values = np.zeros(mdp.S)
    delta = 1
    delta_min = 1e-7
    while delta > delta_min:
        old_values = values.copy()
        for s in range(mdp.S):
            values[s] = sum([mdp.T[s][policy.A[s]][s1]*(
                mdp.R[s][policy.A[s]][s1]+mdp.gamma*values[s1]
                ) for s1 in range(mdp.S)])
            actions[s] = policy.A[s]
        delta = np.max(np.abs(old_values-values))
    return values, actions
    

def Value_Iteration(mdp):
    infnorm_delta = np.inf
    delta_min = 1e-8
    values = np.zeros(mdp.S)
    optimal_actions = np.zeros(mdp.S, dtype=int)

    if mdp.mdptype == "continuing":
        while(infnorm_delta > delta_min):
            new_values = np.zeros(mdp.S)
            for s in range(mdp.S):
                optimal_actions[s] = np.argmax([sum([mdp.T[s][a][s1]*(
                    mdp.R[s][a][s1]+mdp.gamma*values[s1]
                    ) for s1 in range(mdp.S)]
                    ) for a in range(mdp.A)])
                new_values[s] = sum([mdp.T[s][optimal_actions[s]][s1]*(
                    mdp.R[s][optimal_actions[s]][s1]+mdp.gamma*values[s1]
                    ) for s1 in range(mdp.S)])
            infnorm_delta = np.max(np.abs(new_values-values))
            values = new_values
    else:
        while(infnorm_delta > delta_min):
            new_values = np.zeros(mdp.S)
            for s in range(mdp.S):
                if s in mdp.end:
                    continue
                optimal_actions[s] = np.argmax([sum([mdp.T[s][a][s1]*(
                    mdp.R[s][a][s1]+mdp.gamma*values[s1]
                    ) for s1 in range(mdp.S)]
                    ) for a in range(mdp.A)])
                new_values[s] = sum([mdp.T[s][optimal_actions[s]][s1]*(
                    mdp.R[s][optimal_actions[s]][s1]+mdp.gamma*values[s1]
                    ) for s1 in range(mdp.S)])
            infnorm_delta = np.max(np.abs(new_values-values))
            values = new_values

    return (values, optimal_actions)


def action_value(mdp, s, a, values):
    return sum([mdp.T[s][a][s1]*(
        mdp.R[s][a][s1]+mdp.gamma*values[s1]
        ) for s1 in range(mdp.S)])

def Howards_Policy_Iteration(mdp):
    optimal_actions = np.zeros(mdp.S, dtype=int)
    values = np.zeros(mdp.S)
    
    improvable_states = 1
    if mdp.mdptype == "continuing":
        while improvable_states > 0:
            improvable_states = 0
            for s in range(mdp.S):
                for a in range(mdp.A):
                    new_val = action_value(mdp, s, a, values)
                    if new_val > values[s]:
                        optimal_actions[s] = a
                        values[s] = new_val
                        improvable_states += 1
                        break
    else:
        while improvable_states > 0:
            improvable_states = 0
            for s in range(mdp.S):
                if s in mdp.end:
                    continue
                for a in range(mdp.A):
                    new_val = action_value(mdp, s, a, values)
                    if new_val > values[s]:
                        optimal_actions[s] = a
                        values[s] = new_val
                        improvable_states += 1
                        break

    return (values, optimal_actions)


def Linear_Programming(mdp):
    # PROBLEM
    prob = pulp.LpProblem("MDP", pulp.LpMinimize)
    values = [pulp.LpVariable("v{s}".format(s=s)) for s in range(mdp.S)]
    # actions = [pulp.LpVariable("a{}".format(s), lowBound=0, upBound=mdp.A-1, cat="Integer") for s in range(mdp.S)]
    
    # OBJECTIVE FXN
    prob += pulp.lpSum(values)

    # CONSTRAINTS
    for s in range(mdp.S):
        for a in range(mdp.A):
            prob += values[s] - sum([mdp.T[s][a][s1]*(mdp.R[s][a][s1] + mdp.gamma*values[s1]) for s1 in range(mdp.S)]) >= 0
 
    # SOLVE
    prob.solve(pulp.PULP_CBC_CMD(msg=False))

    # GET RESULTS
    final_values = [v.varValue for v in values]
    optimal_actions = [np.argmax([action_value(mdp, s, a, final_values) for a in range(mdp.A)]) for s in range(mdp.S)]
    return final_values, optimal_actions


if __name__ == "__main__":
    parser.add_argument("--mdp",type=str)
    parser.add_argument("--algorithm",type=str, default="lp")
    parser.add_argument("--policy",type=str,default=None)

    args = parser.parse_args()

    mdp = MDP(args.mdp)

    if args.policy != None:
        policy = Policy(args.policy)
        values, actions = Evaluate_Policy(mdp, policy)
    elif args.algorithm == "vi":
        values, actions = Value_Iteration(mdp)
    elif args.algorithm == "hpi":
        values, actions = Howards_Policy_Iteration(mdp)
    elif args.algorithm == "lp":
        values, actions = Linear_Programming(mdp)

    for s in range(mdp.S):
        print(values[s], actions[s])