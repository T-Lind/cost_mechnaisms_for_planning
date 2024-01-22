import random

random.seed(0)
from ptcr import MarkovDecisionProcess
import matplotlib.pyplot as plt
from colorama import Fore as Fr, Style as St

# Example usage with dictionaries for states, actions, transitions, and rewards:
state_labels = {0: 'A', 1: 'B', 2: 'C'}
action_labels = {0: 'X', 1: 'Y', 2: 'Z'}
transitions = {
    "A->X->A": 0.2,
    "A->X->B": 0.8,
    "A->Y->C": 0.1,
    "A->Y->A": 0.9,
    "A->Z->B": 1.0,
    "B->X->A": 0.3,
    "B->X->B": 0.7,
    "B->Y->C": 0.4,
    "B->Y->A": 0.6,
    "B->Z->B": 1.0,
    "C->X->A": 0.5,
    "C->X->B": 0.5,
    "C->Y->C": 0.6,
    "C->Y->A": 0.4,
    "C->Z->B": 1.0
}
rewards = {"A->X": 1, "A->Y": -1, "A->Z": 0, "B->X": 0, "B->Y": 1, "B->Z": -1, "C->X": -1, "C->Y": 0, "C->Z": 5}
discount_factor = 0.9

mdp = MarkovDecisionProcess(state_labels, action_labels, transitions, rewards, discount_factor)
print(mdp)
policy = {}
for state in state_labels.values():
    policy[state] = random.choice(list(action_labels.values()))

print(Fr.RED + St.BRIGHT + "Initial policy", policy, St.RESET_ALL)
print(Fr.YELLOW + "Initial policy evaluation", mdp.policy_evaluation(policy), St.RESET_ALL + Fr.MAGENTA)
mdp.simulate_policy(policy, 10)
print("Before-training average reward: {}".format(mdp.get_policy_average(policy)))

solved_policy, sums = mdp.policy_iteration(policy)
print(St.RESET_ALL + Fr.RED + St.BRIGHT + "Solved policy", solved_policy, St.RESET_ALL)
print(Fr.YELLOW + "Solved policy evaluation", mdp.policy_evaluation(solved_policy), St.RESET_ALL + Fr.MAGENTA)
mdp.simulate_policy(solved_policy, 10)
print("After-training average reward: {}".format(mdp.get_policy_average(solved_policy)))

plt.plot(sums)
plt.xlabel("Iteration")
plt.ylabel("Sum of state values")
plt.title("Policy iteration")
plt.show()
