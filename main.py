import random

random.seed(0)
from ptcr import MarkovDecisionProcess
import matplotlib.pyplot as plt
from colorama import Fore as Fr, Style as St

# Example usage with dictionaries for states, actions, transitions, and rewards:
with open("samples/oulu/mdp.json", "r") as f:
    text_json = f.read()
mdp = MarkovDecisionProcess.load_model(text_json)
print(mdp)
policy = {}
for state in mdp.state_labels.values():
    policy[state] = random.choice(list(mdp.action_labels.values()))

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
