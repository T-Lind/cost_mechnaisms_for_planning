import numpy as np

from ptcr import MarkovDecisionProcess

# Example usage with dictionaries for states, actions, and transitions:
state_labels = {0: 'A', 1: 'B', 2: 'C'}
action_labels = {0: 'X', 1: 'Y'}
transitions = {"A->X->B": 0.4, "A->Y->C": 0.6, "B->X->A": 0.1, "B->Y->C": 0.9, "C->X->A": 0.2, "C->Y->B": 0.8}
rewards = np.array([[1, -1], [2, 0], [0, 5]])
discount_factor = 0.9

mdp = MarkovDecisionProcess(state_labels, action_labels, transitions, rewards, discount_factor)
print(mdp)
