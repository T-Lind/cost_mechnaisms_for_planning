from ptcr import MarkovDecisionProcess

# Example usage with dictionaries for states, actions, transitions, and rewards:
state_labels = {0: 'A', 1: 'B', 2: 'C'}
action_labels = {0: 'X', 1: 'Y'}
transitions = {
    "A->X->B": 0.4,
    "A->Y->C": 0.6,
    "B->X->A": 0.1,
    "B->Y->C": 0.9,
    "C->X->A": 0.2,
    "C->Y->B": 0.8,
    "A->Y->A": 0.8,
    'A->X->A': 0.9,
    'A->X->C': 0.1
}
rewards = {"A->X": 1, "A->Y": -1, "B->X": 2, "B->Y": 0, "C->X": 0, "C->Y": 5}
discount_factor = 0.9

mdp = MarkovDecisionProcess(state_labels, action_labels, transitions, rewards, discount_factor)
print(mdp)
policy = {'A': 'X', 'B': 'Y', 'C': 'X'}
print("Initial policy", policy)
print("Initial policy evaluation", mdp.policy_evaluation(policy))
policy_improv = mdp.policy_improvement(policy)
print("Iterated policy", policy_improv)
print("Iterated policy evaluation", mdp.policy_evaluation(policy_improv))
