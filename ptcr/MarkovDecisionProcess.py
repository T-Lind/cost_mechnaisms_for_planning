from typing import Dict, Union

import numpy as np


class MarkovDecisionProcess:
    def __init__(self, states: Dict[Union[int, str], str],
                 actions: Dict[Union[int, str], str],
                 transitions: Dict[str, float],
                 rewards: np.ndarray,
                 discount_factor: float):
        self.states = list(states.keys())
        self.state_labels = states
        self.actions = list(actions.keys())
        self.action_labels = actions
        self.transitions = transitions
        self.rewards = rewards
        self.discount_factor = discount_factor

    def build_transition_matrix(self, transitions: Dict[str, float]) -> np.ndarray:
        """
        Build the transition matrix from the transition labels and probabilities.

        Parameters:
        - transitions: Dictionary mapping transition labels to probabilities

        Returns:
        - Transition matrix
        """
        num_states = len(self.states)
        num_actions = len(self.actions)
        transition_matrix = np.zeros((num_states, num_actions, num_states))

        for transition_label, prob in transitions.items():
            current_state, action, next_state = transition_label.split("->")
            current_state_idx = self.states.index(current_state)
            action_idx = self.actions.index(action)
            next_state_idx = self.states.index(next_state)

            transition_matrix[current_state_idx, action_idx, next_state_idx] = prob

        return transition_matrix

    def get_transition_prob(self, current_state: Union[int, str], action: Union[int, str],
                            next_state: Union[int, str]) -> float:
        """
        Get the transition probability P(s' | s, a).

        Parameters:
        - current_state: Current state index or label
        - action: Chosen action index or label
        - next_state: Next state index or label

        Returns:
        - Transition probability
        """
        current_state_idx = self.states.index(current_state) if isinstance(current_state, str) else current_state
        action_idx = self.actions.index(action) if isinstance(action, str) else action
        next_state_idx = self.states.index(next_state) if isinstance(next_state, str) else next_state

        transition_label = f"{self.state_labels[current_state_idx]}->{self.action_labels[action_idx]}->{self.state_labels[next_state_idx]}"

        return float(self.transitions[transition_label])

    def get_reward(self, current_state: Union[int, str], action: Union[int, str]) -> float:
        """
        Get the immediate reward R(s, a).

        Parameters:
        - current_state: Current state index or label
        - action: Chosen action index or label

        Returns:
        - Immediate reward
        """
        current_state_idx = self.states.index(current_state) if isinstance(current_state, str) else current_state
        action_idx = self.actions.index(action) if isinstance(action, str) else action

        return float(self.rewards[current_state_idx, action_idx])

    def __str__(self) -> str:
        """
        String representation of the MDP.
        """
        result = "State Labels: {}\n".format(self.state_labels)
        result += "Action Labels: {}\n".format(self.action_labels)
        result += "Transition Probabilities:\n{}\n".format(self.transitions)
        result += "Immediate Rewards:\n{}\n".format(self.rewards)
        result += "Discount Factor (gamma): {}".format(self.discount_factor)
        return result
