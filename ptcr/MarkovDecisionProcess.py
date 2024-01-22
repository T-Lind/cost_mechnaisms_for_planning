from typing import Dict, Union

import numpy as np


class MarkovDecisionProcess:
    def __init__(self, states: Dict[Union[int, str], str],
                 actions: Dict[Union[int, str], str],
                 transitions: Dict[str, float],
                 rewards: Dict[str, float],
                 discount_factor: float):
        self.states = list(states.keys())
        self.state_labels = states
        self.actions = list(actions.keys())
        self.action_labels = actions
        self.transitions = transitions
        self.rewards = self.build_reward_matrix(rewards)
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

    def __find_key_by_value(self, dictionary: Dict[Union[int, str], str], value: str) -> Union[int, str]:
        """
        Find a key in a dictionary by its value.

        Parameters:
        - dictionary: Dictionary to search
        - value: Value to search for

        Returns:
        - Key corresponding to the value
        """
        for key, val in dictionary.items():
            if val == value:
                return key
        raise ValueError(f"Value {value} not found in dictionary")

    def build_reward_matrix(self, rewards: Dict[str, float]) -> np.ndarray:
        """
        Build the reward matrix from the reward labels and values.

        Parameters:
        - rewards: Dictionary mapping state-action labels to immediate rewards

        Returns:
        - Reward matrix
        """
        num_states = len(self.states)
        num_actions = len(self.actions)
        reward_matrix = np.zeros((num_states, num_actions))

        for reward_label, value in rewards.items():
            state, action = reward_label.split("->")
            state_idx = self.states.index(self.__find_key_by_value(self.state_labels, state))
            action_idx = self.actions.index(self.__find_key_by_value(self.action_labels, action))

            reward_matrix[state_idx, action_idx] = value

        return reward_matrix

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

        transition_result = self.transitions.get(transition_label, None)

        if transition_result is None:
            return 0.0
        else:
            return float(transition_result)

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

    def policy_evaluation(self,
                          policy: Dict[Union[int, str], Union[int, str]],
                          tol: float = 1e-6, max_iter: int = 1000) -> np.ndarray:
        """
        Perform policy evaluation to estimate the state values for a given policy.

        Parameters:
        - policy: Dictionary mapping state to action
        - tol: Tolerance for convergence
        - max_iter: Maximum number of iterations

        Returns:
        - Estimated state values
        """
        num_states = len(self.states)
        V = np.zeros(num_states)

        for _ in range(max_iter):
            delta = 0
            for s in self.states:
                v = V[s]
                action = self.__find_key_by_value(self.action_labels, policy[self.state_labels[s]])
                expected_return = 0
                for s_prime in self.states:
                    expected_return += self.get_transition_prob(s, action, s_prime) * (
                            self.get_reward(s, action) + self.discount_factor * V[s_prime])
                    V[s] = expected_return
                    delta = max(delta, abs(v - V[s]))

            if delta < tol:
                break

        return V

    def policy_improvement(self, policy: Dict[Union[int, str], Union[int, str]]) -> Dict[Union[int, str], Union[int, str]]:
        """
        Perform policy improvement based on the estimated state values.

        Parameters:
        - policy: Current policy

        Returns:
        - Improved policy
        """

        new_policy = policy.copy()

        for s in self.states:
            best_action = None
            best_value = float('-inf')

            for a in self.actions:
                expected_return = 0
                for s_prime in self.states:
                    expected_return += self.get_transition_prob(s, a, s_prime) * (
                            self.get_reward(s, a) + self.discount_factor * self.get_transition_prob(s, a, s_prime))

                if expected_return > best_value:
                    best_value = expected_return
                    best_action = a

            new_policy[self.state_labels[s]] = self.action_labels[best_action]

        return new_policy

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
