from random import choice
from typing import Dict, Union, Any, List

import numpy as np


@DeprecationWarning
class MarkovDecisionProcessOld:
    def __init__(self, states: List[str],
                 actions: List[str],
                 transitions: Dict[str, float],
                 rewards: Dict[str, float],
                 discount_factor: float):
        # create a dict where each state is mapped to an index starting at 0 (same for actions)
        states = {i: state for i, state in enumerate(states)}
        actions = {i: action for i, action in enumerate(actions)}

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
        raise ValueError(f"Value {value} not found in dictionary {dictionary}")

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

    def policy_improvement(self, policy: Dict[Union[int, str], Union[int, str]]) -> Dict[
        Union[int, str], Union[int, str]]:
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

    def policy_iteration(self, policy: Dict[Union[int, str], Union[int, str]] = None, tol: float = 1e-6,
                         max_iter: int = 1000, return_eval_sums=True) -> tuple[dict[int | str, int | str], list[Any]] | \
                                                                         dict[int | str, int | str]:
        """
        Perform policy iteration to find the optimal policy.

        Parameters:
        - tol: Tolerance for convergence
        - max_iter: Maximum number of iterations

        Returns:
        - Optimal policy
        """
        if not policy:
            policy = {}
            for state in self.state_labels.values():
                policy[state] = choice(list(self.action_labels.values()))

        old_policy = policy.copy()
        old_evaluation = self.policy_evaluation(old_policy)

        iters = 0

        eval_sums = []
        eval_sums.append(np.sum(old_evaluation))

        while iters < max_iter:
            policy = self.policy_improvement(old_policy)
            evaluation = self.policy_evaluation(policy)

            eval_sums.append(np.sum(evaluation))

            if np.allclose(evaluation, old_evaluation, atol=tol):
                break

            old_policy = policy
            old_evaluation = evaluation

            iters += 1

        if return_eval_sums:
            return policy, eval_sums
        return policy

    def simulate_policy(self, policy: Dict[Union[int, str], Union[int, str]], num_steps: int = 10,
                        verbose=True) -> tuple[list[str], list[float]]:
        """
        Simulate the agent's behavior based on the given policy and print the states visited.

        Parameters:
        - policy: Dictionary mapping state to action
        - num_steps: Number of steps to simulate
        """
        current_state = np.random.choice(self.states)

        if verbose:
            print(f"Initial state: {self.state_labels[current_state]}")

        state_tracker = []
        cumulative_reward = []

        for _ in range(num_steps):
            state = self.state_labels[current_state]
            state_tracker.append(state)

            action = policy[state]
            numerical_action = self.__find_key_by_value(self.action_labels, action)

            # Transition to the next state based on the chosen action
            next_state_probs = [
                self.get_transition_prob(current_state, numerical_action, next_state)
                for next_state in self.states
            ]
            next_state = np.random.choice(self.states, p=next_state_probs)
            # get the probability that that state was selected
            prob = next_state_probs[self.states.index(next_state)]

            reward = self.get_reward(current_state, numerical_action)
            cumulative_reward.append(reward + cumulative_reward[-1] if len(cumulative_reward) > 0 else reward)

            if verbose:
                print(
                    f"State: {state}, Action: {action}, Reward: {reward} | Transition prob: {prob:.2f}")

            current_state = next_state

        return state_tracker, cumulative_reward

    def get_policy_average(self,
                           policy: Dict[Union[int, str], Union[int, str]],
                           num_iterations: int = 1000,
                           steps_per_iteration: int = 25) -> float:
        """
        Get the average reward for a given policy.

        Parameters:
        - policy: Dictionary mapping state to action
        - num_iterations: Number of iterations to average over
        - steps_per_iteration: Number of steps to simulate per iteration

        Returns:
        - Average reward
        """

        rewards = []
        for _ in range(num_iterations):
            _, cumulative_reward = self.simulate_policy(policy, steps_per_iteration, verbose=False)
            rewards.append(cumulative_reward[-1])

        return np.mean(rewards)

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

    @classmethod
    def load_model(cls, model: str):
        """
        Load a model from a JSON string.

        Parameters:
        - model: JSON string representing the model

        Returns:
        - MarkovDecisionProcessOld object
        """
        import json
        model = json.loads(model)
        return cls(model['state_labels'], model['action_labels'], model['transitions'], model['rewards'],
                   model['discount_factor'])


class MDPState:
    def __init__(self, name: str, anchor=None):
        self.name = name
        self.anchor = anchor
        self.is_initial = False
        self.is_goal = False
        self.index = -1
        self.transitions = []
        self.reachable = True
        self.evidence_distribution = []
        self.sub_probability_transitions = 0
        self.available_actions = []
        self.action_transitions = {}

        self.scc_index = -1
        self.lowlink = -1

        self.scc = None

        self.visited = False

        self.avoid_actions = []

        self.a_goal_is_reachable = False

        self.transitions_to = []
        self.weight_b_vector = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MDP:
    def __init__(self):
        self.states: list[MDPState] = []
        self.actions = []
        self.initial_state = MDPState("")
        self.goal_states = []
        self.transitions = []
        self.transitions_dict = {}

        self.made_transition_dict = False
        self.has_evidence = False

        self.evidence_list = []
        self.observations = []

        self.observation_function = {}
        self.belief_tree = None

        self.strong_connected_components = []
        self.initial_scc = None
        self.scc_transitions = []

        self.states_dict_by_name = {}

    def compute_avoidable_actions(self):
        for state in self.states:
            state.a_goal_is_reachable = False

        queue = []
        for state in self.states:
            if state.is_goal:
                queue.append(state)
                state.a_goal_is_reachable = True

        while queue:
            state = queue.pop(0)
            for transition in state.transitions_to:
                all_reach_to_goals = True
                for transition_2 in transition.src_state.actions_transitions[transition.action]:
                    if not transition_2.dst_state.a_goal_is_reachable:
                        all_reach_to_goals = False
                        break
                if all_reach_to_goals and not transition.src_state.a_goal_is_reachable:
                    transition.src_state.a_goal_is_reachable = True
                    queue.append(transition.src_state)

        for state in self.states:
            for action in self.actions:
                all_dst_reachable = True

                if not action not in state.action_transitions.keys():
                    for transition_3 in state.action_transitions[action]:
                        if not transition_3.dst_state.a_goal_is_reachable:
                            all_dst_reachable = False
                            break

                if not all_dst_reachable:
                    state.avoid_actions.append(action)

    def add_state(self, state: MDPState):
        state.index = len(self.states)
        self.states.append(state)
        self.states_dict_by_name[state.name] = state
