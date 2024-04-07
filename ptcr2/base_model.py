import os
import pickle
import sys
import time
from abc import ABC, abstractmethod

# Increase the recursion limit to avoid RecursionError while loading models
sys.setrecursionlimit(10_000)


class BaseModel(ABC):
    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep = None
        self.mc = None
        self.alphabet_s = None
        self.cost_matrix = None
        self.epsilon = 0.01

    @abstractmethod
    def make_event_predictor(self, spec: dict):
        pass

    @abstractmethod
    def get_dfa(self):
        pass

    def compute_optimal_policy(self, spec: dict, cost_based=False):
        self.make_event_predictor(spec)
        if not cost_based and not spec.get('cost_based', False):
            self.computed_policy = self.ep.optimal_policy_infinite_horizon(epsilon_of_convergence=self.epsilon)
        else:
            print("COMPUTING COST BASED POLICY!!!")
            self.computed_policy = self.ep.optimal_policy_infinite_horizon_cost_based(
                epsilon_of_convergence=self.epsilon)
        return self.computed_policy

    def simulate(self, spec: dict):
        if not self.computed_policy or not self.ep:
            self.compute_optimal_policy(spec)

        run_number_of_steps, recorded_story = self.ep.simulate(self.computed_policy['optimal_policy'])

        return {
            "expected": self.computed_policy['expected'],
            "run_number_of_steps": run_number_of_steps,
            "recorded_story": recorded_story,
            "policy_comp_time": self.computed_policy['elapsed_time'],
            "diff_tracker": self.computed_policy['diff_tracker']
        }

    def simulate_greedy_algorithm(self, spec: dict):
        if not self.ep:
            self.make_event_predictor(spec)
        return self.ep.simulate_greedy_algorithm()

    def simulate_general_and_greedy_algorithms(self, spec: dict = None):
        if not self.computed_policy:
            if not spec:
                raise ValueError("Specification is required to compute optimal policy")
            self.compute_optimal_policy(spec)

        policy = self.computed_policy['optimal_policy']
        return self.ep.simulate_general_and_greedy_algorithms(policy)

    def save(self, filename=None):
        """
        Save the model to a file, based on the current file name and timestamp
        :param filename: The file name to use instead of the auto-generated one. Optional.
        :return: The file name that was used to save the model, either the one provided or the auto-generated one.
        """
        if filename is None:
            current_time_str = time.strftime("%Y%m%d-%H%M%S")

            if not os.path.exists("saves"):
                os.makedirs("saves")

            filename = f"saves/ptcr_model_{current_time_str}.pkl"

        with open(filename, "wb") as file:
            pickle.dump(self, file)

        return filename

    @staticmethod
    def load(filename):
        """
        Static method used to load a BaseModel (i.e. FOM/POM) from a file
        :param filename: File name to load the model from
        :return: The loaded model object
        """
        with open(filename, "rb") as file:
            return pickle.load(file)
