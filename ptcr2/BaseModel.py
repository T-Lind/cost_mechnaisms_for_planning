from abc import ABC, abstractmethod

from ptcr2.EventPredictor import EventPredictor


class BaseModel(ABC):
    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep: EventPredictor = None
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

    def compute_optimal_policy(self, spec: dict):
        self.make_event_predictor(spec)
        self.computed_policy = self.ep.optimal_policy_infinite_horizon(epsilon_of_convergence=self.epsilon)
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
            self.compute_optimal_policy(spec)

        policy = self.computed_policy['optimal_policy']
        return self.ep.simulate_general_and_greedy_algorithms(policy)
