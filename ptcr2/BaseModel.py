from abc import ABC, abstractmethod

from ptcr2.EventPredictor import EventPredictor


class BaseModel(ABC):
    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep: EventPredictor = None
        self.mc = None
        self.alphabet_s = None

    @abstractmethod
    def make_event_predictor(self, spec: dict):
        pass

    @abstractmethod
    def get_dfa(self):
        pass

    def compute_optimal_policy(self, spec: dict):
        self.make_event_predictor(spec)
        self.computed_policy = self.ep.optimal_policy_infinite_horizon(0.01)
        return self.computed_policy

    def simulate(self, spec: dict):
        if not self.computed_policy:
            self.compute_optimal_policy(spec)
        policy, _, expected, policy_comp_time, diff_tracker = self.computed_policy
        run_number_of_steps, recorded_story = self.ep.simulate(policy)
        return {
            "expected": expected,
            "run_number_of_steps": run_number_of_steps,
            "recorded_story": recorded_story,
            "policy_comp_time": policy_comp_time,
            "diff_tracker": diff_tracker
        }

    def simulate_greedy_algorithm(self, spec: dict):
        if not self.ep:
            self.make_event_predictor(spec)
        return self.ep.simulate_greedy_algorithm()

    def simulate_general_and_greedy_algorithms(self, spec: dict):
        if not self.computed_policy:
            self.compute_optimal_policy(spec)
        policy, _, expected, policy_comp_time, d, diff_tracker = self.computed_policy
        return self.ep.simulate_general_and_greedy_algorithms(policy)
