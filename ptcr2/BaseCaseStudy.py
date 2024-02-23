from abc import ABC, abstractmethod

from ptcr2.EventPredictor import EventPredictor


class BaseCaseStudy(ABC):
    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep: EventPredictor = None
        self.mc = None
        self.alphabet_s = None

    @abstractmethod
    def make_event_predictor(self):
        pass

    @abstractmethod
    def get_dfa(self):
        pass

    def compute_optimal_policy(self):
        self.make_event_predictor()
        self.computed_policy = self.ep.optimal_policy_infinite_horizon(0.01)
        return self.computed_policy

    def simulate(self):
        if not self.computed_policy:
            self.compute_optimal_policy()
        policy, _, expected, policy_comp_time = self.computed_policy
        run_number_of_steps, recorded_story = self.ep.simulate(policy)
        return expected, run_number_of_steps, recorded_story, policy_comp_time

    def simulate_greedy_algorithm(self):
        if not self.ep:
            self.make_event_predictor()
        return self.ep.simulate_greedy_algorithm()

    def simulate_general_and_greedy_algorithms(self):
        if not self.computed_policy:
            self.compute_optimal_policy()
        policy, _, expected, policy_comp_time, d = self.computed_policy
        return self.ep.simulate_general_and_greedy_algorithms(policy)
