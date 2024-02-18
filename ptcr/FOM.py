import json

from ptcr import DeterministicFiniteAutomaton
from ptcr.EventPredictor import EventPredictor
from ptcr.MarkovChain import MarkovChain


class FOM:
    def __init__(self, input_str: str):
        self.computed_policy_with_metadata = None
        self.ep = None
        self.ep = None

        # convert to dict
        self.input_dict = json.loads(input_str)

        self.state_names = list(self.input_dict['transitions'].keys())

        # load state events into this form:
        self.state_events = [[] for i in range(len(self.state_names))]
        for state, events in self.input_dict['events'].items():
            index = self.state_names.index(state)
            self.state_events[index] = events
        self.transition_matrix = [[0.0 for i in range(len(self.state_names))] for j in range(len(self.state_names))]
        for state, transitions in self.input_dict['transitions'].items():
            row_index = self.state_names.index(state)
            for event, next_state in transitions.items():
                column_index = self.state_names.index(event)
                self.transition_matrix[row_index][column_index] = next_state

        self.initial_distribution = self.input_dict['initial_distribution']

        self.evidence_distribution = None
        if 'identity' in self.input_dict['evidence_distribution']:
            self.evidence_distribution = self.identity_matrix(len(self.state_names))

        self.alphabet = set(self.input_dict['alphabet'])

        self.dfa_states = set(self.input_dict['dfa'].keys())
        self.initial_dfa_state = self.input_dict['initial_dfa_state']
        self.final_dfa_states = set(self.input_dict['final_dfa_states'])

        self.dfa_transitions = {}
        # the tuple is ((from_state, to_state), event)
        for from_state, transitions in self.input_dict['dfa'].items():
            for event, to_state in transitions.items():
                self.dfa_transitions[(from_state, event)] = to_state

        self.markov_chain = MarkovChain(self.state_names, self.state_events, self.transition_matrix,
                                        self.initial_distribution, self.evidence_distribution)
        self.dfa = DeterministicFiniteAutomaton(self.dfa_states, self.alphabet, self.dfa_transitions,
                                                self.initial_dfa_state, self.final_dfa_states)

        self.event_predictor = EventPredictor(self.dfa, self.markov_chain, self.alphabet)

        print("state names\n", self.state_names)
        print("state events\n", self.state_events)
        print("transition matrix\n", self.transition_matrix)
        print("initial distribution\n", self.initial_distribution)
        print("evidence distribution\n", self.evidence_distribution)
        print("alphabet\n", self.alphabet)
        print("dfa states\n", self.dfa_states)
        print("initial dfa state\n", self.initial_dfa_state)
        print("final dfa states\n", self.final_dfa_states)
        print("dfa transitions\n", self.dfa_transitions)

    def identity_matrix(self, size: int):
        return [[1 if i == j else 0 for i in range(size)] for j in range(size)]

    def simulate(self):
        if not self.computed_policy_with_metadata:
            self.compute_optimal_policy()

        n_iters, story = self.event_predictor.simulate(self.computed_policy_with_metadata['policy'])

        return {
            "n_iters": n_iters,
            "story": story,
            "expected_cost": self.computed_policy_with_metadata['expected_cost'],
            "time_elapsed": self.computed_policy_with_metadata['time_elapsed']
        }

    def compute_optimal_policy(self):
        response = self.event_predictor.optimal_policy_infinite_horizon(0.001)
        self.computed_policy_with_metadata = {
            "policy": response[0],
            "expected_cost": response[1],
            "time_elapsed": response[2]
        }
