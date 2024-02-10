import random

class MarkovState:
    def __init__(self, name: str, events: list, index: int, evidence_distribution=None):
        if events is None:
            events = []
        if evidence_distribution is None:
            evidence_distribution = {}
        self.name = name
        self.events = events
        self.evidence_distribution = evidence_distribution
        self.index = index

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name


class MarkovChain:
    def __init__(self, state_names: list, state_events: list, transition_matrix: list, initial_distribution: list, initial_state_index: int=0):
        self.states = []
        self.state_names = state_names
        self.initial_state_index = initial_state_index

        for i in range(len(state_names)):
            self.states.append(MarkovState(self.state_names[i], state_events[i], len(self.states)))

        self.initial_distribution = initial_distribution


