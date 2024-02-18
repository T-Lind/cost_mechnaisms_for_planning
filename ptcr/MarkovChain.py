import numpy as np


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
    def __init__(self, state_names: list, state_events: list, transition_matrix: list, initial_distribution: list,
                 evidence_distribution: list, initial_state_index: int = 0, evidence_list: list = None):
        if evidence_list is None:
            self.evidence_list = []
        else:
            self.evidence_list = evidence_list

        self.states = []
        self.state_names = state_names  # AKA evidence_list?
        self.initial_state_index = initial_state_index

        for i in range(len(state_names)):
            self.states.append(MarkovState(self.state_names[i], state_events[i], len(self.states)))

        self.initial_distribution = initial_distribution

        self.transition_matrix = transition_matrix

        self.events = set()
        for state in self.states:
            for e in state.events:
                if not e in self.events:
                    self.events.add(e)

        self.initial_state = self.states[self.initial_state_index]
        self.null_state = MarkovState("none", [], -1)

        self.has_evidence = False

        for row in range(len(self.transition_matrix)):
            self.states[row].evidence_distribution = evidence_distribution[row]

    def get_transition_probability(self, srcState: MarkovState, dstState: MarkovState):
        return self.transition_matrix[srcState.index][dstState.index]

    def next_state(self, currentState):
        if currentState == self.null_state:
            return np.random.choice(self.states, p=self.initial_distribution)

        return np.random.choice(self.states, p=self.transition_matrix[currentState.index])
