from typing import Set, Dict, List

import numpy as np


class EventModel:
    def __init__(self,
                 states: List[str],
                 events: Set[str],
                 transitions: Dict[str, float],
                 event_function: Dict[str, Dict[str, float]],
                 start_state: str,
                 ):
        self.states = states
        self.events = events
        self.transitions = transitions
        self.transition_matrix = self.build_transition_matrix(transitions)
        self.start_state = start_state
        self.event_function = event_function

        self.current_state = start_state

    def build_transition_matrix(self, transitions: Dict[str, float]) -> np.ndarray:
        """
        Builds a transition matrix from a dictionary of transitions.
        :param transitions: A dictionary of transitions.
        :return: A transition matrix.
        """
        matrix = np.zeros((len(self.states), len(self.states)))
        for transition in transitions:
            source, destination = transition.split("->")
            matrix[int(source), int(destination)] = transitions[transition]
        return matrix

    def step(self):
        # identify the row of the transition matrix corresponding to the current state
        row = self.transition_matrix[int(self.current_state), :]
        # next, select a random index based on its probability, which is stored in the matrix
        # select a random state
        self.current_state = np.random.choice(self.states, p=row)

        # at this new state, select a random event
        event_probs = self.event_function[self.current_state]

        if len(event_probs) == 0:
            return self.current_state, None

        # event_probs is a Dict[str, float], so based on each value which is the corresponding value, select the key
        event = np.random.choice(list(event_probs.keys()), p=list(event_probs.values()))

        return self.current_state, event

