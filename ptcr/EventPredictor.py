from ptcr import DeterministicFiniteAutomaton


class EventPredictor:
    def __init__(self, dfa: DeterministicFiniteAutomaton, markov_chain, event_list: list):
        self.dfa = dfa
        self.markov_chain = markov_chain
        self.event_list = event_list

        self.mdp = None
        self.current_markov_state_visible = True

