from ptcr import DeterministicFiniteAutomaton, MarkovDecisionProcess


class RtmEnvironment:
    def __init__(self, dfa: DeterministicFiniteAutomaton, mdp: MarkovDecisionProcess):
        self.dfa = dfa
        self.mdp = mdp


