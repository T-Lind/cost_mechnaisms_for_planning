from ptcr import DeterministicFiniteAutomaton, MarkovDecisionProcessOld


class RtmEnvironment:
    def __init__(self, dfa: DeterministicFiniteAutomaton, mdp: MarkovDecisionProcessOld):
        self.dfa = dfa
        self.mdp = mdp


