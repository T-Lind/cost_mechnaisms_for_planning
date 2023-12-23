'''
Created on Aug 14, 2019

@author: hazhar
'''

from automata.fa.dfa import DFA

from planner.Markov.MarkovChain import MarkovChain
from planner.Narration.EventPredictor import EventPredictor
from planner.case_studies.case_study import CaseStudy

"""
This class is used to implement the case study of Touring Paris
"""


class ParisFOM(CaseStudy):

    def __init__(self):
        self.verbose = False
        self.copmutedPolicy = None
        self.ep = None

    def make_event_predictor(self):
        state_names = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]

        state_events = [set(), set(), set(["h"]), set(["k"]), set(), set(), set(["t"]), set(["c"]), set()]
        transition_matrix = [[0.0, 1, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.25, 0.25, 0.25, 0.0, 0.0, 0.0, 0.0, 0.25],
                             [0.0, 0.25, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.25, 0.0, 0.25, 0.5, 0.0, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.25, 0.25, 0.5, 0.0, 0.0, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.2, 0.3, 0.0],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.25, 0.0, 0.5],
                             [0.0, 0.0, 0.0, 0.0, 0.0, 0.25, 0.0, 0.25, 0.5],
                             [0.25, 0.0, 0.0, 0.0, 0.0, 0.0, 0.2, 0.3, 0.25]]

        initial_distribution = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.markov_chain = MarkovChain(state_names, state_events, transition_matrix, initial_distribution, 0)
        self.markov_chain.has_evidence = True
        self.markov_chain.evidence_list = ["s0", "s1", "s2", "s3", "s4", "s5", "s6", "s7", "s8"]
        self.markov_chain.states[0].evidence_distribution = [1, 0, 0, 0, 0, 0, 0, 0, 0]
        self.markov_chain.states[1].evidence_distribution = [0, 1, 0, 0, 0, 0, 0, 0, 0]
        self.markov_chain.states[2].evidence_distribution = [0, 0, 1, 0, 0, 0, 0, 0, 0]
        self.markov_chain.states[3].evidence_distribution = [0, 0, 0, 1, 0, 0, 0, 0, 0]
        self.markov_chain.states[4].evidence_distribution = [0, 0, 0, 0, 1, 0, 0, 0, 0]
        self.markov_chain.states[5].evidence_distribution = [0, 0, 0, 0, 0, 1, 0, 0, 0]
        self.markov_chain.states[6].evidence_distribution = [0, 0, 0, 0, 0, 0, 1, 0, 0]
        self.markov_chain.states[7].evidence_distribution = [0, 0, 0, 0, 0, 0, 0, 1, 0]
        self.markov_chain.states[8].evidence_distribution = [0, 0, 0, 0, 0, 0, 0, 0, 1]

        self.alphabet_set = {"h", "k", "t", "c"}
        self.alphabet_list = ["h", "k", "t", "c"]
        states = ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]

        transitions = {}
        transitions["q0"] = {}
        transitions["q0"]["h"] = "q1"
        transitions["q0"]["k"] = "q2"
        transitions["q0"]["c"] = "q3"
        transitions["q0"]["t"] = "q3"

        transitions["q1"] = {}
        transitions["q1"]["h"] = "q1"
        transitions["q1"]["k"] = "q4"
        transitions["q1"]["c"] = "q5"
        transitions["q1"]["t"] = "q5"

        transitions["q2"] = {}
        transitions["q2"]["h"] = "q4"
        transitions["q2"]["k"] = "q2"
        transitions["q2"]["c"] = "q6"
        transitions["q2"]["t"] = "q6"

        transitions["q3"] = {}
        transitions["q3"]["h"] = "q5"
        transitions["q3"]["k"] = "q6"
        transitions["q3"]["c"] = "q3"
        transitions["q3"]["t"] = "q3"

        transitions["q4"] = {}
        transitions["q4"]["h"] = "q4"
        transitions["q4"]["k"] = "q4"
        transitions["q4"]["c"] = "q7"
        transitions["q4"]["t"] = "q7"

        transitions["q5"] = {}
        transitions["q5"]["h"] = "q5"
        transitions["q5"]["k"] = "q7"
        transitions["q5"]["c"] = "q5"
        transitions["q5"]["t"] = "q5"

        transitions["q6"] = {}
        transitions["q6"]["h"] = "q7"
        transitions["q6"]["k"] = "q6"
        transitions["q6"]["c"] = "q6"
        transitions["q6"]["t"] = "q6"

        transitions["q7"] = {}
        transitions["q7"]["h"] = "q7"
        transitions["q7"]["k"] = "q7"
        transitions["q7"]["c"] = "q7"
        transitions["q7"]["t"] = "q7"

        initial_state = "q0"

        final_states = set()
        final_states.add("q7")

        dfa = DFA(states=set(states), input_symbols=self.alphabet_set, transitions=transitions,
                  initial_state=initial_state, final_states=final_states)

        self.ep = EventPredictor(dfa, self.markov_chain, self.alphabet_set, self.verbose)

    def compute_optimal_policy(self):
        self.make_event_predictor()
        self.copmutedPolicy = self.ep.optimalPolicyInfiniteHorizon(0.001, True)
        return self.copmutedPolicy

    def make_pomdpx_file(self, filePath):
        if self.ep is None:
            self.make_event_predictor()
        self.ep.mdp.write_POMDPX_XML(filePath)
        print("POMDPX file created in path '" + filePath + "'")

    def simulate(self, show_results=True):
        if self.copmutedPolicy is None:
            self.compute_optimal_policy()
        tple = self.copmutedPolicy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate(policy, True)
        avg = tple2[0]
        if show_results:
            print("expc=" + str(expc) + ", avg=" + str(avg) + ", recorded=" + tple2[1])
        # return expc, avg, tple2[1], tple[3]  # changed to this
        return {"expected": expc, "average": avg, "story": tple2[1], "computation_time": tple[3]}

    def simulate_greedy_algorithm(self, show_results=True):
        if self.ep is None:
            self.make_event_predictor()
        tple2 = self.ep.simulate_greedy_algorithm(False)
        avg = tple2[0]
        if show_results:
            # print("expc="+str(expc)+", avg="+str(avg)+", recorded="+tple2[1])
            print("avg=" + str(avg) + ", recorded=" + tple2[1])
        return avg, tple2[1]

    def simulate_general_and_greedy_algorithms(self, show_results=True):
        if self.copmutedPolicy == None:
            self.compute_optimal_policy()
        tple = self.copmutedPolicy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate_general_and_greedy_algorithms(policy, False)
        avg = tple2[0]
        avg2 = tple2[2]
        if show_results:
            print("expc=" + str(expc) + ", avgGeneralAlg=" + str(avg) + ", recordedGeneralAlg=" + tple2[
                1] + ", avgGreedyAlg=" + str(avg2) + ", recordedGreedyAlg=" + tple2[3])
        return expc, avg, tple2[1], avg2, tple2[3], tple[3]

    def simulate_invisible_markov_state_case_gel_or_gen(self, showResults):
        states = ["q0", "q1", "q2", "q3", "q4", "q5", "q6", "q7"]

        transitions = {}
        transitions["q0"] = {}
        transitions["q0"]["h"] = "q1"
        transitions["q0"]["k"] = "q2"
        transitions["q0"]["c"] = "q3"
        transitions["q0"]["t"] = "q3"

        transitions["q1"] = {}
        transitions["q1"]["h"] = "q1"
        transitions["q1"]["k"] = "q4"
        transitions["q1"]["c"] = "q5"
        transitions["q1"]["t"] = "q5"

        transitions["q2"] = {}
        transitions["q2"]["h"] = "q4"
        transitions["q2"]["k"] = "q2"
        transitions["q2"]["c"] = "q6"
        transitions["q2"]["t"] = "q6"

        transitions["q3"] = {}
        transitions["q3"]["h"] = "q5"
        transitions["q3"]["k"] = "q6"
        transitions["q3"]["c"] = "q3"
        transitions["q3"]["t"] = "q3"

        transitions["q4"] = {}
        transitions["q4"]["h"] = "q4"
        transitions["q4"]["k"] = "q4"
        transitions["q4"]["c"] = "q7"
        transitions["q4"]["t"] = "q7"

        transitions["q5"] = {}
        transitions["q5"]["h"] = "q5"
        transitions["q5"]["k"] = "q7"
        transitions["q5"]["c"] = "q5"
        transitions["q5"]["t"] = "q5"

        transitions["q6"] = {}
        transitions["q6"]["h"] = "q7"
        transitions["q6"]["k"] = "q6"
        transitions["q6"]["c"] = "q6"
        transitions["q6"]["t"] = "q6"

        transitions["q7"] = {}
        transitions["q7"]["h"] = "q7"
        transitions["q7"]["k"] = "q7"
        transitions["q7"]["c"] = "q7"
        transitions["q7"]["t"] = "q7"

        initial_state = "q0"

        final_states = set()
        final_states.add("q7")

        dfa = DFA(states=set(states), input_symbols=self.alphabet_set, transitions=transitions,
                  initial_state=initial_state, final_states=final_states)

        ep = EventPredictor(dfa, self.markov_chain, self.alphabet_set, self.verbose, False)
        tple = ep.optimalPolicyInfiniteHorizon(0.01, True)
        policy = tple[0]
        expc = tple[2]
        tple2 = ep.simulate(policy, True)
        avg = tple2[0]
        if showResults:
            print("expc=" + str(expc) + ", avg=" + str(avg) + ", recorded=" + tple2[1])
        return expc, avg, tple2[1]
