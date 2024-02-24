from copy import deepcopy

import ptcr2.AutomataUtility as AutomataUtility
from ptcr2.BaseCaseStudy import BaseCaseStudy
from ptcr2.EventPredictor import EventPredictor
from ptcr2.MarkovChain import MarkovChain


class FOM(BaseCaseStudy):
    def __init__(self):
        super().__init__()
        self.verbose = True
        self.computed_policy = None
        self.ep = None

    def make_event_predictor(self, spec: dict):
        # state_names1 = ["I", "E", "B", "C", "D", "S"]
        # state_events1 = [set([]), set(["e1"]), set(["b1"]), set(["c1"]), set(["d1"]), set(["s1"])]
        # transition_matrix = [
        #     [0, 0.1, 0.3, 0.1, 0.2, 0.3],
        #     [0, 0.1, 0.2, 0.1, 0.3, 0.3],
        #     [0, 0.2, 0.1, 0.1, 0.3, 0.3],
        #     [0, 0.1, 0.2, 0.2, 0.3, 0.2],
        #     [0, 0.2, 0.3, 0.1, 0.1, 0.3],
        #     [0, 0.4, 0.2, 0.2, 0.2, 0.0]]
        # initial_distribution = [0, 0.1, 0.3, 0.2, 0.2, 0.2]
        # mc1 = MarkovChain(state_names1, state_events1, transition_matrix, initial_distribution, 0)
        #
        # state_names2 = ["I", "E", "B", "C", "D", "S"]
        # state_events2 = [set([]), set(["e2"]), set(["b2"]), set(["c2"]), set(["d2"]), set(["s2"])]
        # mc2 = MarkovChain(state_names2, state_events2, transition_matrix, initial_distribution, 0)
        #
        # stateNames3 = ["I", "E", "B", "C", "D", "S"]
        # state_events3 = [set([]), set(["e3"]), set(["b3"]), set(["c3"]), set(["d3"]), set(["s3"])]
        # mc3 = MarkovChain(stateNames3, state_events3, transition_matrix, initial_distribution, 0)
        #
        # mc12 = mc1.product_singleInitialState(mc2, [[("d1", "d2"), "d12"]])
        #
        # mc = mc12.product_singleInitialState(mc3, [[("d2, d3"), "d23"]])
        #
        # self.alphabet_s = {"e1", "b1", "c1", "d1", "s1", "e2", "b2", "c2", "d2", "s2", "d12", "e3", "b3", "c3", "d3",
        #                    "s3", "d23"}
        #
        #
        # dfa = self.get_dfa()
        #
        # self.ep = EventPredictor(dfa, mc, self.alphabet_s, self.verbose)
        #
        # return self.ep

        state_names = spec["state_names"]
        state_events_1, state_events_2, state_events_3 = spec["state_events"]

        for i in range(len(state_events_1)):
            state_events_1[i] = set(state_events_1[i])
            state_events_2[i] = set(state_events_2[i])
            state_events_3[i] = set(state_events_3[i])

        transition_matrix = spec["transition_matrix"]
        initial_distribution = spec["initial_distribution"]

        print("transition matrix:", transition_matrix)
        print("initial distribution:", initial_distribution)

        mc1 = MarkovChain(state_names, state_events_1, transition_matrix, initial_distribution, 0)

        state_names2 = deepcopy(state_names)

        mc2 = MarkovChain(state_names2, state_events_2, transition_matrix, initial_distribution, 0)

        state_names3 = deepcopy(state_names)

        mc3 = MarkovChain(state_names3, state_events_3, transition_matrix, initial_distribution, 0)


        single_initial_state_0 = spec['single_initial_states'][0]
        single_initial_state_1 = spec['single_initial_states'][1]

        single_initial_state_0[0][0] = tuple(single_initial_state_0[0][0])
        single_initial_state_1[0][0] = tuple(single_initial_state_1[0][0])

        print("single initial state 0:", single_initial_state_0)
        print("single initial state 1:", single_initial_state_1)

        mc12 = mc1.product_singleInitialState(mc2, single_initial_state_0)

        mc = mc12.product_singleInitialState(mc3, single_initial_state_1)

        self.alphabet_s = set(spec['alphabet'])

        print("alphabet:", self.alphabet_s)

        dfa = self.get_dfa()

        self.ep = EventPredictor(dfa, mc, self.alphabet_s, self.verbose)

        return self.ep



    def get_dfa(self):
        dfa111 = AutomataUtility.dfa_accepting_a_sequence(["c3"], self.alphabet_s)
        dfa112 = AutomataUtility.dfa_accepting_a_sequence(["s3"], self.alphabet_s)
        dfa11 = AutomataUtility.union(dfa111, dfa112)

        dfa11 = AutomataUtility.new_state_names(dfa11)
        dfa11 = AutomataUtility.closure_plus(dfa11)
        dfa11 = dfa11.minify()
        dfa11 = AutomataUtility.new_state_names(dfa11)

        dfa12 = AutomataUtility.dfa_accepting_a_sequence(["d12"], self.alphabet_s)
        dfa1 = AutomataUtility.concatenate(dfa11, dfa12)
        dfa1 = dfa1.minify()
        dfa1 = AutomataUtility.new_state_names(dfa1)

        dfa31 = AutomataUtility.dfa_accepting_a_sequence(["s3"], self.alphabet_s)
        dfa32 = AutomataUtility.dfa_accepting_a_sequence(["c3"], self.alphabet_s)
        dfa33 = AutomataUtility.union(dfa31, dfa32)

        dfa34 = AutomataUtility.concatenate(dfa33, dfa33)
        dfa35 = AutomataUtility.closure_plus(dfa33)
        dfa3 = AutomataUtility.concatenate(dfa34, dfa35)
        dfa3.minify()

        dfa211 = AutomataUtility.dfa_accepting_a_sequence(["d2"], self.alphabet_s)
        dfa212 = AutomataUtility.dfa_accepting_a_sequence(["d12"], self.alphabet_s)
        dfa213 = AutomataUtility.dfa_accepting_a_sequence(["d23"], self.alphabet_s)
        dfa21 = AutomataUtility.union(dfa211, dfa212)
        dfa22 = AutomataUtility.union(dfa21, dfa213)
        dfa22 = AutomataUtility.new_state_names(dfa22)
        dfa23 = AutomataUtility.closure_plus(dfa22)
        dfa24 = AutomataUtility.dfa_accepting_a_sequence(["d12"], self.alphabet_s)
        dfa25 = AutomataUtility.concatenate(dfa23, dfa24)
        dfa2 = dfa25.minify()
        dfa2 = AutomataUtility.new_state_names(dfa2)

        dfa1 = AutomataUtility.super_sequence(dfa1)
        dfa3 = AutomataUtility.super_sequence(dfa3)
        dfa2 = AutomataUtility.super_sequence(dfa2)
        dfa2 = AutomataUtility.new_state_names(dfa2)

        dfa = AutomataUtility.intersection(dfa1, dfa3)
        dfa = dfa.minify()
        dfa = AutomataUtility.new_state_names(dfa)
        dfa = AutomataUtility.intersection(dfa, dfa2)
        dfa = AutomataUtility.new_state_names(dfa)
        dfa = dfa.minify()

        # May need to remove 
        dfa = AutomataUtility.new_state_names(dfa)

        AutomataUtility.print_dfa(dfa)

        return dfa
