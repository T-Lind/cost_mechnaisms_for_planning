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
        state_names = spec["state_names"]
        state_events_1, state_events_2, state_events_3 = spec["state_events"]

        for i in range(len(state_events_1)):
            state_events_1[i] = set(state_events_1[i])
            state_events_2[i] = set(state_events_2[i])
            state_events_3[i] = set(state_events_3[i])

        transition_matrix = spec["transition_matrix"]
        initial_distribution = spec["initial_distribution"]

        # Check if state names only has unique elements that are all strings
        if len(state_names) != len(set(state_names)):
            raise ValueError("state names are not unique")

        for state in state_names:
            if not isinstance(state, str):
                raise ValueError("state names are not all strings")

        # Check if the transition matrix is a square matrix of size len(state_names)

        if len(transition_matrix) != len(state_names):
            raise ValueError("transition matrix and state names have different lengths")

        for row in transition_matrix:
            # make sure every element in row is a float
            for element in row:
                if not isinstance(element, float):
                    raise ValueError("transition matrix has non-float elements")
            if len(row) != len(state_names):
                raise ValueError("transition matrix is not square")
            if sum(row) != 1:
                raise ValueError(f"transition matrix row {row} does not sum to 1")

        if len(initial_distribution) != len(state_names):
            raise ValueError("initial distribution and state names have different lengths")

        for element in initial_distribution:
            if not isinstance(element, float):
                raise ValueError("initial distribution has non-float elements")

        if sum(initial_distribution) != 1:
            raise ValueError("initial distribution does not sum to 1")


        if len(state_events_1) == 0:
            state_events_1 = [set() for _ in state_names]

        if len(state_events_2) == 0:
            state_events_2 = [set() for _ in state_names]

        if len(state_events_3) == 0:
            state_events_3 = [set() for _ in state_names]

        # Check if state_events_1, state_events_2, and state_events_3 are of the same length as state_names
        if len(state_events_1) != len(state_names):
            raise ValueError("state events 1 and state names have different lengths")

        if len(state_events_2) != len(state_names):
            raise ValueError("state events 2 and state names have different lengths")

        if len(state_events_3) != len(state_names):
            raise ValueError("state events 3 and state names have different lengths")

        print("Formatting checks succeeded.")

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
