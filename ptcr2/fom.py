from copy import deepcopy

from rich import print
from rich.console import Console
from rich.table import Table

import ptcr2.automata_utility as AutomataUtility
from ptcr2.base_model import BaseModel
from ptcr2.event_predictor import EventPredictor
from ptcr2.markov_chain import MarkovChain

console = Console()


def flatten_and_get_strings(input_list):
    result_set = set()

    for item in input_list:
        if isinstance(item, list):
            # Recursively call the function for nested lists
            result_set.update(flatten_and_get_strings(item))
        elif isinstance(item, str):
            # Add string elements to the result set
            result_set.add(item)

    return result_set


class FOM(BaseModel):
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
        self.cost_matrix = spec["cost_matrix"]
        initial_distribution = spec["initial_distribution"]
        self.alphabet_s = set(spec['alphabet'])
        self.epsilon = spec.get('epsilon', 0.01)

        single_initial_state_0 = spec['single_initial_states'][0]
        single_initial_state_1 = spec['single_initial_states'][1]

        # Check if state names only has unique elements that are all strings
        if len(state_names) != len(set(state_names)):
            raise ValueError("state names are not unique")

        for initial_states in state_names:
            if not isinstance(initial_states, str):
                raise ValueError("state names are not all strings")

        # Check if the transition matrix is a square matrix of size len(state_names)
        if len(transition_matrix) != len(state_names):
            raise ValueError("transition matrix and state names have different lengths")

        for row in transition_matrix:
            # make sure every element in row is a float or int
            for element in row:
                if not isinstance(element, float) and not isinstance(element, int):
                    raise ValueError("transition matrix has non-float/int elements")
            if len(row) != len(state_names):
                raise ValueError("transition matrix is not square")
            if sum(row) != 1:
                raise ValueError(f"transition matrix row {row} does not sum to 1")

        # Check if the cost matrix is a square matrix of size len(state_names)
        if len(self.cost_matrix) != len(state_names):
            raise ValueError("cost matrix and state names have different lengths")

        for row in self.cost_matrix:
            # make sure every element in row is a float or int
            for element in row:
                if not isinstance(element, float) and not isinstance(element, int):
                    raise ValueError("cost matrix has non-float/int elements")
            if len(row) != len(state_names):
                raise ValueError("cost matrix is not square")

        if len(initial_distribution) != len(state_names):
            raise ValueError("initial distribution and state names have different lengths")

        for element in initial_distribution:
            if not isinstance(element, float) and not isinstance(element, int):
                raise ValueError("initial distribution has non-float/int elements")

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

        for event_list in state_events_1 + state_events_2 + state_events_3:
            for event in event_list:
                if event not in self.alphabet_s:
                    raise ValueError(f"event {event} not in alphabet")

        for initial_states in flatten_and_get_strings(single_initial_state_0 + single_initial_state_1):
            if initial_states not in self.alphabet_s:
                raise ValueError(f"state {initial_states} not in alphabet")

        print("Formatting checks succeeded.")

        transition_table = Table(title="Transition Matrix")
        transition_table.add_column("")
        for i in range(len(state_names)):
            transition_table.add_column(state_names[i])
        for i in range(len(state_names)):
            row_str = [str(element) for element in transition_matrix[i]]
            transition_table.add_row(state_names[i], *row_str)

        console.print(transition_table)

        initial_dist_table = Table(title="Initial Distribution")
        for i in range(len(state_names)):
            initial_dist_table.add_column(state_names[i])
        initial_dist_table.add_row(*[str(element) for element in initial_distribution])

        console.print(initial_dist_table)

        cost_matrix_table = Table(title="Cost Matrix")
        cost_matrix_table.add_column("")
        for i in range(len(state_names)):
            cost_matrix_table.add_column(state_names[i])
        for i in range(len(state_names)):
            row_str = [str(element) for element in self.cost_matrix[i]]
            cost_matrix_table.add_row(state_names[i], *row_str)

        console.print(cost_matrix_table)

        alphabet_table = Table(title="Alphabet")

        alphabet_table.add_column("Items")

        for element in self.alphabet_s:
            alphabet_table.add_row(element)

        console.print(alphabet_table)

        mc1 = MarkovChain(state_names, state_events_1, transition_matrix, initial_distribution, 0)

        state_names2 = deepcopy(state_names)

        mc2 = MarkovChain(state_names2, state_events_2, transition_matrix, initial_distribution, 0)

        state_names3 = deepcopy(state_names)

        mc3 = MarkovChain(state_names3, state_events_3, transition_matrix, initial_distribution, 0)

        single_initial_state_0[0][0] = tuple(single_initial_state_0[0][0])
        single_initial_state_1[0][0] = tuple(single_initial_state_1[0][0])

        print("Single initial state 0:", single_initial_state_0)
        print("Single initial state 1:", single_initial_state_1)

        mc12 = mc1.product_single_initial_state(mc2, single_initial_state_0)

        mc = mc12.product_single_initial_state(mc3, single_initial_state_1)

        dfa = self.get_dfa()

        self.ep = EventPredictor(dfa, mc, self.alphabet_s, self.cost_matrix, self.verbose)

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

        # AutomataUtility.print_dfa(dfa)

        return dfa
