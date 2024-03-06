import random

import numpy


class MarkovState:
    # name = ""
    # events = set()
    # index = -1
    def __init__(self, name="", events=set(), evidence_distribution=[]):
        self.name = name
        self.events = events
        self.evidence_distribution = evidence_distribution
        self.index = -1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
        return self.name + ":" + str(self.events)
        s = "(" + self.name + ", " + str(self.events) + ")"
        return s


class MarkovChain:

    def __init__(self, state_names, state_events, transition_matrix, initial_distribution, initial_state_index=0,
                 has_evidence=False, evidence_list=[], cost_matrix=None):
        self.states = []
        self.state_names = state_names
        self.state_events = state_events
        self.initial_state_index = initial_state_index
        self.__create_states(state_names, state_events)
        self.initial_distribution = initial_distribution
        self.cost_matrix = cost_matrix
        self.transition_matrix = transition_matrix
        self.events = set()
        for s in self.states:
            for e in s.events:
                if not (e in self.events):
                    self.events.add(e)
        self.initial_state = self.states[initial_state_index]
        self.null_state = MarkovState("none", "none")
        self.has_evidence = has_evidence
        self.evidence_list = evidence_list

    def __create_states(self, state_names, state_events):
        self.states = []

        i = 0

        for name in state_names:
            state = MarkovState(name, state_events[i])
            state.index = len(self.states)
            self.states.append(state)
            i += 1

    def get_transition_probability(self, src_state, dst_state):
        return self.transition_matrix[src_state.index][dst_state.index]

    def pomdp_raiseEvidence(self, currentState):
        # print("current_state.evidence_distribution: "+str(current_state.evidence_distribution))
        return numpy.random.choice(self.evidence_list, p=currentState.evidence_distribution)

    def next_state(self, current_state):
        if current_state == self.null_state:
            next_state = numpy.random.choice(self.states, p=self.initial_distribution)
            return next_state
        next_state = numpy.random.choice(self.states, p=self.transition_matrix[current_state.index])
        return next_state


    def get_successors_having_event(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transition_matrix[state.index][j] > 0 and (event in self.states[j].events):
                succ.add(self.states[j])
        return succ

    def get_successors_not_having_event(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transition_matrix[state.index][j] > 0 and not (event in self.states[j].events):
                succ.add(self.states[j])
        return succ

    def get_set_successors_having_event(self, stateSet, event):
        succ = set()
        for state in stateSet:
            scc = self.get_successors_having_event(state, event)
            for s in scc:
                if not (s in succ):
                    succ.add(s)
        return succ

    def get_set_successors_not_having_event(self, state_set, event):
        succ = set()
        for state in state_set:
            scc = self.get_successors_not_having_event(state, event)
            for s in scc:
                if not (s in succ):
                    succ.add(s)
        return succ

    """
    The probablity that the given event happens in the next time step given that the event model is currently in state current_state
    """

    def p_of_happening_in_next_step(self, current_state, event):
        result = 0.0
        for state in self.states:
            if event not in state.events:
                continue
            result += self.get_transition_probability(current_state, state)
        return result

    def get_next_time_most_plausible_event(self, event_list, current_state):
        max_prob = -1
        probs = [0] * len(event_list)
        i = 0
        for ev in event_list:
            prob = self.get_next_time_probability_of_event(ev, current_state)
            probs[i] = prob
            if prob > max_prob:
                max_prob = prob
            i += 1

        selected_events = []
        for i in range(len(event_list)):
            if probs[i] == max_prob:
                selected_events.append(event_list[i])
                # print("selectedEvents="+str(selectedEvents))

        if len(selected_events) > 0:
            return random.choice(selected_events)
        elif len(event_list) > 0:
            return event_list[0]

    def get_next_time_probability_of_event(self, event, current_state):
        result = 0
        for state in self.states:
            if self.get_transition_probability(current_state, state) == 0:
                continue
            if event not in state.events:
                continue
            result += self.get_transition_probability(current_state, state)

        return result

    def product_singleInitialState(self, markov_chain, pair_events_list=None):
        if pair_events_list is None:
            pair_events_list = []
        prod_states = []

        state_names = []

        state_events = []

        initial_distribution = []

        initial_state_index = -1

        k = 0

        for i in range(len(self.states)):
            for j in range(len(markov_chain.states)):
                if self.states[i] == self.initial_state and markov_chain.states[j] != markov_chain.initial_state:
                    continue
                if markov_chain.states[j] == markov_chain.initial_state and self.states[i] != self.initial_state:
                    continue
                if self.states[i] == self.initial_state and markov_chain.states[j] == markov_chain.initial_state:
                    initial_state_index = k
                s1 = self.states[i]
                s2 = markov_chain.states[j]
                initial_distribution.append(self.initial_distribution[i] * markov_chain.initial_distribution[j])
                state_name = s1.name + "_" + s2.name
                state_names.append(state_name)
                event_set = s1.events.union(s2.events)
                for event_pair in pair_events_list:
                    appeared = True
                    event_grabbed = ""
                    for event in event_pair[0]:
                        appeared = appeared and event in event_set
                        event_grabbed = event_grabbed + event
                    if len(event_pair[0]) > 0 and appeared:
                        event_set.add(event_pair[1])
                state_events.append(event_set)
                prod_state = MarkovState(state_name, event_set)
                prod_state.anchor = (s1, s2)
                prod_states.append(prod_state)
                k = k + 1

        num_states = k

        transition_matrix = [[0 for _ in range(num_states)] for _ in range(num_states)]

        for k in range(0, len(prod_states)):
            state = prod_states[k]
            s1, s2 = state.anchor

            for event_pair in range(0, len(prod_states)):
                state_prime = prod_states[event_pair]
                s1_prime, s2_prime = state_prime.anchor
                p = self.transition_matrix[s1.index][s1_prime.index] * markov_chain.transition_matrix[s2.index][
                    s2_prime.index]
                transition_matrix[k][event_pair] = p

        # do something similar for cost matrix
        cost_matrix = [[0 for _ in range(num_states)] for _ in range(num_states)]
        for k in range(0, len(prod_states)):
            state = prod_states[k]
            s1, s2 = state.anchor
            for event_pair in range(0, len(prod_states)):
                state_prime = prod_states[event_pair]
                s1_prime, s2_prime = state_prime.anchor
                c = self.cost_matrix[s1.index][s1_prime.index] + markov_chain.cost_matrix[s2.index][s2_prime.index]
                cost_matrix[k][event_pair] = c


        return MarkovChain(state_names, state_events, transition_matrix, initial_distribution, initial_state_index,
                           cost_matrix=cost_matrix)

    def product(self, markov_chain, pair_events_list=None):
        if pair_events_list is None:
            pair_events_list = []

        prod_states = []

        state_names = []

        state_events = []

        initial_distribution = []

        for i in range(len(self.states)):
            for j in range(len(markov_chain.states)):
                s1 = self.states[i]
                s2 = markov_chain.states[j]
                state_name = s1.name + "_" + s2.name
                state_names.append(state_name)
                event_set = s1.events.union(s2.events)
                for event_pair in pair_events_list:
                    appeared = True
                    grabbed_event = ""
                    for event in event_pair[0]:
                        appeared = appeared and (event in event_set)
                        grabbed_event = grabbed_event + event
                    if len(event_pair[0]) > 0 and appeared:
                        event_set.add(event_pair[1])
                state_events.append(event_set)
                prod_state = MarkovState(state_name, event_set)
                prod_state.anchor = (s1, s2)
                prod_states.append(prod_state)

        num_states = len(markov_chain.states) * len(self.states)

        transition_matrix = [[0 for _ in range(num_states)] for _ in range(num_states)]

        for k in range(0, len(prod_states)):
            state = prod_states[k]
            s1, s2 = state.anchor

            for event_pair in range(0, len(prod_states)):
                state_prime = prod_states[event_pair]
                s1_prime, s2_prime = state_prime.anchor
                p = self.transition_matrix[s1.index][s1_prime.index] * markov_chain.transition_matrix[s2.index][
                    s2_prime.index]
                transition_matrix[k][event_pair] = p

        # do something similar for cost matrix
        cost_matrix = [[0 for _ in range(num_states)] for _ in range(num_states)]
        for k in range(0, len(prod_states)):
            state = prod_states[k]
            s1, s2 = state.anchor
            for event_pair in range(0, len(prod_states)):
                state_prime = prod_states[event_pair]
                s1_prime, s2_prime = state_prime.anchor
                c = self.cost_matrix[s1.index][s1_prime.index] + markov_chain.cost_matrix[s2.index][s2_prime.index]
                cost_matrix[k][event_pair] = c

        return MarkovChain(state_names, state_events, transition_matrix, initial_distribution,
                           cost_matrix=cost_matrix)
