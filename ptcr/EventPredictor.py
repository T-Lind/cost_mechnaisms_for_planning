import time
from sys import float_info
from typing import Tuple

from MarkovDecisionProcess import MDP, MDPState, MDPTransition
from ptcr import DeterministicFiniteAutomaton, MarkovChain
from ptcr.MarkovChain import MarkovState


class EventPredictor:
    def __init__(self, dfa: DeterministicFiniteAutomaton, markov_chain: MarkovChain, event_list: set):
        self.dfa = dfa
        self.markov_chain = markov_chain
        self.event_list = event_list

        self.mdp = MDP()
        self.current_markov_state_visible = True

        if self.current_markov_state_visible:
            self.__create_automaton_single_initial_state_only_reachables()
        else:
            raise NotImplementedError("Not Fully Visible Markov state not implemented yet")


    def optimal_policy_infinite_horizon(self, epsilon_of_convergence: float):
        len_mdp_states = len(self.mdp.states)
        G = [[0.0 for _ in [0, 1]] for _ in range(len_mdp_states)]
        A = ["" for _ in range(len_mdp_states)]

        for i in range(len_mdp_states):
            G[i][0] = 0.0
            G[i][1] = 0.0
            A[i] = "STOP"

        dif = float('inf')

        n_iterations = 0

        time_start = time.time()

        while dif > epsilon_of_convergence:
            n_iterations += 1

            max_diff = 0.0

            for j in range(len_mdp_states):
                if self.mdp.states[j].is_goal:
                    continue

                if not self.mdp.states[j].reachable:
                    continue

                min_value = float_info.max
                opt_action = ""

                state = self.mdp.states[j]

                for action in state.available_actions:
                    val = 0.0

                    for transition in state.action_transitions[action]:
                        term = G[transition.dst_state.index][1] * transition.probability
                        val += term

                    if val < min_value:
                        min_value = val
                        opt_action = action

                if j == 0:
                    dif = min_value - G[j][0]

                max_diff = max(max_diff, abs(min_value - G[j][0]))

                G[j][0] = min_value
                A[j] = opt_action

            for k in range(len_mdp_states):
                G[k][1] = G[k][0]

            dif = max_diff

        optimal_policy = {}
        for state in self.dfa.states:
            optimal_policy[state] = {}

        for l in range(len_mdp_states):
            optimal_policy[self.mdp.states[l].anchor[0]][str(self.mdp.states[l].anchor[1])] = A[l]

        time_end = time.time()

        return optimal_policy, G[0][self.mdp.initial_state.index], time_end - time_start

    def simulate(self, policy):
        if self.current_markov_state_visible:
            return self.__simulate_markov_state_visible(policy)
        else:
            raise NotImplementedError("Simulation for invisible Markov state not implemented yet")

    def __simulate_markov_state_visible(self, policy, max_steps=10_000) -> Tuple[int, str]:
        story = ""
        state = self.markov_chain.initial_state
        dfa_state = self.dfa.initial_state

        dfa_state_previous = dfa_state
        i = 0

        while i < max_steps:
            if dfa_state in self.dfa.accept_states:
                return i, story

            predicted_event = policy[dfa_state][state.name]
            state_2 = self.markov_chain.next_state(state)

            dfa_state_previous = dfa_state
            if predicted_event in state_2.events:
                dfa_state = self.dfa.transitions[(dfa_state, predicted_event)]
                if dfa_state != dfa_state_previous:
                    story += predicted_event
            i += 1
            state = state_2

        raise Exception("Simulation did not reach an accept state after 10,000 steps")

    def __create_automaton_single_initial_state_only_reachables(self):
        self.mdp.has_evidence = self.markov_chain.has_evidence
        self.evidence_list = self.markov_chain.evidence_list

        self.mdp.actions = self.event_list

        t0 = (self.dfa.initial_state, self.markov_chain.initial_state)
        v0 = MDPState(self.dfa.initial_state+"_"+self.markov_chain.initial_state.name, t0)
        v0.evidence_distribution = self.markov_chain.initial_state.evidence_distribution
        q0 = self.dfa.initial_state
        v0.is_initial = True
        self.mdp.add_state(v0)
        self.mdp.initial_state = v0

        queue = []
        queue_names = []
        queue.append(v0)
        queue_names.append(v0.name)

        cnt = 0
        i = 0
        while queue:
            i += 1

            v = queue.pop(0)
            v_name = queue_names.pop(0)

            t = v.anchor
            q = t[0]
            s = t[1]

            for dstState in self.markov_chain.states:
                probability = self.markov_chain.get_transition_probability(s, dstState)
                if probability == 0.0:
                    continue

                for event in self.event_list:
                    if event not in dstState.events:
                        continue

                    dstDFAState = self.dfa.transitions[(q, event)]
                    v2_name = dstDFAState + "_" + dstState.name

                    v2_was_in_mdp = False
                    if v2_name in self.mdp.states_dict_by_name.keys():
                        v2 = self.mdp.states_dict_by_name[v2_name]
                        v2_was_in_mdp = True
                    else:
                        t2 = (dstDFAState, dstState)
                        v2 = MDPState(v2_name, t2)
                        v2.evidence_distribution = dstState.evidence_distribution
                        self.mdp.add_state(v2)
                        cnt += 1
                        if dstDFAState in self.dfa.accept_states:
                            v2.is_goal = True
                            self.mdp.set_as_goal(v2)

                    if v2_name not in queue_names and not v2_was_in_mdp:
                        queue.append(v2)
                        queue_names.append(v2_name)
                    transition = MDPTransition(v, v2, event, dstState.events, probability)
                    self.mdp.add_transition(transition)

                for event in self.event_list:
                    if event in dstState.events:
                        continue

                    v2_name = q + "_" + dstState.name
                    v2_was_in_mdp = False

                    if v2_name in self.mdp.states_dict_by_name.keys():
                        v2 = self.mdp.states_dict_by_name[v2_name]
                        v2_was_in_mdp = True
                    else:
                        t2 = (q, dstState)
                        v2 = MDPState(v2_name, t2)
                        v2.evidence_distribution = dstState.evidence_distribution
                        self.mdp.add_state(v2)
                        cnt += 1
                        if q in self.dfa.accept_states:
                            v2.is_goal = True
                            self.mdp.set_as_goal(v2)

                    if v2_name not in queue_names and not v2_was_in_mdp:
                        queue.append(v2)
                        queue_names.append(v2_name)

                    transition = MDPTransition(v, v2, event, dstState.events, probability)
                    self.mdp.add_transition(transition)

        self.mdp.remove_unreachable_states()
        self.mdp.compute_states_available_actions()

        self.mdp.make_observable_function()





