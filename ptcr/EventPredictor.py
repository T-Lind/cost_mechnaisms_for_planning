import time
from sys import float_info
from typing import Tuple

from MarkovDecisionProcess import MDP, MDPState
from ptcr import DeterministicFiniteAutomaton, MarkovChain
from ptcr.MarkovChain import MarkovState


class EventPredictor:
    def __init__(self, dfa: DeterministicFiniteAutomaton, markov_chain: MarkovChain, event_list: set):
        self.dfa = dfa
        self.markov_chain = markov_chain
        self.event_list = event_list

        self.mdp = MDP()  # TODO: IMPLEMENT ADDING THIS
        self.current_markov_state_visible = True

        if self.current_markov_state_visible:
            self.__create_automaton_single_initial_state_only_reachables()


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

        # TODO: finish implementing this
