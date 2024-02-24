import queue
import time
from builtins import str
from sys import float_info

import ptcr2.AutomataUtility as AutomataUtility
from ptcr2.MarkovDecisionProcess import MDP, MDPState, MDPTransition

MAX_UNSAFE_ITERS = 100_000


class EventPredictor:
    def __init__(self, dfa, markov_chain, actions, verbose, markov_state_visible=True):
        self.dfa = dfa
        self.markov_chain = markov_chain
        self.actions = actions
        self.verbose = verbose
        self.mdp = None
        self.current_markov_state_visible = markov_state_visible
        if markov_state_visible:
            self.__create_product_automaton_single_initial_state_onlyreachables()
        else:
            self.__create_product_automaton_where_current_markov_state_invisible()

    def __create_product_automaton_single_initial_state_onlyreachables(self):
        dfa = self.dfa
        mc = self.markov_chain
        mdp = MDP()
        mdp.has_evidence = mc.has_evidence
        mdp.evidence_list = mc.evidence_list
        self.mdp = mdp

        mdp.actions = self.actions

        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initial_state)
        v0 = MDPState(dfa.initial_state + "_" + mc.initial_state.name, t0)
        v0.evidence_distribution = mc.initial_state.evidence_distribution
        v0.is_initial = True
        mdp.add_state(v0)
        mdp.initial_state = v0

        queue = []
        queue_names = []
        queue.append(v0)
        queue_names.append(v0.name)

        cnt = 0
        i = 0
        while queue:
            i += 1
            v = queue.pop(0)
            t = v.anchor
            q = t[0]
            s = t[1]
            # for e in self.actions:
            for s2 in mc.states:
                probability = mc.get_transition_probability(s, s2)
                if probability == 0:
                    continue
                for e in self.actions:
                    if e not in s2.events:
                        continue
                    q2 = self.dfa.transitions[q][e]
                    v2_name = q2 + "_" + s2.name
                    v2_was_in_mdp = False
                    if v2_name in mdp.states_dict_by_name.keys():
                        v2 = mdp.states_dict_by_name[v2_name]
                        v2_was_in_mdp = True
                    else:
                        t2 = (q2, s2)
                        v2 = MDPState(v2_name, t2)
                        v2.evidence_distribution = s2.evidence_distribution
                        mdp.add_state(v2)
                        cnt += 1
                        if q2 in dfa.final_states:
                            v2.is_goal = True
                            mdp.set_as_goal(v2)
                    if v2_name not in queue_names and not v2_was_in_mdp:
                        queue.append(v2)
                        queue_names.append(v2_name)
                    tran = MDPTransition(v, v2, e, s2.events, probability)
                    mdp.add_transition(tran)
                for e in self.actions:
                    if e in s2.events:
                        continue
                    v2_name = q + "_" + s2.name
                    v2_was_in_mdp = False
                    if v2_name in mdp.states_dict_by_name.keys():
                        v2 = mdp.states_dict_by_name[v2_name]
                        v2_was_in_mdp = True
                    else:
                        t2 = (q, s2)
                        v2 = MDPState(v2_name, t2)
                        v2.evidence_distribution = s2.evidence_distribution
                        mdp.add_state(v2)
                        cnt += 1
                        if q in dfa.final_states:
                            v2.is_goal = True
                            mdp.set_as_goal(v2)
                    if v2_name not in queue_names and not v2_was_in_mdp:
                        queue.append(v2)
                        queue_names.append(v2_name)
                    tran = MDPTransition(v, v2, e, s2.events, probability)
                    mdp.add_transition(tran)
        self.mdp.remove_un_reachable_states()
        self.mdp.compute_states_available_actions()

        self.mdp.make_observation_function()

    def __create_product_automaton_where_current_markov_state_invisible(self):
        dfa = self.dfa
        mc = self.markov_chain
        mdp = MDP()
        self.mdp = mdp

        mdp.actions = self.actions

        """
        Create initial state, which is (q0, U_{s \in S: i_{init}(s) > 0}).
        """
        s0s = set()
        for i in range(len(mc.states)):
            if mc.initial_distribution[i] > 0:
                s0s.add(mc.states[i])
        t0 = (dfa.initial_state, s0s)
        v0 = MDPState(dfa.initial_state + "," + str(s0s), t0)
        v0.is_initial = True
        mdp.add_state(v0)
        mdp.initial_state = v0

        que = queue.Queue()
        que.put(v0)
        while not (que.empty()):
            v1 = que.get()
            t1 = v1.anchor
            q1 = t1[0]
            sts1 = t1[1]
            if q1 in dfa.final_states:
                continue

            for e in mc.events:
                """
                ________________________________begin_________________________________________
                Add from transition (v1, e, v2) where v2 is C_{e+}(v1)
                """
                sts2 = mc.get_set_successors_having_event(sts1, e)
                p1 = 0
                p2 = 0
                if len(sts2) != 0:
                    q2 = dfa.transitions[q1][e]
                    t2 = (q2, sts2)
                    v2 = mdp.getStateByAnchor(t2)
                    if v2 == None:
                        v2 = MDPState(q2 + ", " + str(sts2), t2)
                        mdp.add_state(v2)
                        que.put(v2)
                    evs = set()
                    for s in sts2:
                        for ev in s.events:
                            if not (ev in evs):
                                evs.add(ev)
                    p = 0
                    for s in sts1:
                        for s2 in sts2:
                            p = p + mc.transition_matrix[s.index][s2.index]
                            # print("p1="+str(p)+", len(sts1)="+str(len(sts1)))
                    p = p / len(sts1)
                    p1 = p
                    # print("p1="+str(p))
                    trans = MDPTransition(v1, v2, e, evs, p)
                    trans.eventPositive = True
                    mdp.add_transition(trans)
                    if q2 in dfa.final_states:
                        mdp.set_as_goal(v2)
                """
                _________________________________end__________________________________________
                Add from transition (v1, e, v2) where v2 is C_{e+}(v1)
                """

                # for e in mc.events:

                """
                ________________________________begin_________________________________________
                Add from transition (v1, e, v3) where v3 is C_{e-}(v1)
                """
                sts3 = mc.get_set_successors_not_having_event(sts1, e)
                if len(sts3) != 0:
                    t3 = (q1, sts3)
                    v3 = mdp.getStateByAnchor(t3)
                    if v3 == None:
                        v3 = MDPState(q1 + ", " + str(sts3), t3)
                        mdp.add_state(v3)
                        que.put(v3)
                    evs = set()
                    for s in sts3:
                        for ev in s.events:
                            if not (ev in evs):
                                evs.add(ev)
                    p = 0
                    for s in sts1:
                        for s3 in sts3:
                            p = p + mc.transition_matrix[s.index][s3.index]
                            # print("p2="+str(p)+", len(sts1)="+str(len(sts1)))
                    p = p / len(sts1)
                    p2 = p
                    # print("p2="+str(p))
                    trans = MDPTransition(v1, v3, e, evs, p)
                    trans.event_negative = True
                    mdp.add_transition(trans)
                    if p1 + p2 != 1:
                        print("p1+p2=" + str(p1 + p2))
                        p2 = 1.0 - p1
                """
                _________________________________end__________________________________________
                Add from transition (v1, e, v2) where v3 is C_{e-}(v1)
                """

        if self.verbose == True:
            print("the product automata has been computed. It has " + str(len(mdp.states)) + " states")
            print("----------------------------------------------")

    def optimalPolicyFiniteHorizon(self, F, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        G = [[0.0 for j in range(n)] for i in range(F + 1)]
        A = [["" for j in range(n)] for i in range(F + 1)]

        for j in range(n):
            if (self.mdp.states[j].is_goal):
                G[F][j] = 0.0
                A[F][j] = "STOP"
            else:
                G[F][j] = 10000.0

        for i in range(F - 1, -1, -1):
            # print(i)
            for j in range(n):
                if self.mdp.states[j].is_goal == True:
                    A[i][j] = "STOP"
                    G[i][j] = 0.0
                    continue

                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]

                for action in self.actions:
                    val = 0.0
                    if state.is_goal == False:
                        val += 1
                    for k in range(n):
                        term = G[i + 1][k] * self.mdp.conditionalProbability(k, j, action)
                        val += term
                    if val < minVal:
                        minVal = val
                        optAction = action
                G[i][j] = minVal
                A[i][j] = optAction

        optPolicy = {}
        for q in self.dfa.states:
            optPolicy[q] = {}

        print("mdp.initial_state=[" + self.mdp.initial_state.anchor[0] + "," + self.mdp.initial_state.anchor[
            1].name + "]")

        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][self.mdp.states[j].anchor[1]] = A[0][j]
            if printPolicy == True:
                print("\pi(" + self.mdp.states[j].anchor[0] + "," + self.mdp.states[j].anchor[1].name + ")=" + A[0][j])
                print(
                    "M(" + self.mdp.states[j].anchor[0] + "," + self.mdp.states[j].anchor[1].name + ")=" + str(G[0][j]))

        if self.verbose == True:
            print("optimal policy for finite horizon has been computed")

        # return G[0][self.mdp.initial_state.index]
        return (optPolicy, G, G[0][self.mdp.initial_state.index])

    def optimal_policy_infinite_horizon(self, epsilon_of_convergence=0.01, compute_avoidable_actions=False):
        n = len(self.mdp.states)

        if compute_avoidable_actions:
            self.mdp.compute_avoidable_actions()

        G = [[0.0 for _ in [0, 1]] for _ in range(n)]
        A = ["" for _ in range(n)]

        for j in range(n):
            if self.mdp.states[j].is_goal:
                G[j][0] = 0.0
                G[j][1] = 0.0
                A[j] = "STOP"


        difference = float("inf")

        num_iterations = 0

        diff_tracker = []

        time_start = time.time()

        while difference > epsilon_of_convergence:
            num_iterations += 1
            max_dif = 0

            print(f"dif={difference:.4f}")

            if difference != float("inf"):
                diff_tracker.append(difference)

            for j in range(n):
                if self.mdp.states[j].is_goal:
                    continue

                if not self.mdp.states[j].reachable:
                    continue

                if compute_avoidable_actions and not self.mdp.states[j].a_goal_is_reachable:
                    continue

                min_val = float_info.max
                opt_action = ""
                state = self.mdp.states[j]

                for action in state.available_actions:
                    if compute_avoidable_actions:
                        if action in state.avoid_actions:
                            continue
                    val = 0.0
                    if not state.is_goal:
                        val += 1

                    for tran in state.actions_transitions[action]:
                        term = G[tran.dst_state.index][1] * tran.probability
                        val += term
                    if val < min_val:
                        min_val = val
                        opt_action = action

                if j == 0:
                    difference = min_val - G[j][0]

                max_dif = max(max_dif, min_val - G[j][0])

                G[j][0] = min_val
                A[j] = opt_action

            for j in range(n):
                G[j][1] = G[j][0]

            difference = max_dif

        optimal_policy = {}
        for q in self.dfa.states:
            optimal_policy[q] = {}

        for j in range(n):
            optimal_policy[self.mdp.states[j].anchor[0]][str(self.mdp.states[j].anchor[1])] = A[j]

        time_elapsed = (time.time() - time_start)

        return optimal_policy, G, G[0][self.mdp.initial_state.index], time_elapsed, diff_tracker

    def simulate(self, policy):
        if self.current_markov_state_visible:
            return self.__simulate_markov_state_visible(policy)
        else:
            return self.__simulate_markov_state_invisible(policy)

    def simulate_greedy_algorithm(self):
        if self.current_markov_state_visible:
            return self.__simulate_markov_state_visible_greedyalgorithm()
        else:
            raise NotImplementedError('Invisible Markov state is not implemented yet')

    def simulate_general_and_greedy_algorithms(self, policy):
        if self.current_markov_state_visible:
            return self.__simulate_markov_state_visible_general_and_greedy(policy)
        else:
            raise NotImplementedError('Invisible Markov state is not implemented yet')

    def __simulate_markov_state_visible(self, policy):
        story = ""
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state
        i = 0
        while True:
            if q in self.dfa.final_states:
                return i, story
            predicted_event = policy[q][s.name]
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predicted_event in s2.events:
                q = self.dfa.transitions[q][predicted_event]
                if q != q_previous:
                    story += predicted_event
            i += 1
            s = s2

            if i > MAX_UNSAFE_ITERS:
                raise Exception(f'Max iterations reached in {__name__}')

    def __simulate_markov_state_visible_general_and_greedy(self, policy):
        story = ""  # for the general algorithm
        story2 = ""  # for the greedy algorithm
        # s = self.markov_chain.null_state
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state  # q is for the general algorithm
        q2 = self.dfa.initial_state  # q is for the greedy algorithm
        i = 0
        steps = 0  # Number of steps of general algorithm
        steps2 = 0  # Number of steps for greedy algorithm
        while True:
            if q in self.dfa.final_states and q2 in self.dfa.final_states:
                return (steps, story, steps2, story2)

            s2 = self.markov_chain.next_state(s)

            if q not in self.dfa.final_states:
                steps += 1
                predicted_event = policy[q][s.name]
                q_previous = q
                if predicted_event in s2.events:
                    q = self.dfa.transitions[q][predicted_event]
                    if q != q_previous:
                        story += predicted_event

            if q2 not in self.dfa.final_states:
                steps2 += 1
                event_list_to_predict = AutomataUtility.get_non_self_loop_letters(self.dfa, q2)
                predicted_event2 = self.markov_chain.get_next_time_most_plausible_event(event_list_to_predict, s)
                q2_previous = q2
                if predicted_event2 in s2.events:
                    q2 = self.dfa.transitions[q2][predicted_event2]
                    if q2 != q2_previous:
                        story2 += predicted_event2
            i += 1
            s = s2

            if i > MAX_UNSAFE_ITERS:
                raise Exception(f'Max iterations reached in {__name__}')

    def __simulate_markov_state_visible_greedyalgorithm(self):
        story = ""
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state
        i = 0
        while True:
            if q in self.dfa.final_states:
                return i, story

            event_list_to_predict = AutomataUtility.get_non_self_loop_letters(self.dfa, q)

            predicted_event = self.markov_chain.get_next_time_most_plausible_event(event_list_to_predict, s)
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predicted_event in s2.events:
                q = self.dfa.transitions[q][predicted_event]
                if q != q_previous:
                    story += predicted_event
            i += 1
            s = s2

            if i > MAX_UNSAFE_ITERS:
                raise Exception(f'Max iterations reached in {__name__}')

    def __simulate_markov_state_invisible(self, policy):
        story = ""
        sts = set()
        for i in range(len(self.markov_chain.states)):
            if self.markov_chain.initial_distribution[i] > 0:
                sts.add(self.markov_chain.states[i])
        q = self.dfa.initial_state
        i = 1
        s = self.markov_chain.next_state(self.markov_chain.null_state)
        m = self.mdp.initial_state
        while True:
            if q in self.dfa.final_states:
                return i - 1, story

            predicted_event = policy[q][str(sts)]
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predicted_event in s2.events:
                q = self.dfa.transitions[q][predicted_event]
                if q != q_previous:
                    story += predicted_event
            if predicted_event in s2.events:
                m = self.mdp.getNextStateForEventPositive(m, predicted_event)
                sts = m.anchor[1]
            else:
                m = self.mdp.getNextStateForEventNegative(m, predicted_event)
                sts = m.anchor[1]
            i += 1
            s = s2

            if i > MAX_UNSAFE_ITERS:
                raise Exception(f'Max iterations reached in {__name__}')
