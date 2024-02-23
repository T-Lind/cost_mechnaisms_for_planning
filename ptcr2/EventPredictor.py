from ptcr2.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from sys import float_info
import queue
from builtins import str

import ptcr2.AutomataUtility as AutomataUtility

import time


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
                    trans.eventNegative = True
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

        print("mdp.initial_state=[" + self.mdp.initial_state.anchor[0] + "," + self.mdp.initial_state.anchor[1].name + "]")

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

    def optimal_policy_infinite_horizon(self, epsilonOfConvergance=0.01, printPolicy=False,
                                        computeAvoidableActions=False):
        n = len(self.mdp.states)

        if computeAvoidableActions == True:
            self.mdp.computeAvoidableActions()

        # for state in self.mdp.states:
        # state.computeAvailableActions()

        G = [[0.0 for j in [0, 1]] for i in range(n)]
        # save G to a file
        # with open('G.txt', 'w') as f:
        #     print("G IS WRITING RIGHT NOW RIGHT HERE AS", G, "X" * 50)
        #     f.write(str(G))

        A = ["" for j in range(n)]

        for j in range(n):
            if (self.mdp.states[j].is_goal):
                G[j][0] = 0.0
                G[j][1] = 0.0
                A[j] = "STOP"

        # dif = float_info.max

        dif = float("inf")

        numIterations = 0

        time_start = time.time()

        r = 0

        while dif > epsilonOfConvergance:
            numIterations += 1

            print("dif=" + str(dif))

            # print("r="+str(r))

            # if numIterations > 1000:
            # break

            maxDif = 0

            for j in range(n):
                if self.mdp.states[j].is_goal == True:
                    continue

                if self.mdp.states[j].reachable == False:
                    continue

                if computeAvoidableActions and self.mdp.states[j].aGoalIsReachable == False:
                    continue

                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]

                # print("r="+str(r))

                # for action in self.actions:
                # print("#Available actions for "+str(state)+": "+str(len(state.availableActions)))
                for action in state.availableActions:
                    if computeAvoidableActions:
                        if action in state.avoidActions:
                            continue
                    val = 0.0
                    if state.is_goal == False:
                        val += 1
                    # for k in range(n):
                    #    term = G[k][1]*self.mdp.conditionalProbability(k, j, action)
                    #    val += term
                    for tran in state.actionsTransitions[action]:
                        term = G[tran.dstState.index][1] * tran.probability
                        val += term
                        # r += 1
                        # print("r="+str(r))
                    if val < minVal:
                        minVal = val
                        optAction = action

                        # if minVal-G[j][0] > maxDif:
                    # maxDif = minVal-G[j][0]

                if j == 0:
                    dif = minVal - G[j][0]

                maxDif = max(maxDif, minVal - G[j][0])

                G[j][0] = minVal
                A[j] = optAction

            for j in range(n):
                G[j][1] = G[j][0]

            dif = maxDif

        optPolicy = {}
        for q in self.dfa.states:
            optPolicy[q] = {}

        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][str(self.mdp.states[j].anchor[1])] = A[j]
            if printPolicy == True:
                print("\pi(" + self.mdp.states[j].anchor[0] + "," + str(self.mdp.states[j].anchor[1]) + ")=" + A[j])
                print(
                    "M(" + self.mdp.states[j].anchor[0] + "," + str(self.mdp.states[j].anchor[1]) + ")=" + str(G[j][0]))

        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed in " + str(numIterations) + " iterations")

        time_elapsed = (time.time() - time_start)

        # return G[0][self.mdp.initial_state.index]
        return optPolicy, G, G[0][self.mdp.initial_state.index], time_elapsed

    def simulate(self, policy):
        if self.current_markov_state_visible:
            return self.__simulate_markov_state_visible(policy)
        else:
            return self.__simulate_markov_state_invisible(policy)

    def simulate_greedy_algorithm(self, printOutput=False):
        if self.current_markov_state_visible:
            return self.__simulate_markovStateVisible_greedyalgorithm(printOutput)
        else:
            raise NotImplementedError('Invisible Markov state is not implemented yet')


    def simulate_general_and_greedy_algorithms(self, policy, printOutput=False):
        if self.current_markov_state_visible:
            return self.__simulate_markovStateVisible_generalAndGreedy(policy, printOutput)
        else:
            raise NotImplementedError('Invisible Markov state is not implemented yet')

    def __simulate_markov_state_visible(self, policy, printOutput=False, ):
        story = ""
        # s = self.markov_chain.null_state
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state
        q_previous = q
        i = 0
        while True:
            if q in self.dfa.final_states:
                return (i, story)
            if (s.name in policy[q].keys()) == False:
                print("q=" + q + ", s=" + s.name)
            predictedEvent = policy[q][s.name]
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predictedEvent in s2.events:
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            i += 1

            if printOutput == True:
                print(
                    str(i) + ". " + "q=" + q_previous + ", s=" + s.name + ", predicted=" + predictedEvent + ", actual=" + str(
                        s2.events) + ", sNext=" + s2.name + ", recorded story=" + story)

            s = s2
        return (i, story)

    def __simulate_markovStateVisible_generalAndGreedy(self, policy, printOutput=True):
        story = ""  # for the general algorithm
        story2 = ""  # for the greedy algorithm
        # s = self.markov_chain.null_state
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state  # q is for the general algorithm
        q_previous = q  # q_previous is for the general algorithm
        q2 = self.dfa.initial_state  # q is for the greedy algorithm
        q2_previous = q2  # q_previous is for the greedy algorithm
        i = 0
        steps = 0  # Number of steps of general algorithm
        steps2 = 0  # Number of steps for greedy algorithm
        while True:
            if q in self.dfa.final_states and q2 in self.dfa.final_states:
                return (steps, story, steps2, story2)
            if (s.name in policy[q].keys()) == False:
                print("q=" + q + ", s=" + s.name)

            s2 = self.markov_chain.next_state(s)

            if q not in self.dfa.final_states:
                steps += 1
                predictedEvent = policy[q][s.name]
                q_previous = q
                if predictedEvent in s2.events:
                    q = self.dfa.transitions[q][predictedEvent]
                    if q != q_previous:
                        story += predictedEvent

            if q2 not in self.dfa.final_states:
                steps2 += 1
                eventLisToPredict = AutomataUtility.get_non_self_loop_letters(self.dfa, q2)
                print("eventLisToPredict: " + str(eventLisToPredict))
                print("s=" + str(s))
                predictedEvent2 = self.markov_chain.get_next_time_most_plausible_event(eventLisToPredict, s)
                q2_previous = q2
                if predictedEvent2 in s2.events:
                    q2 = self.dfa.transitions[q2][predictedEvent2]
                    if q2 != q2_previous:
                        story2 += predictedEvent2

            i += 1

            if predictedEvent2 == None:
                print("predictedEvent2 is None")

            if printOutput == True:
                print(str(i) + ". " + "q=" + (q_previous if q_previous not in self.dfa.final_states else "") + "q2=" + (
                    q2_previous if q2_previous not in self.dfa.final_states else "") + ", s=" + s.name + ", predicted=" + predictedEvent + ", predicted2=" + predictedEvent2 + ", actual=" + str(
                    s2.events) + ", sNext=" + s2.name + ", recorded story=" + story + ", recorded story2=" + story2)

            s = s2
        return (steps, story, steps2, story2)

    def __simulate_markovStateVisible_greedyalgorithm(self, printOutput=True):
        story = ""
        # s = self.markov_chain.null_state
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state
        q_previous = q
        i = 0
        while True:
            if q in self.dfa.final_states:
                return (i, story)

            eventLisToPredict = AutomataUtility.get_non_self_loop_letters(self.dfa, q)

            predictedEvent = self.markov_chain.get_next_time_most_plausible_event(eventLisToPredict, s)
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predictedEvent in s2.events:
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            i += 1

            if printOutput == True:
                print(
                    str(i) + ". " + "q=" + q_previous + ", s=" + s.name + ", predicted=" + predictedEvent + ", actual=" + str(
                        s2.events) + ", sNext=" + s2.name + ", recorded story=" + story)

            s = s2
        return (i, story)

    def __simulate_markovStateVisible_pomdp_greedyalgorithm(self, printOutput=True):
        useStateWithMaxProb = False
        story = ""
        # s = self.markov_chain.null_state
        # print("Start simulation a greedy algorithm for POMDP")
        mc = self.markov_chain
        dfa = self.dfa
        mdp = self.mdp
        s = self.markov_chain.initial_state
        q = self.dfa.initial_state
        q_previous = q
        b = self.mdp.createAnInitialBeleifState()
        # print("__simulate_markovStateVisible_pomdp_greedyalgorithm")
        i = 0
        while True:
            if q in dfa.final_states:
                return (i, story)

            eventLisToPredict = AutomataUtility.get_non_self_loop_letters(self.dfa, q)

            # print("Step "+str(i))

            # print("useStateWithMaxProb = "+str(useStateWithMaxProb))

            if useStateWithMaxProb == True:
                x = mdp.getMostPlausibleState(b)
                x_s = x.anchor[1]
                predictedEvent = self.markov_chain.get_next_time_most_plausible_event(eventLisToPredict, x_s)
            else:
                predictedEvent = self.mdp.get_next_time_most_plausible_event(b, eventLisToPredict, mc)

            # print("beleif = "+str(b)+", predictedEvent="+predictedEvent)

            s2 = self.markov_chain.next_state(s)

            q_previous = q

            pred_result = False

            if predictedEvent in s2.events:
                pred_result = True
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent

            evidence = mc.pomdp_raiseEvidence(s2)

            # print("raised evidence: "+evidence+", at state: "+str(s2))

            o = mdp.getObservationOfTuple(pred_result, evidence)

            # print("Start updating beleif state")

            b = mdp.createBeleif(b, predictedEvent, o)

            # print("End updating beleif state")

            i += 1

            if printOutput == True:
                print(
                    str(i) + ". " + "q=" + q_previous + ", s=" + s.name + ", predicted=" + predictedEvent + ", actual=" + str(
                        s2.events) + ", sNext=" + s2.name + ", recorded story=" + story)

            s = s2

        # print("End simulation a greedy algorithm for POMDP")

        return (i, story)

    def __simulate_markov_state_invisible(self, policy, printOutput=True):
        story = ""
        sts = set()
        for i in range(len(self.markov_chain.states)):
            if self.markov_chain.initial_distribution[i] > 0:
                sts.add(self.markov_chain.states[i])
        q = self.dfa.initial_state
        q_previous = q
        i = 1
        s = self.markov_chain.next_state(self.markov_chain.null_state)
        m = self.mdp.initial_state
        while True:
            if q in self.dfa.final_states:
                return (i - 1, story)

            predictedEvent = policy[q][str(sts)]
            s2 = self.markov_chain.next_state(s)

            q_previous = q
            if predictedEvent in s2.events:
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            if printOutput == True:
                print(
                    str(i) + ". " + "q=" + q_previous + ", s=" + s.name + ", predicted=" + predictedEvent + ", actual=" + str(
                        s2.events) + ", possibleStates=" + str(
                        sts) + ", sNext=" + s2.name + ", recorded story=" + story)
            if predictedEvent in s2.events:
                m = self.mdp.getNextStateForEventPositive(m, predictedEvent)
                sts = m.anchor[1]
            else:
                m = self.mdp.getNextStateForEventNegative(m, predictedEvent)
                sts = m.anchor[1]
            i += 1
            s = s2
        return (i - 1, story)
