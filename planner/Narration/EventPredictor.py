import queue
import time
from builtins import str
from sys import float_info

from planner.Markov.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from planner.Utility import AutomataUtility as AU


class EventPredictor:

    def __init__(self, dfa, markovChain, eventList, verbose, currentMarkovStateVisibile=True):
        self.dfa = dfa
        self.markovChain = markovChain
        self.eventList = eventList
        self.verbose = verbose
        self.mdp = None
        self.currentMarkovStateVisibile = currentMarkovStateVisibile
        if currentMarkovStateVisibile == True:
            # self.__createProductAutomaton_singleInitialState()
            self.__createProductAutomaton_singleInitialState_onlyreachables()
        else:
            self.__createProductAutomaton_WhereCurrentMarkovStateInvisible()

    def __createProductAutomaton(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        self.mdp = mdp

        mdp.actions = self.eventList

        """
        Create initial state, which is (q0, _), where _ denotes where the Markov chain has not 
        started yet. Consider that the Markov chain does not have a single initial state. 
        The set of the initial states of a Markov chain could be any state for which the 
        initial probability distribution has a value greater than zero.
        """
        t0 = (dfa.initial_state, mc.nullState)
        v0 = MDPState(dfa.initial_state + "," + "_" + mc.initialState.name, t0)
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0

        """
        Create the state space of the MDP. 
        The state space of the MDP is V = (Q \cup S) \cup {(q0, _)}.
        """
        for q in dfa.states:
            for s in mc.states:
                t = (q, s)
                v = MDPState(q + "_" + s.name, t)
                mdp.addState(v)
                if q in dfa.final_states:
                    v.isGoal = True
                    mdp.setAsGoal(v)

        """
        Create transitions from the initial state of the MDP, which is (q0, _)
        """
        for i in range(0, len(mc.states)):
            if mc.initial_distribution[i] == 0:
                continue
            s = mc.states[i]
            evs = s.events
            p = mc.getInitialProbability(s)
            for e in evs:
                q = dfa.transitions[q0][e]
                t = (q, s)
                v = mdp.getStateByAnchor(t)
                trans = MDPTransition(v0, v, e, evs, p)
                mdp.addTransition(trans)
            for e1 in self.eventList:
                if e1 in evs:
                    continue
                t1 = (q0, s)
                v1 = mdp.getStateByAnchor(t1)
                trans1 = MDPTransition(v0, v1, e1, evs, p)
                mdp.addTransition(trans1)

        for v1 in mdp.states:
            if v1 == v0:
                continue
            q = v1.anchor[0]
            s = v1.anchor[1]

            for v2 in mdp.states:
                if v2 == v0:
                    continue
                if v1 == v2:
                    for e1 in self.eventList:
                        p = mc.getTransitionProbability(s, s2)
                        trans = MDPTransition(v1, v2, e1, s2.events, p)
                        mdp.addTransition(trans)
                    continue

                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                if mc.getTransitionProbability(s, s2) == 0:
                    continue

                for e in s2.events:
                    if dfa.transitions[q][e] == q2:
                        p = mc.getTransitionProbability(s, s2)
                        trans = MDPTransition(v1, v2, e, s2.events, p)
                        mdp.addTransition(trans)
                if q == q2:
                    for e1 in self.eventList:
                        if e1 in s2.events:
                            continue
                        p = mc.getTransitionProbability(s, s2)
                        trans = MDPTransition(v1, v2, e1, s2.events, p)
                        mdp.addTransition(trans)
        if self.verbose == True:
            print("the product automata has been computed. It has " + str(len(mdp.states)) + " states")
            print("----------------------------------------------")

    def __createProductAutomaton_singleInitialState(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.hasEvidence = mc.has_evidence
        mdp.evidenceList = mc.evidence_list
        self.mdp = mdp

        mdp.actions = self.eventList

        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state + "_" + mc.initialState.name, t0)
        v0.evidenceDistribution = mc.initialState.evidence_distribution
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0

        """
        Create the state space of the MDP. 
        The state space of the MDP is V = (Q \cup S) \cup {(q0, _)}.
        """
        for q in dfa.states:
            for s in mc.states:
                if q == dfa.initial_state and s == mc.initialState:
                    continue
                t = (q, s)
                v = MDPState(q + "_" + s.name, t)
                v.evidenceDistribution = s.evidence_distribution
                mdp.addState(v)
                if q in dfa.final_states:
                    v.isGoal = True
                    mdp.setAsGoal(v)

        selfLoopAdded = {}
        for q in dfa.states:
            selfLoopAdded[q] = False

        """
        Create transitions
        """
        for v1 in mdp.states:
            # if v1 == v0:
            #    continue
            q = v1.anchor[0]
            s = v1.anchor[1]

            for v2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """
                # if v2 == v0:
                #   continue
                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                if mc.getTransitionProbability(s, s2) == 0:
                    continue

                #                 if v1 == v2:
                #                     for e1 in self.eventList:
                #                         p = mc.getTransitionProbability(s, s2)
                #                         trans = MDPTransition(v1, v2, e1, s2.events, p)
                #                         mdp.addTransition(trans)
                #                     continue

                for e in s2.events:
                    if dfa.transitions[q][e] == q2:
                        p = mc.getTransitionProbability(s, s2)
                        trans = MDPTransition(v1, v2, e, s2.events, p)
                        mdp.addTransition(trans)
                if q == q2:
                    for e1 in self.eventList:
                        if e1 in s2.events:
                            continue
                        # if selfLoopAdded[q]:
                        #    continue
                        # selfLoopAdded[q] = True
                        p = mc.getTransitionProbability(s, s2)
                        trans = MDPTransition(v1, v2, e1, s2.events, p)
                        mdp.addTransition(trans)
                        # selfLoopAdded[q] = True
        # self.mdp.recognizeReachableStates()

        # self.mdp.makeGoalStatesAbsorbing()

        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()

        # if mdp.has_evidence:
        mdp.makeObservationFunction()

        if self.verbose == True:
            print("the product automata has been computed. It has " + str(len(mdp.states)) + " states")
            print("----------------------------------------------")

    def __createProductAutomaton_singleInitialState_onlyreachables(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.hasEvidence = mc.has_evidence
        mdp.evidenceList = mc.evidence_list
        self.mdp = mdp

        mdp.actions = self.eventList

        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state + "_" + mc.initialState.name, t0)
        v0.evidenceDistribution = mc.initialState.evidence_distribution
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0

        queue = []
        queueNames = []
        queue.append(v0)
        queueNames.append(v0.name)

        cnt = 0
        i = 0
        while queue:
            i += 1
            print("Iteration " + str(i))
            v = queue.pop(0)
            vName = queueNames.pop(0)
            t = v.anchor
            q = t[0]
            s = t[1]
            # for e in self.eventList:
            for s2 in mc.states:
                probability = mc.getTransitionProbability(s, s2)
                if probability == 0:
                    continue
                for e in self.eventList:
                    if e not in s2.events:
                        continue
                    # print("Entereded the for loop")
                    q2 = self.dfa.transitions[q][e]
                    v2Name = q2 + "_" + s2.name
                    v2wasInMDP = False
                    if v2Name in mdp.statesDictByName.keys():
                        v2 = mdp.statesDictByName[v2Name]
                        v2wasInMDP = True
                    else:
                        t2 = (q2, s2)
                        v2 = MDPState(v2Name, t2)
                        v2.evidenceDistribution = s2.evidence_distribution
                        mdp.addState(v2)
                        cnt += 1
                        if q2 in dfa.final_states:
                            v2.isGoal = True
                            mdp.setAsGoal(v2)
                    if v2Name not in queueNames and v2wasInMDP == False:
                        queue.append(v2)
                        queueNames.append(v2Name)
                    tran = MDPTransition(v, v2, e, s2.events, probability)
                    mdp.addTransition(tran)
                for e in self.eventList:
                    if e in s2.events:
                        continue
                    v2Name = q + "_" + s2.name
                    v2wasInMDP = False
                    if v2Name in mdp.statesDictByName.keys():
                        v2 = mdp.statesDictByName[v2Name]
                        v2wasInMDP = True
                    else:
                        t2 = (q, s2)
                        v2 = MDPState(v2Name, t2)
                        v2.evidenceDistribution = s2.evidence_distribution
                        mdp.addState(v2)
                        cnt += 1
                        if q in dfa.final_states:
                            v2.isGoal = True
                            mdp.setAsGoal(v2)
                    if v2Name not in queueNames and v2wasInMDP == False:
                        queue.append(v2)
                        queueNames.append(v2Name)
                    tran = MDPTransition(v, v2, e, s2.events, probability)
                    mdp.addTransition(tran)
        #                 for e in self.eventList:
        #                     if dfa.transitions[q][e] != q:
        #                         continue
        #                     v2Name = q+"_"+s2.name
        #                     v2 = mdp.statesDictByName[v2Name]
        #                     tran = v.getTranisionByDistAndAction(v2, e)
        #                     tran.probability += mc.getTransitionProbability(s, s2)

        #         """
        #         Create transitions
        #         """
        #         for v1 in mdp.states:
        #             #if v1 == v0:
        #             #    continue
        #             q = v1.anchor[0]
        #             s = v1.anchor[1]
        #
        #             for v2 in mdp.states:
        #                 """
        #                 There should be no transition to the initial state of the MDP
        #                 """
        #                 #if v2 == v0:
        #                  #   continue
        #                 q2 = v2.anchor[0]
        #                 s2 = v2.anchor[1]
        #                 if mc.getTransitionProbability(s, s2) == 0:
        #                     continue
        #
        #                 if v1 == v2:
        #                     for e1 in self.eventList:
        #                         p = mc.getTransitionProbability(s, s2)
        #                         trans = MDPTransition(v1, v2, e1, s2.events, p)
        #                         mdp.addTransition(trans)
        #                     continue
        #
        #                 for e in s2.events:
        #                     if dfa.transitions[q][e] == q2:
        #                         p = mc.getTransitionProbability(s, s2)
        #                         trans = MDPTransition(v1, v2, e, s2.events, p)
        #                         mdp.addTransition(trans)
        #                 if q == q2:
        #                     for e1 in self.eventList:
        #                         if e1 in s2.events:
        #                             continue
        #                         p = mc.getTransitionProbability(s, s2)
        #                         trans = MDPTransition(v1, v2, e1, s2.events, p)
        #                         mdp.addTransition(trans)

        print("Number of added states = " + str(cnt))

        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()

        self.mdp.makeObservationFunction()

        if self.verbose == True:
            print("the product automata has been computed. It has " + str(len(mdp.states)) + " states")
            print("----------------------------------------------")

    def __createProductAutomaton_WhereCurrentMarkovStateInvisible(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        self.mdp = mdp

        mdp.actions = self.eventList

        """
        Create initial state, which is (q0, U_{s \in S: i_{init}(s) > 0}).
        """
        s0s = set()
        for i in range(len(mc.states)):
            if mc.initial_distribution[i] > 0:
                s0s.add(mc.states[i])
        t0 = (dfa.initial_state, s0s)
        v0 = MDPState(dfa.initial_state + "," + str(s0s), t0)
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0

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
                        mdp.addState(v2)
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
                    mdp.addTransition(trans)
                    if q2 in dfa.final_states:
                        mdp.setAsGoal(v2)
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
                        mdp.addState(v3)
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
                    mdp.addTransition(trans)
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
            if (self.mdp.states[j].isGoal):
                G[F][j] = 0.0
                A[F][j] = "STOP"
            else:
                G[F][j] = 10000.0

        for i in range(F - 1, -1, -1):
            # print(i)
            for j in range(n):
                if self.mdp.states[j].isGoal == True:
                    A[i][j] = "STOP"
                    G[i][j] = 0.0
                    continue

                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]

                for action in self.eventList:
                    val = 0.0
                    if state.isGoal == False:
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

        print("mdp.initialState=[" + self.mdp.initialState.anchor[0] + "," + self.mdp.initialState.anchor[1].name + "]")

        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][self.mdp.states[j].anchor[1]] = A[0][j]
            if printPolicy == True:
                print("\pi(" + self.mdp.states[j].anchor[0] + "," + self.mdp.states[j].anchor[1].name + ")=" + A[0][j])
                print(
                    "M(" + self.mdp.states[j].anchor[0] + "," + self.mdp.states[j].anchor[1].name + ")=" + str(G[0][j]))

        if self.verbose == True:
            print("optimal policy for finite horizon has been computed")

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index])

    def optimalPolicyInfiniteHorizon(self, epsilonOfConvergance=0.01, printPolicy=True, computeAvoidableActions=False):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)

        if computeAvoidableActions == True:
            self.mdp.computeAvoidableActions()

        # for state in self.mdp.states:
        # state.computeAvailableActions()

        G = [[0.0 for j in [0, 1]] for i in range(n)]
        A = ["" for j in range(n)]

        for j in range(n):
            if (self.mdp.states[j].isGoal):
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
                if self.mdp.states[j].isGoal == True:
                    continue

                if self.mdp.states[j].reachable == False:
                    continue

                if computeAvoidableActions and self.mdp.states[j].aGoalIsReachable == False:
                    continue

                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]

                # print("r="+str(r))

                # for action in self.eventList:
                # print("#Available actions for "+str(state)+": "+str(len(state.availableActions)))
                for action in state.availableActions:
                    if computeAvoidableActions:
                        if action in state.avoidActions:
                            continue
                    val = 0.0
                    if state.isGoal == False:
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

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)

    def optimalPolicyInfiniteHorizonForLayeredDAG(self, epsilonOfConvergance, printPolicy,
                                                  computeAvoidableActions=False):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)

        # for state in self.mdp.states:
        # state.computeAvailableActions()

        if computeAvoidableActions == True:
            if self.mdp.availableActionsComputed == False:
                self.mdp.computeAvoidableActions()

        G = [[0.0 for j in [0, 1]] for i in range(n)]
        A = ["" for j in range(n)]

        for j in range(n):
            if (self.mdp.states[j].isGoal):
                G[j][0] = 0.0
                G[j][1] = 0.0
                A[j] = "STOP"

        if computeAvoidableActions == True:
            for j in range(n):
                if self.mdp.states[j].aGoalIsReachable == False:
                    G[j][0] = G[j][1] = float("inf")
                    A[j] = "DeadEnd"
                    print("Set optimal action for dead end state " + self.mdp.states[j].name)

        dif = float_info.max

        numIterations = 0

        r = 0

        time_start = time.time()

        if self.mdp.initialSCC == None:
            self.mdp.decomposeToStrngConnComponents()
            self.mdp.computeSCCTopologicalOrder()

        for scc in self.mdp.sccTopologicalOrder:
            dif = float("inf")
            if len(scc.states) == 0:
                continue
            if scc.states[0].isGoal:
                continue
            firstTimeComputeDif = True
            # print("Start Computing Optimal Policy for states within SCC "+scc.name)
            while dif > epsilonOfConvergance:
                numIterations += 1
                maxDif = 0
                for state in scc.states:
                    if state.isGoal == True:
                        continue

                    if state.reachable == False:
                        continue

                    if computeAvoidableActions == True and state.aGoalIsReachable == False:
                        A[j] = "DeadEnd"
                        G[j][0] = G[j][1] = float("inf")
                        continue

                    minVal = float_info.max
                    optAction = ""

                    if len(state.availableActions) == 0:
                        print("Number of available actions of state " + str(state) + " is " + str(
                            len(state.availableActions)))

                    for action in state.availableActions:
                        if computeAvoidableActions == True:
                            if action in state.avoidActions:
                                continue

                        val = 0.0

                        if state.isGoal == False:
                            val += 1
                        for tran in state.actionsTransitions[action]:
                            # term = G[tran.dstState.index][1]*self.mdp.conditionalProbability(tran.dstState.index, state.index, action)
                            term = G[tran.dstState.index][1] * tran.probability
                            val += term
                        # r += 1
                        # print("r="+str(r))
                        if val < minVal:
                            minVal = val
                            optAction = action

                            # if minVal-G[j][0] > maxDif:
                    # maxDif = minVal-G[j][0]

                    maxDif = max(maxDif, minVal - G[state.index][0])

                    if state == scc.Leader:
                        dif = minVal - G[state.index][0]

                    G[state.index][0] = minVal
                    A[state.index] = optAction

                for s in scc.states:
                    G[s.index][1] = G[s.index][0]
                dif = maxDif
                print("numIterations=" + str(numIterations) + ", dif = " + str(dif))
            print("Optimal Policy for states within SCC " + scc.name + " was computed")

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
            print("Initial state is: " + str(self.mdp.initialState.name))

        time_elapsed = (time.time() - time_start)

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)

    def optimalPolicyInfiniteHorizon_Only4ReachableStates(self, epsilonOfConvergance, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)

        self.mdp.computeStatesAvailableActions()
        # for state in self.mdp.states:
        # state.computeAvailableActions()

        G = [[0.0 for j in [0, 1]] for i in range(n)]
        A = ["" for j in range(n)]

        for j in range(n):
            if (self.mdp.states[j].isGoal):
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
                if self.mdp.states[j].isGoal == True:
                    continue

                if self.mdp.states[j].reachable == False:
                    continue

                if self.mdp.states[j].aGoalIsReachable == False:
                    continue

                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]

                # print("r="+str(r))

                # for action in self.eventList:
                # print("#Available actions for "+str(state)+": "+str(len(state.availableActions)))
                for action in state.availableActions:
                    if action in state.avoidActions:
                        continue
                    val = 0.0
                    if state.isGoal == False:
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

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)

    def simulate(self, policy, printOutput=True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible(policy, printOutput)
        else:
            return self.__simulate_markovStateInvisible(policy, printOutput)

    def simulate_greedyAlgorithm(self, printOutput=True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible_greedyalgorithm(printOutput)
        # else:
        #    return self.__simulate_markovStateInvisible( policy, printOutput)     

    def simulate_greedyAlgorithm_pomdp(self, printOutput=True):
        # if self.currentMarkovStateVisibile == True:
        return self.__simulate_markovStateVisible_pomdp_greedyalgorithm(printOutput)

    def simulate_generalAndGreedyAlgorithms(self, policy, printOutput=True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible_generalAndGreedy(policy, printOutput)
        # else:
        #    return self.__simulate_markovStateInvisible( policy, printOutput)         

    def __simulate_markovStateVisible(self, policy, printOutput=True, ):
        story = ""
        # s = self.markovChain.nullState
        s = self.markovChain.initialState
        q = self.dfa.initial_state
        q_previous = q
        i = 0
        while True:
            if q in self.dfa.final_states:
                return (i, story)
            if (s.name in policy[q].keys()) == False:
                print("q=" + q + ", s=" + s.name)
            predictedEvent = policy[q][s.name]
            s2 = self.markovChain.next_state(s)

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
        # s = self.markovChain.nullState
        s = self.markovChain.initialState
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

            s2 = self.markovChain.next_state(s)

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
                eventLisToPredict = AU.getNonSelfLoopLetters(self.dfa, q2)
                print("eventLisToPredict: " + str(eventLisToPredict))
                print("s=" + str(s))
                predictedEvent2 = self.markovChain.getNexTimeMostPlausibleEvent(eventLisToPredict, s)
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
        # s = self.markovChain.nullState
        s = self.markovChain.initialState
        q = self.dfa.initial_state
        q_previous = q
        i = 0
        while True:
            if q in self.dfa.final_states:
                return (i, story)

            eventLisToPredict = AU.getNonSelfLoopLetters(self.dfa, q)

            predictedEvent = self.markovChain.getNexTimeMostPlausibleEvent(eventLisToPredict, s)
            s2 = self.markovChain.next_state(s)

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
        # s = self.markovChain.nullState
        # print("Start simulation a greedy algorithm for POMDP")
        mc = self.markovChain
        dfa = self.dfa
        mdp = self.mdp
        s = self.markovChain.initialState
        q = self.dfa.initial_state
        q_previous = q
        b = self.mdp.createAnInitialBeleifState()
        # print("__simulate_markovStateVisible_pomdp_greedyalgorithm")
        i = 0
        while True:
            if q in dfa.final_states:
                return (i, story)

            eventLisToPredict = AU.getNonSelfLoopLetters(self.dfa, q)

            # print("Step "+str(i))

            # print("useStateWithMaxProb = "+str(useStateWithMaxProb))

            if useStateWithMaxProb == True:
                x = mdp.getMostPlausibleState(b)
                x_s = x.anchor[1]
                predictedEvent = self.markovChain.getNexTimeMostPlausibleEvent(eventLisToPredict, x_s)
            else:
                predictedEvent = self.mdp.getNexTimeMostPlausibleEvent(b, eventLisToPredict, mc)

            # print("beleif = "+str(b)+", predictedEvent="+predictedEvent)

            s2 = self.markovChain.next_state(s)

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

    def __simulate_markovStateInvisible(self, policy, printOutput=True):
        story = ""
        sts = set()
        for i in range(len(self.markovChain.states)):
            if self.markovChain.initial_distribution[i] > 0:
                sts.add(self.markovChain.states[i])
        q = self.dfa.initial_state
        q_previous = q
        i = 1
        s = self.markovChain.next_state(self.markovChain.nullState)
        m = self.mdp.initialState
        while True:
            if q in self.dfa.final_states:
                return (i - 1, story)

            predictedEvent = policy[q][str(sts)]
            s2 = self.markovChain.next_state(s)

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
