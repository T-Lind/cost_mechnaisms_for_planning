import random

from planner.Markov.BeliefTree import BeliefTreeNode, BeliefTreeEdge, BeliefTree


# from gettext import lngettext


class MDPState:
    # name = ""
    # anchor = None
    # is_initial = False
    # is_goal = False
    # index = -1

    def __init__(self, name="", anchor=None):
        self.name = name
        self.anchor = anchor
        self.is_initial = False
        self.is_goal = False
        self.index = -1
        self.transitions = []
        self.reachable = True
        self.evidence_distr = []
        self.sum_prob_trans = 0

        self.availableActions = []
        self.actionsTransitions = {}

        ''' 
         The following three fields are used for decomposing the graph into strongly connected components
        '''
        self.scc_index = -1
        self.lowlink = -1
        self.scc = None

        """
        This field is used for DFS purposes
        """
        self.visited = False

        """
        This keeps the actions that the robot should avoid to do. 
        Doing any of them makes the robot to reach a dead end, a state from which no goal state is reachable.
        """
        self.avoidActions = []

        self.aGoalIsReachable = False

        """
        Set of transitions whose destination state is this
        """
        self.transitionsTo = []

        """
        A Boolean vector assigned to the state as the weight of the state
        """
        self.weightBVector = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def addTransition(self, trans):
        if trans in self.transitions:
            return
        if trans.probability > 1:
            print("state with probability greater than 1: state=" + trans.srcState.name + ", prob=" + str(
                trans.probability))

        self.sum_prob_trans += trans.probability

        # if self.sum_prob_trans > 1:
        # print("sum probability greater than 1: state="+trans.src_state.name+", prob="+str(self.sum_prob_trans))

        self.transitions.append(trans)

    def addTransitionTo(self, trans):
        if trans in self.transitionsTo:
            return
        self.transitionsTo.append(trans)

    def computeAvailableActions(self):
        for tran in self.transitions:
            if tran.action not in self.availableActions:
                self.actionsTransitions[tran.action] = []
                self.availableActions.append(tran.action)
            self.actionsTransitions[tran.action].append(tran)


class MDPTransition:

    def __init__(self, srcState, dstState, action, natureAction, probability):
        self.srcState = srcState
        self.dstState = dstState
        self.action = action
        self.natureAction = natureAction
        self.probability = probability
        self.eventPositive = False
        self.eventNegative = False

    def __str__(self):
        return str(self.srcState) + "--" + str(self.action) + "--" + str(self.probability) + "-->" + str(self.dstState)

    def __repr__(self):
        return str(self.srcState) + "--" + str(self.action) + "--" + str(self.probability) + "-->" + str(self.dstState)


class MDPStrgConnComponent:
    def __init__(self):
        self.states = []
        self.name = ""
        self.sccTransitions = []

        self.Leader = None

        """
        Used for DFS purposes
        """
        self.visited = False

    def addState(self, state):
        self.states.append(state)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def getFullName(self):
        result = "[name:" + self.name + ", states: {"
        i = 0
        for state in self.states:
            result += ("," if i > 0 else ",") + str(state)
            i += 1
        result += "} ]"
        return result

    def hasSCCTransitionTo(self, dstSCC):
        for scc in self.sccTransitions:
            if scc.dstSCC == dstSCC:
                return True
        return False

    def isSingular(self):
        if len(self.states) <= 1:
            return True
        return False

    def isASingularGoal(self):
        if len(self.states) > 1:
            return False
        if self.states[0].is_goal == False:
            return False
        return True


class SCCTransition:
    def __init__(self, srcSCC, dstSCC):
        self.srcSCC = srcSCC
        self.dstSCC = dstSCC

    def __str__(self):
        result = "[" + self.srcSCC.name + ", " + self.dstSCC.name + "]"
        return result

    def __repr__(self):
        return "[" + self.srcSCC.name + ", " + self.dstSCC + "]"


class MDP:

    def __init__(self):
        self.states = []
        self.actions = []
        self.initialState = MDPState()
        self.goalStates = []
        self.transitions = []
        self.transitionsDict = {}
        self.madeTransitionDict = False
        self.hasEvidence = False
        self.evidenceList = []
        self.observations = []

        """
        To access an entry of the observation function use observationFunction[o][s][a], where
        'o' is an action, 's' is a state, and 'a' is an action.
        """
        self.observationFunction = {}
        self.verbose = True
        self.beliefTree = None

        """
        These fields are used for decomposing the MDP into strongly connected components
        """
        self.strgConnCompoments = []
        self.initialSCC = None
        self.sccTransitions = []
        self.sccTopologicalOrder = []  # The strongly connected components are saved in a reverse topological ordering

        """
        This field is used for saving a topological ordering of the states 
        """
        self.topologicalOrder = []

        self.statesDictByName = {}

        """
        The property evidence_list contains only the original evidence and not the results of whether the prediction was right or not.
        We make new list of evidences here.  
        """
        self.evidenceTupleList = []

        """
        The observation that a goal state is reached
        """
        self.goal_reached_observation = "goal_reached"

        self.availableActionsComputed = False

        self.baseNum4CostLiniarization = 10

    def getObservationString(self, o):
        result = ""
        if isinstance(o, tuple):
            result = str(o[0]) + "_" + str(o[1])
        else:
            result = str(o)
        return result

    def createAnInitialBeleifState(self):
        beleif = [0.0] * len(self.states)
        beleif[self.initialState.index] = 1.0
        return beleif

    """
    Given a belief state over the POMDP, return a state that has the highest outcome in the beleif state
    """

    def getMostPlausibleState(self, beleif):
        maxOutCome = 0
        for i in range(len(beleif)):
            if beleif[i] > maxOutCome:
                maxOutCome = beleif[i]
        maxIndices = []
        for i in range(len((beleif))):
            if beleif[i] == maxOutCome:
                maxIndices.append(i)
        chosenIndex = random.choice(maxIndices)
        return self.states[chosenIndex]

    """
    Given that the current beleif state over the mdp is beleif, give from the list eventList, an event that highest probability to happen in the next time step
    """

    def getNexTimeMostPlausibleEvent(self, beleif, eventList, markovChain):
        maxProb = 0
        probs = [0] * len(eventList)
        for j in range(len(eventList)):
            event = eventList[j]
            prob = 0
            for i in range(len(beleif)):
                if beleif[i] == 0:
                    continue
                x = self.states[i]
                s = x.anchor[1]
                prob += markovChain.possbilityOfHappeningInNextStep(s, event)
            if prob > maxProb:
                maxProb = prob
            probs[j] = prob
        eventsToBeChosen = []
        for j in range(len(eventList)):
            if probs[j] == maxProb:
                eventsToBeChosen.append(eventList[j])
        if len(eventsToBeChosen) > 0:
            return random.choice(eventsToBeChosen)
        return None

    """
    prediction_result: True, False (whether the predicted event happened). evidence: raised from the event model
    """

    def getObservationOfTuple(self, prediction_result, evidence):
        for o in self.observations:
            if o[0] != prediction_result:
                continue
            if o[1] != evidence:
                continue
            return o
        return None

    def getBooleanObservation(self, booleanValue):
        for o in self.observations:
            if o == booleanValue:
                return o
        return None

    def noOfSingularSCCs(self):
        cnt = 0
        for scc in self.strgConnCompoments:
            if scc.isSingular() == True:
                cnt += 1
        return cnt

    def noOfSingularGoalSCCs(self):
        cnt = 0
        for scc in self.strgConnCompoments:
            if scc.isASingularGoal() == True:
                cnt += 1
        return cnt

    def makeObservationFunction(self):
        # if self.verbose:
        print("Start making observation function")
        self.observations = []
        if len(self.evidenceList) > 0:
            for ev in self.evidenceList:
                o1 = (True, ev)
                o2 = (False, ev)
                self.observations.append(o1)
                self.observations.append(o2)
        else:
            self.observations.append(True)
            self.observations.append(False)
        self.observations.append(self.goal_reached_observation)

        self.createObservationFunctionDict()

        if len(self.evidenceList) > 0:
            for x in self.states:
                s = x.anchor[1]  # The state of event model
                for a in self.actions:  # Event
                    for i in range(len(self.evidenceList)):
                        y = self.evidenceList[i]
                        o1 = self.getObservationOfTuple(True, y)
                        o2 = self.getObservationOfTuple(False, y)
                        o3 = self.goal_reached_observation
                        if x.is_goal:
                            self.observationFunction[o1][x][a] = 0
                            self.observationFunction[o2][x][a] = 0
                            self.observationFunction[o3][x][a] = 1
                        else:
                            self.observationFunction[o3][x][a] = 0
                            if a in s.events:
                                self.observationFunction[o1][x][a] = s.evidence_distribution[i]
                                self.observationFunction[o2][x][a] = 0
                            else:
                                self.observationFunction[o2][x][a] = s.evidence_distribution[i]
                                self.observationFunction[o1][x][a] = 0
        else:
            for x in self.states:
                s = x.anchor[1]
                for a in self.actions:
                    o1 = self.getBooleanObservation(True)
                    o2 = self.getBooleanObservation(False)
                    o3 = self.goal_reached_observation
                    if x.is_goal:
                        self.observationFunction[o1][x][a] = 0
                        self.observationFunction[o2][x][a] = 0
                        self.observationFunction[o3][x][a] = 1
                    else:
                        self.observationFunction[o3][x][a] = 0
                        if a in s.events:
                            self.observationFunction[o1][x][a] = 1
                            self.observationFunction[o2][x][a] = 0
                        else:
                            self.observationFunction[o2][x][a] = 1
                            self.observationFunction[o1][x][a] = 0

        # print("observations = "+str(self.observations))
        # print("observationFunction = "+str(self.observationFunction))

    def createObservationFunctionDict(self):
        self.observationFunction = {}
        for o in self.observations:
            self.observationFunction[o] = {}
            for x in self.states:
                self.observationFunction[o][x] = {}
                for e in self.actions:
                    self.observationFunction[o][x][e] = 0
        # print(str(self.observationFunction))

    def sumTransitionProbs(self, dstState, action, beleifState):
        total = 0
        #         for t in self.transitions:
        #             if t.dstState != dstState:
        #                 continue
        #             if t.action != action:
        #                 continue
        #             total += beleifState[t.src_state.index]*t.probability
        #         return total
        for srcState in self.states:
            total += beleifState[srcState.index] * self.conditionalProbability(dstState.index, srcState.index, action)
        return total

    def probabilityObservationGivenActionAndbeleif(self, observation, action, beleifState):
        total = 0
        for x in self.states:
            total += self.observationFunction[observation][x][action] * self.sumTransitionProbs(x, action, beleifState)
        return total

    def createBeleif(self, beleif, action, observation):
        b = [0] * len(self.states)
        # probObsAction = self.probabilityObservationGivenActionAndbeleif(observation, action, beleif)
        probObsAction = 0
        sumTransProbs = {}
        sumOfThem = 0

        for x in self.states:
            stp = self.sumTransitionProbs(x, action, beleif)
            sumTransProbs[x] = stp
            probObsAction += self.observationFunction[observation][x][action] * stp

        for x in self.states:
            # total = self.sumTransitionProbs(x, action, beleif)
            total = sumTransProbs[x]
            if total == 0 or self.observationFunction[observation][x][action] == 0:
                b[x.index] = 0
            else:
                b[x.index] = (self.observationFunction[observation][x][action] * total) / probObsAction
                """
            if self.observationFunction[observation][x][action] ==0 and self.probabilityObservationGivenActionAndbeleif(observation, action, beleif) == 0:
                print("O["+str(observation)+"]["+str(x)+"]["+str(action)+"]")
                b[x.index]
            else:
                if self.probabilityObservationGivenActionAndbeleif(observation, action, beleif) == 0:
                    print("b:"+ str(beleif))
                    print("O["+str(observation)+"]["+str(x)+"]["+str(action)+"]="+str(self.observationFunction[observation][x][action]))
                   
                b[x.index]=(self.observationFunction[observation][x][action]*total)/self.probabilityObservationGivenActionAndbeleif(observation, action, beleif)
                """
        # print("b="+str(b))
        return b

    def getGoalAvgValue(self, beleif):
        total = 0
        for x in self.goalStates:
            total += beleif[x.index]
            # print("x.index="+str(x.index))
        return total

    def addState(self, mdpState):
        mdpState.index = len(self.states)
        self.states.append(mdpState)
        self.statesDictByName[mdpState.name] = mdpState

    def setAsGoal(self, mdpState):
        mdpState.is_goal = True
        if not (mdpState in self.goalStates):
            self.goalStates.append(mdpState)

    def addTransition(self, mdpTransition):
        # if mdpTransition in self.transitions:
        #    print("Error. Transition has been added before")
        #    return

        # if self.hasTransition(mdpTransition.src_state, mdpTransition.dstState, mdpTransition.action):
        #    return

        self.transitions.append(mdpTransition)
        mdpTransition.srcState.addTransition(mdpTransition)
        mdpTransition.dstState.addTransitionTo(mdpTransition)

    def hasTransition(self, srcState, dstState, event):
        for t in self.transitions:
            if t.srcState == srcState:
                if t.dstState == dstState:
                    if t.action == event:
                        return True

        return False

    def getTransition(self, srcState, dstState, action):
        for t in srcState.transitions:
            if t.dstState == dstState and t.action == action:
                return t

        return None

    def computeAvoidableActions(self, printResults=False):
        print("Start computing avoidable actions")

        for state in self.states:
            state.aGoalIsReachable = False

        queue = []
        for state in self.states:
            if state.is_goal == True:
                queue.append(state)
                state.aGoalIsReachable = True

        while queue:
            state = queue.pop(0)
            for t in state.transitionsTo:
                # if t.dstState != state:
                #    continue
                allReachToGoals = True
                for t2 in t.srcState.actionsTransitions[t.action]:
                    # if t2.action != t.action:
                    #    continue 
                    if t2.dstState.aGoalIsReachable == False:
                        allReachToGoals = True
                if allReachToGoals == True and t.srcState.aGoalIsReachable == False:
                    t.srcState.aGoalIsReachable = True
                    queue.append(t.srcState)
        for state in self.states:
            for action in self.actions:
                allDstReachable = True
                # if action not in state.actionsTransitions.keys():
                #    continue
                if action not in state.actionsTransitions.keys():
                    print("Action " + str(
                        action) + " has no key in dictionary state.actionsTransitions for state " + state.name)
                else:
                    for trans in state.actionsTransitions[action]:
                        if trans.dstState.aGoalIsReachable == False:
                            allDstReachable = False
                            break
                if allDstReachable == False:
                    state.avoidActions.append(action)

        if printResults:
            for state in self.states:
                print("State: " + str(state.name) + ", aGoalIsReachable: " + str(state.aGoalIsReachable))

        if printResults:
            for state in self.states:
                if state.aGoalIsReachable == False:
                    continue
                # print("State: "+str(state.name)+", aGoalIsReachable: "+str(state.aGoalIsReachable))
                for action in state.avoidActions:
                    print("Avoid " + str(action) + " in state " + state.name)

        print("End computing avoidable actions ")

    def topologicalOrderHelper(self, state):
        state.visited = True
        for tran in state.transitions:
            if tran.dstState.visited == False:
                self.topologicalOrderHelper(tran.dstState)
        self.topologicalOrder.append(state)

    def computeTopologicalOrder(self):
        self.topologicalOrder = []
        for state in self.states:
            state.visited = False
        for state in self.states:
            if state.visited == False:
                self.topologicalOrderHelper(state)
    def sccTopologicalOrderHelper(self, scc):
        scc.visited = True
        for sccTrans in scc.sccTransitions:
            if sccTrans.dstSCC.visited == False:
                self.sccTopologicalOrderHelper(sccTrans.dstSCC)
        self.sccTopologicalOrder.append(scc)

    def computeSCCTopologicalOrder(self):
        self.sccTopologicalOrder = []
        for scc in self.strgConnCompoments:
            scc.visited = False
        for scc in self.strgConnCompoments:
            if scc.visited == False:
                self.sccTopologicalOrderHelper(scc)

    def makeSCCTransitions(self):
        for scc in self.strgConnCompoments:
            for state in scc.states:
                for trans in state.transitions:
                    if trans.dstState.scc == scc:
                        continue
                    if state.scc.hasSCCTransitionTo(trans.dstState.scc) == False:
                        sccTransition = SCCTransition(scc, trans.dstState.scc)
                        self.sccTransitions.append(sccTransition)
                        state.scc.sccTransitions.append(sccTransition)

    def decomposeToStrngConnComponents(self):
        self.sccIndex = 0
        self.stack4scc = []
        for state in self.states:
            if state.scc_index == -1:
                self.strongconnect(state)

        for i in range(len(self.strgConnCompoments)):
            self.strgConnCompoments[i].name = "C" + str(len(self.strgConnCompoments) - i - 1)
        self.makeSCCTransitions()

    def computeStatesAvailableActions(self):
        for state in self.states:
            state.computeAvailableActions()
        self.availableActionsComputed = True

    def reindexStates(self):
        i = 0
        for state in self.states:
            state.index = i
            i += 1

    def removeUnReachableStates(self):
        self.recognizeReachableStates()
        if self.verbose:
            print("Unreachable state have been recognized")
        statesToRemove = []
        for state in self.states:
            if state.reachable == False:
                statesToRemove.append(state)
        transitionsToRemove = []
        if self.verbose:
            print("Start checking transitions to remove")
            print("Number of transitions: " + str(len(self.transitions)))
        for trans in self.transitions:
            if trans.srcState.reachable == False or trans.dstState.reachable == False:
                transitionsToRemove.append(trans)
        if self.verbose:
            print("End checking transitions to remove")

        if self.verbose:
            print("Start removing unreachable transitions")
        cntRemovedTransitions = 0
        for trans in transitionsToRemove:
            self.transitions.remove(trans)
            cntRemovedTransitions += 1
        if self.verbose:
            print("End removing unreachable transitions")

        if self.verbose:
            print("Start removing unreachable states")
        ctnRemovedStates = 0
        for state in statesToRemove:
            self.states.remove(state)
            if state in self.goalStates:
                self.goalStates.remove(state)
            ctnRemovedStates += 1
        if self.verbose:
            print("End removing unreachable states")

        if self.verbose:
            print("Start reindexing states")
        self.reindexStates()
        if self.verbose:
            print("End reindexing states")

        if self.verbose:
            print("Number of unreachable states removed = " + str(ctnRemovedStates))
            print("Number of unreachable transitions removed = " + str(cntRemovedTransitions))

    def recognizeReachableStates(self):
        self.visited = [False] * len(self.states)
        for state in self.states:
            state.reachable = False
        self.dfs(self.initialState)

    def dfs(self, state):
        # print("DFS "+str(state))
        self.visited[state.index] = True
        state.reachable = True
        for t in state.transitions:
            if self.visited[t.dstState.index]:
                continue
            self.dfs(t.dstState)

    def getStateByAnchor(self, anchor):
        for s in self.states:
            if s.anchor == anchor:
                return s

        for s in self.states:
            eq = True
            for i in range(len(anchor)):
                if isinstance(anchor[i], set):
                    if len(anchor[i].difference(s.anchor[i])) != 0 or (len(s.anchor[i].difference(anchor[i])) != 0):
                        eq = False
                        break
                else:
                    if str(anchor[i]) != str(s.anchor[i]):
                        eq = False
                        break
            if eq:
                return s

        return None

    def getNextStateForEventPositive(self, state, event):
        for t in self.transitions:
            if t.srcState == state and t.action == event and t.eventPositive == True:
                return t.dstState
        return None

    def getNextStateForEventNegative(self, state, event):
        for t in self.transitions:
            if t.srcState == state and t.action == event and t.eventNegative == True:
                return t.dstState
        return None
    def check_transition_function(self):
        ok = True
        notFixed = ""
        for s in self.states:
            for a in self.actions:
                if self.availableActionsComputed:
                    if a not in s.availableActions:
                        continue
                sum = 0
                numOfTrans = 0
                recentT = None
                trans = []
                for t in s.transitions:
                    if t.action == a:
                        sum += t.probability
                        numOfTrans += 1
                        trans.append(t)
                        if t.probability != 0:
                            recentT = t
                if sum != 1:
                    ok = False
                    # print("Probability transition function does not sum to one: state="+s.name+", action="+ a+", prob="+str(sum)+", numOfTrans="+str(numOfTrans)+". "+str(trans))
                    # raise Exception("state="+s.name+", action="+ a+", prob="+str(sum)+", numOfTrans="+str(numOfTrans)+". "+str(trans))
                    if sum == 0.9999999999999999:
                        # print("Fix the error of mathematical operations")
                        recentT.probability = 1 - (sum - recentT.probability)
                    elif abs(sum - 1) < 0.000000001:
                        # print("Fix the error of mathematical operations")
                        recentT.probability = 1 - (sum - recentT.probability)
                    else:
                        print("Could not fix the error")
                        notFixed += "Could not fix error for: state=" + s.name + ", is_goal=" + str(
                            s.is_goal) + ", action=" + str(a) + ", prob=" + str(sum) + ", numOfTrans=" + str(
                            numOfTrans) + ". " + str(trans) + "\n"

        if notFixed != "":
            print(notFixed)
            raise Exception(notFixed)
        return ok

    def make_belief_tree(self, H):

        if self.verbose:
            print("Start making the belief Tree with height " + str(H))
            print("The size of the beleif tree will be in the order of (" + str(len(self.actions)) + "*" + str(
                len(self.observations)) + ")^" + str(H) + ", " + str(
                pow(len(self.actions) * len(self.observations), H)))

        beleif = [0] * len(self.states)
        for i in range(len(self.states)):
            beleif[i] = 0

        j = self.states.index(self.initialState)
        beleif[j] = 1

        node = BeliefTreeNode(beleif)
        node.height = 0
        if self.initialState in self.goalStates:
            node.goalAvgValue = 1

        bt = BeliefTree(node)

        queue = []

        queue.append(node)

        i = 0

        while len(queue) > 0:
            node = queue.pop(0)
            b = node.beliefState
            if node.height == H:
                continue
            for a in self.actions:
                for o in self.observations:
                    b2 = self.createBeleif(b, a, o)
                    node2 = BeliefTreeNode(b2)
                    node2.goalAvgValue = self.getGoalAvgValue(b2)
                    node2.height = node.height + 1
                    prob = self.probabilityObservationGivenActionAndbeleif(o, a, b)
                    edge = BeliefTreeEdge(node, node2, a, o, prob)
                    node.add_edge(edge)
                    node2.probabilityTo = node.probabilityTo * prob
                    queue.append(node2)
            if self.verbose and i % 1000 == 0:
                print("Number of nodes added to the belief tree is " + str(i))
            i = i + 1

        self.beliefTree = bt

        if self.verbose:
            print("The belief tree was made ")
            # print("mdp.beliefTree="+str(self.beliefTree))

        return bt

    def getObservation(self, exMarkovChainState, eventOccured):
        for o in self.observations:
            if exMarkovChainState == o[1]:
                if eventOccured == o[0]:
                    return o
        return None

    """
    Check if vect1 is dominated by vect2
    """

    def isDominated(self, vect1, vect2):
        m = len(vect1)
        result = False
        # print("vect="+str(vect))
        for j in range(m):
            if vect2[j] > vect1[j]:
                result = True
                break
            elif vect2[j] < vect1[j]:
                result = False
                break
        return result

    def isVectsEqual(self, vect1, vect2):
        m = len(vect1)
        result = True
        for j in range(m):
            if vect2[j] != vect1[j]:
                result = False
                break
        return result

    def containtsValTuple(self, valTupleList, valTuple):
        for v in valTupleList:
            if self.areValTuplesEqual(v, valTuple):
                return True
        return False

    def areValTuplesEqual(self, valueTuple1, valueTuple2):
        if valueTuple1[0] != valueTuple2[0]:
            return False
        for i in range(len(valueTuple1[1])):
            if valueTuple1[1][i] != valueTuple2[1][i]:
                return False
        return True

    def getNonDominantVectors(self, v_vectors):
        nonDominants = []
        for vect in v_vectors:
            dominated = False
            for vect2 in v_vectors:
                if vect2[0] < vect[0] and self.getPreferenceValueVector(vect2[1]) > self.getPreferenceValueVector(
                        vect[1]):
                    dominated = True
                    break

            if not dominated:
                nonDominants.append(vect)

        return nonDominants

    def getNonDominantActionsAndVectors(self, v_action_vectors):
        nonDominants = []
        # print("v_action_vectors: "+str(v_action_vectors))
        for a_vect in v_action_vectors:
            dominated = False
            for a_vect2 in v_action_vectors:
                if a_vect2[1][0] <= a_vect[1][0] and self.isDominated(a_vect[1][1], a_vect2[1][1]):
                    dominated = True
                    break
                elif a_vect2[1][0] < a_vect[1][0] and (
                        self.isDominated(a_vect[1][1], a_vect2[1][1]) or self.isVectsEqual(a_vect[1][1],
                                                                                           a_vect2[1][1])):
                    dominated = True
                    break
                elif a_vect2[1][0] == a_vect[1][0] and self.isVectsEqual(a_vect[1][1], a_vect2[1][1]) and a_vect2[0] < \
                        a_vect[0]:
                    dominated = True
                    break

            if not dominated:
                nonDominants.append(a_vect)

        return nonDominants

    def getNonDominantTimeViolationCostVectors(self, v_vectors):
        nonDominants = []
        # print("v_action_vectors: "+str(v_action_vectors))
        for a_vect in v_vectors:
            dominated = False
            for a_vect2 in v_vectors:
                if a_vect2[0] <= a_vect[0] and self.isDominated(a_vect[1], a_vect2[1]):
                    dominated = True
                    break
                elif a_vect2[0] < a_vect[0] and (
                        self.isDominated(a_vect[1], a_vect2[1]) or self.isVectsEqual(a_vect[1], a_vect2[1])):
                    dominated = True
                    break
                    # elif a_vect2[0] == a_vect[0] and self.isVectsEqual(a_vect[1], a_vect2[1]) and a_vect2[0] < a_vect[0]:
                #    dominated = True
                #    break

            if not dominated:
                nonDominants.append(a_vect)

        return nonDominants

    def roundValueVect(self, valueVect, digits):
        # print("valueVect: "+str(valueVect))
        # print("valueVect[0]: " +str(valueVect[0]))
        result = []
        lng = round(valueVect[0], digits)
        softs = []
        for num in valueVect[1]:
            softs.append(round(num, digits))
        tp = (lng, softs)
        # print("tp: "+str(tp))
        return tp

    def getOrder4ComputingPolicy(self):
        queue = []
        for s in self.goalStates:
            for t in s.transitionsTo:
                if t.srcState not in queue and (not t.srcState.is_goal):
                    queue.append(t.srcState)
        i = 0
        while i < len(queue):
            s = queue[i]
            for t in s.transitionsTo:
                if t.srcState not in queue and (not t.srcState.is_goal):
                    queue.append(t.srcState)
            i += 1
        return queue

    def computeNumstepsAndSatisProbOfAPolicy(self, states, policy, V, byStateNameOrIndex=False):
        """
        First compute the values of goal states
        """
        if byStateNameOrIndex:
            V = {}
        else:
            V = [() for i in range(len(self.states))]
        m = len(self.goalStates[0].weightBVector)
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.is_goal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            if byStateNameOrIndex:
                V[s.name] = vect
            else:
                V[s.index] = vect

        cnt = 1
        maxCnt = 500
        while cnt < maxCnt:
            for s in states:
                if s.is_goal:
                    continue
                # print(f"self.availableActionsComputed: {self.availableActionsComputed}")
                # if self.availableActionsComputed and not s.aGoalIsReachable:
                #    continue
                if byStateNameOrIndex:
                    a = policy[s.name]
                else:
                    a = policy[s.index]
                steps = 1
                soft_probs = [0] * m
                if a not in s.availableActions:
                    print(f"Error1: Action {a} is not in availableActions of state {s}")
                    print("Action: " + str(a))
                if a not in s.actionsTransitions.keys():
                    print(f"Error2: Action {a} is not in availableActions of state {s}")
                    print("Action: " + str(a))
                    self.printActions()
                    continue
                for tran in s.actionsTransitions[a]:
                    s2 = tran.dstState
                    if byStateNameOrIndex:
                        v_vect = V[s2.name]
                    else:
                        v_vect = V[s2.index]
                    steps += v_vect[0] * tran.probability
                    for j in range(m):
                        soft_probs[j] += v_vect[1][j] * tran.probability
                tple = (steps, soft_probs)
                if byStateNameOrIndex:
                    V[s.name] = tple
                else:
                    V[s.index] = tple
            cnt += 1
        print("Recalculated V: " + str(V))
        return V

    def printActions(self):
        i = 1
        for a in self.actions:
            print(f"{i}. {a}")

    def write_POMDPX_XML(self, filePath):
        f = open(filePath, "w+")
        f.write(self.getPOMDPX_XML2())
        f.close()
