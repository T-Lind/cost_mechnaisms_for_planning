import itertools
import math
import random
import time
from copy import deepcopy
from sys import float_info

from planner.Markov.BeliefTree import BeliefTreeNode, BeliefTreeEdge, BeliefTree


# from gettext import lngettext


class MDPState:
    # name = ""
    # anchor = None
    # isInitial = False
    # isGoal = False
    # index = -1

    def __init__(self, name="", anchor=None):
        self.name = name
        self.anchor = anchor
        self.isInitial = False
        self.isGoal = False
        self.index = -1
        self.transitions = []
        self.reachable = True
        self.evidenceDistribution = []
        self.sumProbabilityTransitions = 0

        self.availableActions = []
        self.actionsTransitions = {}

        ''' 
         The following three fields are used for decomposing the graph into strongly connected components
        '''
        self.sccIndex = -1
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

        self.sumProbabilityTransitions += trans.probability

        # if self.sumProbabilityTransitions > 1:
        # print("sum probability greater than 1: state="+trans.src_state.name+", prob="+str(self.sumProbabilityTransitions))

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

    def getTranisionByDistAndAction(self, dstState, action):
        for tran in self.transitions:
            if tran.action != action and tran.dstState == dstState:
                return tran
        return None


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
        if self.states[0].isGoal == False:
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
                        if x.isGoal:
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
                    if x.isGoal:
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
        mdpState.isGoal = True
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
            if state.isGoal == True:
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

    def topologicalOrderStack(self, state):
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

    def computeTopologicalOrderRecursive(self):
        self.topologicalOrder = []
        for state in self.states:
            state.visited = False
        stack = []
        stack.append(self.initialState)
        state = self.initialState
        state.visited = True

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
            if state.sccIndex == -1:
                self.strongconnect(state)

        for i in range(len(self.strgConnCompoments)):
            self.strgConnCompoments[i].name = "C" + str(len(self.strgConnCompoments) - i - 1)
        self.makeSCCTransitions()

    def strongconnect(self, state):
        state.sccIndex = self.sccIndex
        state.lowlink = self.sccIndex
        self.sccIndex = self.sccIndex + 1
        self.stack4scc.append(state)
        state.onStack = True
        for trans in state.transitions:
            state2 = trans.dstState
            if state2.sccIndex == -1:
                self.strongconnect(state2)
                state.lowlink = min(state.lowlink, state2.lowlink)
            elif state2.onStack:
                state.lowlink = min(state.lowlink, state2.lowlink)

        if state.sccIndex == state.lowlink:
            scc = MDPStrgConnComponent()
            scc.Leader = state
            # scc.name = "C"+str(len(self.strgConnCompoments))
            self.strgConnCompoments.append(scc)
            if state == self.initialState:
                self.initialSCC = scc
            hasPopedState = False
            while hasPopedState == False:
                state2 = self.stack4scc.pop()
                state2.onStack = False
                scc.addState(state2)
                state2.scc = scc
                if state2 == state:
                    hasPopedState = True

    def printStrgConnComponents(self):
        print("------------------------- Start Printing Strongly Connected Components ----------------------")
        print("The MDP has " + str(len(self.strgConnCompoments)) + " strongly connected components")
        for scc in self.strgConnCompoments:
            print(scc.getFullName())
        print("------------------------- End Printing Strongly Connected Components ----------------------")

    def printGraphOfStrgConnComponents(self):
        print(
            "------------------------- Start Printing The Graph of Strongly Connected Components ----------------------")
        print("The MDP has " + str(len(self.strgConnCompoments)) + " strongly connected components")
        print("Initial SCC:" + self.initialSCC.name)
        for sccTrans in self.sccTransitions:
            print(str(sccTrans))
        print(
            "------------------------- End Printing The Graph of Strongly Connected Components ----------------------")

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

    def makeTransitionsDict(self):

        n = len(self.states)
        self.transitionsDict = {}
        for a in self.actions:
            self.transitionsDict[a] = {}
            for i in range(n):
                self.transitionsDict[a][i] = {}
                for j in range(n):
                    self.transitionsDict[a][i][j] = 0
        for t in self.transitions:
            self.transitionsDict[t.action][t.srcState.index][t.dstState.index] = t.probability
        self.madeTransitionDict = True
        print("TransitionsDict has been made")

    def finiteHorizonOptimalPolicy(self, F, verbose):
        if verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)
        G = [[0.0 for j in range(n)] for i in range(F + 1)]
        A = [["" for j in range(n)] for i in range(F + 1)]

        for j in range(n):
            if (self.states[j].isGoal):
                G[F][j] = 0.0
            else:
                G[F][j] = 10000.0

        for i in range(F - 1, -1, -1):
            # print(i)
            for j in range(n):
                if self.states[j].isGoal == True:
                    A[i][j] = "STOP"
                    G[i][j] = 0.0
                    continue

                minVal = float_info.max;
                optAction = ""
                state = self.states[j]

                for action in self.actions:
                    val = 0.0
                    if state.isGoal == False:
                        val += 1
                    for k in range(n):
                        term = G[i + 1][k] * self.conditionalProbability(k, j, action)
                        val += term
                    if val < minVal:
                        minVal = val
                        optAction = action
                G[i][j] = minVal
                A[i][j] = optAction

        optPolicy = {}

        for j in range(n):
            optPolicy[self.states[j]] = A[0][j]
            if verbose == True:
                print("\pi(" + self.states[j].name + ")=" + A[0][j])

        if verbose == True:
            print("optimal policy for finite horizon has been computed")

        return optPolicy

    def conditionalProbability(self, nextStateIndex, currentStateIndex, action):
        if self.madeTransitionDict == False:
            self.makeTransitionsDict()
        return self.transitionsDict[action][currentStateIndex][nextStateIndex]

    def printAll(self):
        print("-------------------------------------------MDP--------------------------------------------")
        print("States:")
        for s in self.states:
            print(s)
        print("Transitions:")

        for t in self.transitions:
            print(t)

        print("Goal States")
        for g in self.goalStates:
            print(g)

        print("-------------------------------------------------------------------------------------------")

    def printToFile(self, fileName):
        strP = "-------------------------------------------MDP--------------------------------------------" + "\n"
        strP += "Number of states=" + str(len(self.states)) + "\n"
        strP += "States:" + "\n"
        for s in self.states:
            strP += str(s) + "\n"
        strP += "Transitions:" + "\n"

        for t in self.transitions:
            strP += str(t) + "\n"

        strP += "Goal States" + "\n"
        for g in self.goalStates:
            strP += str(g) + "\n"

        strP += "-------------------------------------------------------------------------------------------" + "\n"

        f = open(fileName, "w")
        f.write(strP)
        f.close()

    def makeGoalStatesAbsorbing(self):
        cnt = 0

        for t in self.transitions:
            if t.srcState.isGoal == False:
                continue

            if t.srcState != t.dstState:
                t.dstState = t.srcState
                cnt = cnt + 1

        for s in self.states:
            if s.isGoal == False:
                continue

            for t in s.transitions:
                if t.srcState != t.dstState:
                    t.dstState = t.srcState

        self.makeTransitionsDict()

        print("Number of transitions become self-loops:" + str(cnt))

        cnt = 0
        for t in self.transitions:
            if t.srcState.isGoal:
                if t.srcState != t.dstState:
                    cnt = cnt + 1

        print("Number of remained transitions from goal states:" + str(cnt))

    def checkTransitionFunction(self):
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
                        notFixed += "Could not fix error for: state=" + s.name + ", isGoal=" + str(
                            s.isGoal) + ", action=" + str(a) + ", prob=" + str(sum) + ", numOfTrans=" + str(
                            numOfTrans) + ". " + str(trans) + "\n"

        if notFixed != "":
            print(notFixed)
            raise Exception(notFixed)
        return ok

    """
    It checks whether
    \Sigma_{o \in observations} observationFunction[o][s][a] = 1
    """

    def checkObservationFunction(self):
        ok = True
        for a in self.actions:
            for s in self.states:
                sumProb = 0
                for o in self.observations:
                    sumProb += self.observationFunction[o][s][a]
                if sumProb != 1:
                    ok = False
                    print("The sum of observations for s=" + str(
                        s) + " and a=" + a + " does not sum to 1; it sums to " + str(sumProb))

        return ok

    def makeBeliefTree(self, H):

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

    def getPOMDPX_XML(self):

        #         transFunOk = self.checkTransitionFunction()
        #         print("Checking transition function")
        #         if transFunOk == False:
        #             print("Transition function is not a distribution")
        #         else:
        #             print("Transition function is fine")
        #
        #         obserFunOk = self.checkObservationFunction()
        #         print("Checking observation function")
        #         if obserFunOk == False:
        #             print("Observation function is not a distribution")
        #         else:
        #             print("Observation function is fine")

        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            # if s.reachable == False:
            #        continue
            st += s.name + " "
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        st += "<ObsVar vname='prediction_result'>" + "\n"
        st += "<ValueEnum>correct wrong goal_reached</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        if self.hasEvidence:
            st += "<ObsVar vname='evidence'>" + "\n"
            st += "<ValueEnum>"
            for ev in self.evidenceList:
                st = st + ev + " "
            st += "</ValueEnum>" + "\n"
            st += "</ObsVar>" + "\n"

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initialState)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            # if t.dstState.reachable == False:
            #    continue
            if t.srcState.isGoal == True:  ## Added for testing
                print("Not making a transition for the goal state " + t.srcState.name)
                continue
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.srcState.name + " " + t.dstState.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        for a in self.actions:
            for s in self.states:
                if s.isGoal == False:
                    continue
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + s.name + " " + s.name + "</Instance>" + "\n"
                st += "<ProbTable>1</ProbTable>" + "\n"
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>prediction_result</Var>" + "\n"
        st += "<Parent>predicted_event state_1</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            for a in self.actions:
                # if self.states[i].reachable == False:
                #    continue
                if self.states[i].isGoal:

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                    st += "<ProbTable>1</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                    st += "<ProbTable>0</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                    st += "<ProbTable>0</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"
                else:
                    if a in self.states[i].anchor[1].events:
                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                        st += "<ProbTable>1</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"
                    else:
                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                        st += "<ProbTable>1</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        # st += "</ObsFunction>" + "\n"

        if self.hasEvidence:
            # st += "<ObsFunction>" + "\n"
            st += "<CondProb>" + "\n"
            st += "<Var>evidence</Var>" + "\n"
            st += "<Parent>state_1</Parent>" + "\n"
            st += "<Parameter type = 'TBL'>" + "\n"
            for s in self.states:
                # if s.reachable == False:
                #    continue
                for j in range(len(self.evidenceList)):
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + s.name + " " + self.evidenceList[j] + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(s.evidence_distribution[j]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

            st += "</Parameter>" + "\n"
            st += "</CondProb>" + "\n"
            # st += "</ObsFunction>" + "\n"

        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            # if self.states[i].reachable == False:
            #    continue
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].isGoal:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>-1</ValueTable>" + "\n"
                # if self.states[i].isGoal == False:
                #    st += "<ValueTable>-1</ValueTable>"+"\n"                    
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    """
    In this version, the observation function of the pomdpx comes directly from the field observationFunction of this object
    """

    def getPOMDPX_XML2(self):

        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            # if s.reachable == False:
            #        continue
            st += s.name + " "
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        observationTitle = "predictionResult"
        if len(self.evidenceList) > 0:
            observationTitle += "Evidence"

        st += "<ObsVar vname='" + observationTitle + "'>" + "\n"
        st += "<ValueEnum>" + "\n"
        for o in self.observations:
            st += self.getObservationString(o) + " "
        st += "</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initialState)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            # if t.dstState.reachable == False:
            #    continue
            if t.srcState.isGoal == True:  ## Added for testing
                continue
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.srcState.name + " " + t.dstState.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        for a in self.actions:
            for s in self.states:
                if s.isGoal == False:
                    continue
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + s.name + " " + s.name + "</Instance>" + "\n"
                st += "<ProbTable>1</ProbTable>" + "\n"
                st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>" + observationTitle + "</Var>" + "\n"
        st += "<Parent>predicted_event state_1</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for s in self.states:
            for a in self.actions:
                # if self.states[i].reachable == False:
                #    continue
                for o in self.observations:
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + s.name + " " + self.getObservationString(
                        o) + " " + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(self.observationFunction[o][s][a]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            # if self.states[i].reachable == False:
            #    continue
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].isGoal:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>-1</ValueTable>" + "\n"
                # if self.states[i].isGoal == False:
                #    st += "<ValueTable>-1</ValueTable>"+"\n"                    
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    def getPOMDPX_4_EXMDP_XML(self):
        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.99999999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            st += s.name + " "
        # st += "trapping "    # adding a trapping state
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        st += "<ObsVar vname='prediction_result_state'>" + "\n"
        st += "<ValueEnum>"
        for ob in self.observations:
            st += str(ob).replace(", ", "_").replace("(", "").replace(")", "") + " "
        st += "</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        """
        if self.has_evidence:
            st += "<ObsVar vname='evidence'>"+"\n"
            st += "<ValueEnum>"
            for ev in self.evidence_list:
                st = st + ev + " " 
            st += "</ValueEnum>"+"\n"
            st += "</ObsVar>"+"\n"
        """

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initialState)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        # st += "0 "      # initial distribution entry for the trapping state
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.srcState.name + " " + t.dstState.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>prediction_result_state</Var>" + "\n"
        st += "<Parent>state_1 predicted_event</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for x in self.states:
            for e in self.actions:
                for o in self.observations:
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + x.name + " " + e + " " + str(o).replace(", ", "_").replace("(", "").replace(
                        ")", "") + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(self.observationFunction[o][x][e]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"

        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].isGoal:
                    st += "<ValueTable>1</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    def addShadowStatesForGoalStates(self):
        for s in self.states:
            if s.isGoal == False:
                continue
            s2 = MDPState(s.name + "_shadow", s)
            self.addState(s2)
            for o in self.observations:
                self.observationFunction[o][s2] = {}
                for a in self.actions:
                    self.observationFunction[o][s2][a] = self.observationFunction[o][s][a]

        for t in self.transitions:
            if t.srcState.isGoal == False:
                continue
            if t.dstState.isGoal == False:
                continue
            srcStateShadow = self.getStateByAnchor(t.srcState)
            dstStateShadow = self.getStateByAnchor(t.dstState)
            t2 = MDPTransition(srcStateShadow, dstStateShadow, t.action, t.natureAction, t.probability)
            self.addTransition(t2)

        for t in self.transitions:
            if t.srcState.isGoal == False:
                continue
            dstStateShadow = self.getStateByAnchor(t.dstState)
            t.dstState = dstStateShadow

    def getPreferenceValueBooleanVector(self, booleanVector):
        m = len(booleanVector)
        val = 0.0
        for j in range(m):
            val += math.pow(2, m - j) if booleanVector[j] == True else 0.0
        return val

    def getPreferenceValueVector(self, vect):
        m = len(vect)
        val = 0.0
        # print("vect="+str(vect))
        for j in range(m):
            val += math.pow(2, m - j) * vect[j]
        return val

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

    def roundValueVects(self, valueVect, digits):
        result = []
        for t in valueVect:
            print("t: " + str(t))
            print("t[0]: " + str(t[0]))
            lng = round(t, digits)
            softs = []
            for num in t[1]:
                softs.append(round(num, digits))
            tp = (lng, softs)
            print("tp: " + str(tp))
            result.append(tp)
        return result

    def getOrder4ComputingPolicy(self):
        queue = []
        for s in self.goalStates:
            for t in s.transitionsTo:
                if t.srcState not in queue and (not t.srcState.isGoal):
                    queue.append(t.srcState)
        i = 0
        while i < len(queue):
            s = queue[i]
            for t in s.transitionsTo:
                if t.srcState not in queue and (not t.srcState.isGoal):
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
            if s.isGoal:
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
                if s.isGoal:
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

    def arePolicisEqual(self, policy1, policy2):
        for i in range(len(self.states)):
            if str(policy1[i]) != str(policy2[i]):
                return False
        return True

    def isDominatedByVectorNumStepsAndViolCost(self, valuePolicy1, valuePolicy2):

        if valuePolicy2[0] <= valuePolicy1[0] and self.isDominated(valuePolicy1[1], valuePolicy2[1]):
            return True
        elif valuePolicy2[0] < valuePolicy1[0] and (
                self.isDominated(valuePolicy1[1], valuePolicy2[1]) or self.isVectsEqual(valuePolicy1[1],
                                                                                        valuePolicy2[1])):
            return True
        #                 elif a_vect2[1][0] == a_vect[1][0] and self.isVectsEqual(a_vect[1][1], a_vect2[1][1]) and a_vect2[0] < a_vect[0]:
        #                     dominated = True
        #                     break

        return False

    def isPolicyDominated(self, policy1, policy2):
        for i in range(len(self.states)):
            if self.isDominatedByVectorNumStepsAndViolCost(policy1[i], policy2[i]):
                return True
        return False

    def selectARandomPolicy(self, V_O, V_O_Actions):
        policy = [None] * len(self.states)
        for s in self.states:
            if s.isGoal:
                policy[s.index] = V_O_Actions[s.name]
            else:
                policy[s.index] = random.choice(V_O_Actions[s.name])
        policy_values = self.computeNumstepsAndSatisProbOfAPolicy(self.states, list(policy), V_O)
        return (policy, policy_values)

    def recomputeParetoPoliciesValueFunctions(self, V_O, V_O_Actions):
        paretoPolicy = [[] for _ in range(len(self.states))]
        for s in self.states:
            print("s.index: " + str(s.index))
            paretoPolicy[s.index] = list(set(V_O_Actions[s.name]))
        print("paretoPolicy: " + str(paretoPolicy))
        # print("num of policies within the pareto: "+str(len(itertools.product(*paretoPolicy))))
        num = 1
        policies = []
        policyValues = []
        print("")
        for policy in itertools.product(*paretoPolicy):
            print(f"{num}. policy: {list(policy)}")
            # print("policy: "+str(list(policy)))
            policies.append(list(policy))
            policyValue = self.computeNumstepsAndSatisProbOfAPolicy(self.states, list(policy), V_O)
            policyValues.append(policyValue)
            num += 1

        numEqPolicies = 0
        numOfDominated = 0
        for i in range(len(policies)):
            for j in range(len(policies)):
                if i == j:
                    continue
            if self.arePolicisEqual(policies[i], policies[j]):
                numEqPolicies += 1
            if self.isPolicyDominated(policies[i], policies[j]):
                numOfDominated += 1

        print("number of polices for which values are recomputed: " + str(num))
        print("Number of policies that are equal: " + str(numEqPolicies))
        print("Number of dominated policies is: " + str(numOfDominated))

    def arePoliciesConsistent(self, policies):
        return False

    def getValueOfPreferences(self, satisProbs, numOfDigits):
        n = len(satisProbs)

    def getConvexHull_old(self, valueVectors):
        valueVectors = self.sortBasedOnViolationCost(valueVectors)
        # hull = self.convexHull(valueVectors, len(valueVectors))
        hull = self.convex_hull(valueVectors)
        # print("hull="+str(hull))
        newValVectors = []
        for tple in hull:
            newTple = (tple[0], tple[1])
            newValVectors.append(newTple)
        return newValVectors

    def getConvexHull(self, valueVectors, baseNum4CostLiniarization=10):
        valueVectors = self.compViolationCostsAndConct(valueVectors, baseNum4CostLiniarization)
        # hull = self.convexHull(valueVectors, len(valueVectors))
        hull = self.convex_hull(valueVectors)
        # print("hull="+str(hull))
        newValVectors = []
        for tple in hull:
            newTple = (tple[0], tple[1])
            newValVectors.append(newTple)
        return newValVectors

    def compViolationCostsAndConct(self, valueVectors, baseNum4CostLiniarization):
        for i in range(len(valueVectors)):
            violationCost = 0.0
            m = len(self.goalStates[0].weightBVector)
            for j in range(m):
                violationCost += (1 - valueVectors[i][1][j]) * math.pow(baseNum4CostLiniarization, m - j - 1)
            valueVectors[i] = valueVectors[i] + (violationCost,)
        return valueVectors

    def sortBasedOnViolationCost(self, valueVectors):
        n = len(valueVectors)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                # if self.isDominated(valueVectors[j + 1], valueVectors[j]):
                # print("valueVectors[j]: " + str(valueVectors[j]))
                if self.isDominated(valueVectors[j + 1][1], valueVectors[j][1]):
                    # if valueVectors[j] > valueVectors[j + 1] :
                    valueVectors[j], valueVectors[j + 1] = valueVectors[j + 1], valueVectors[j]

        for i in range(len(valueVectors)):
            valueVectors[i] = valueVectors[i] + (i,)

        # print("sorted items="+str(valueVectors))

        return valueVectors

    def convex_hull(self, points):
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (x, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        # points = sorted(set(points))

        """
        Written by Hazhar
        """
        points = sorted(points, key=lambda k: [k[0], k[2]])

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(points) <= 1:
            return points

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[0] - o[0])

        # Build lower hull 
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list. 
        return lower[:-1] + upper[:-1]

    def mergeTwoConvexHulls_TwoNextStates(self, tran1, tran2, hull1, hull2, applyLimitedPrecision, precision):
        m = len(self.goalStates[0].weightBVector)
        Q_L = []
        for q1 in hull1:
            for q2 in hull2:
                steps = 1
                soft_probs = [0] * m

                steps += q1[0] * tran1.probability
                for k in range(m):
                    soft_probs[k] += q1[1][k] * tran1.probability

                steps += q2[0] * tran2.probability
                for k in range(m):
                    soft_probs[k] += q2[1][k] * tran2.probability

                tple = (steps, soft_probs)

                if applyLimitedPrecision:
                    tple = self.roundValueVect(tple, precision)
                    if not self.containtsValTuple(Q_L, tple):
                        Q_L.append(tple)
                else:
                    Q_L.append(tple)

        Q_L = self.getConvexHull(Q_L)
        return Q_L

    def mergeSumConvexHullandNextStateConvexHull(self, tran2, hull1, hull2, applyLimitedPrecision, precision):
        m = len(self.goalStates[0].weightBVector)
        Q_L = []
        # print(f"-------mergeSumConvexHullandNextStateConvexHull--------------")
        # print(f"tran2: {tran2}")
        # print(f"hull1: {hull1}")
        # print(f"hull2: {hull2}")
        for q1 in hull1:
            for q2 in hull2:
                steps = q1[0]
                soft_probs = deepcopy(q1[1])
                # print(f"q1: {q1}")
                # print(f"q2: {q2}")
                # print(f"soft_probs: {soft_probs}")

                steps += q2[0] * tran2.probability
                for k in range(m):
                    # print(f"before soft_probs[{k}]: {soft_probs[k]}")
                    soft_probs[k] += q2[1][k] * tran2.probability
                    # print(f"soft_probs[{k}]: {soft_probs[k]}")
                    # print(f"q2[1][{k}]: {q2[1][k]}")
                    # print(f"tran2: {tran2}")
                    if (soft_probs[k] >= 1):
                        if soft_probs[k] < 1.001:
                            soft_probs[k] = 1
                            print(f" Fixed soft_probs[{k}] from {soft_probs[k]} to 1")
                        else:
                            raise Exception(f"Sum of soft_probs[{k}] is greate than 1 and it is {soft_probs[k]}")

                tple = (steps, soft_probs)

                if applyLimitedPrecision:
                    tple = self.roundValueVect(tple, precision)
                    if not self.containtsValTuple(Q_L, tple):
                        Q_L.append(tple)
                else:
                    Q_L.append(tple)

        Q_L = self.getConvexHull(Q_L)
        # print("Q_L: {Q_L}")
        return Q_L

    def QDValuesDifferLess(self, Q_D1, Q_D2, epsilonExpcNumSteps, epsilonSoftConstSatis):
        m = len(self.goalStates[0].weightBVector)

        Q_1 = []
        for vec in Q_D1:
            violationCost = 0.0
            for j in range(m):
                violationCost += (1 - vec[1][j]) * math.pow(self.baseNum4CostLiniarization, m - j - 1)
            vec2 = (vec[0], vec[1], violationCost)
            Q_1.append(vec2)
        Q_1 = sorted(Q_1, key=lambda k: [k[0], k[1]])

        Q_2 = []
        for vec in Q_D2:
            violationCost = 0.0
            for j in range(m):
                violationCost += (1 - vec[1][j]) * math.pow(self.baseNum4CostLiniarization, m - j - 1)
            vec2 = (vec[0], vec[1], violationCost)
            Q_2.append(vec2)
        Q_2 = sorted(Q_2, key=lambda k: [k[0], k[2]])

        for i in range(len(Q_1)):
            vec1 = Q_1[i]
            vec2 = Q_2[i]
            if abs(vec1[0] - vec2[0]) > epsilonExpcNumSteps:
                return False
            for j in range(m):
                if abs(vec1[1][j] - vec2[1][j]) > epsilonSoftConstSatis:
                    return False
        return True

    def paretoOptimalPolicies_InfiniteHorizon_ConvexHullValueIteration2(self, epsilonExpcNumSteps=0.01,
                                                                        epsilonSoftConstSatis=0.001, printPolicy=True,
                                                                        printPolicy2File=True,
                                                                        fileName2Output="ParetoOptimalPolcies.txt",
                                                                        compAvoidActions=False,
                                                                        applyLimitedPrecision=False, precision=3,
                                                                        maxValsPerState=-1, chooseRandomPolicy=False,
                                                                        choosePoliciesbaseOnWeights=True, weights=[]):
        if self.verbose == True:
            print(
                "------computing pareto optimal policy for infinite horizon using convex hull value iteration--------------")
        n = len(self.states)

        time_start = time.time()

        if compAvoidActions == True:
            # if self.availableActionsComputed == False:
            self.computeAvoidableActions(True)

        m = len(self.goalStates[0].weightBVector)

        V_O = {}
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.isGoal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            V_O[s.name] = [vect]

        V_O_Actions = {}
        for s in self.states:
            if s.isGoal:
                V_O_Actions[s.name] = ["STOP"]
            else:
                V_O_Actions[s.name] = []

        # Set of Q-vectors which might contain the ones that are dominated. 
        Q_D = {}
        Q_D_Before = {}
        for s in self.states:
            Q_D[s.name] = {}
            Q_D_Before[s.name] = {}
            for a in self.actions:
                Q_D[s.name][a] = []
                Q_D_Before[s.name][a] = []
                softConstProbVect = [0] * m
                if s.isGoal:
                    for k in range(m):
                        softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
                Q_D[s.name][a].append(vect)
                Q_D_Before[s.name][a].append(vect)

        maxCnt = 120
        cnt = 1
        stateQueue = self.getOrder4ComputingPolicy()
        if self.initialSCC == None:
            self.decomposeToStrngConnComponents()
            self.computeSCCTopologicalOrder()

        useConvergance = True

        for scc in self.sccTopologicalOrder:
            cnt = 1
            # while cnt <= maxCnt:
            converged = False
            # while not converged:
            # while cnt <= maxCnt:
            while (not converged or not useConvergance) and cnt <= maxCnt:

                print(
                    f"Computing Q_D values of states within SCC {scc.name}, which has {len(scc.states)} states, in the {cnt}'th iteration")

                # for a in s.availableActions:
                #    Q_D[s.name][a] = []
                for s in scc.states:
                    for a in s.availableActions:
                        Q_D_Before[s.name][a] = Q_D[s.name][a]

                for s in scc.states:
                    if s.isGoal:
                        continue
                    if s.reachable == False:
                        continue

                    if compAvoidActions and not s.aGoalIsReachable:
                        # print(f"State {s.name} is a dead-end state")
                        continue

                    # if computeAvoidableActions == True and s.aGoalIsReachable == False:
                    #    continue

                    for a in s.availableActions:

                        # print(f"Computing Q_D[{s.name}][{a}]")

                        if compAvoidActions and a in s.avoidActions:
                            # print(f"Action {a} should be avoided at state {s.name}")
                            continue

                            # if computeAvoidableActions == True:
                        #    if a in s.avoidActions:
                        #        continue

                        # Q_D_Before[s.name][a] = Q_D[s.name][a]
                        # print("s: "+s.name+", a: "+a)
                        """
                        First obtain the successor states of s by a, then obtain a list containing a list
                        of Pareto optimal values for each of those successor states
                        """
                        # print(f"----------------s: {s.name}, a:{a}----------------------")
                        Q_D_s2_s = {}
                        maxNumOfVals = 1
                        for tran in s.actionsTransitions[a]:
                            s2 = tran.dstState
                            Q_D_s2_s[s2.name] = []
                            for a2 in s2.availableActions:
                                # if computeAvoidableActions and a2 in s2.avoidActions:
                                #    continue
                                # print(f"Q_D_Before[{s2.name}][{a2}]: {Q_D_Before[s2.name][a2]}")
                                for vc in Q_D_Before[s2.name][a2]:
                                    if vc not in Q_D_s2_s[s2.name]:
                                        Q_D_s2_s[s2.name].append(vc)
                            Q_D_s2_s[s2.name] = self.getConvexHull(Q_D_s2_s[s2.name])
                            maxNumOfVals = maxNumOfVals * len(Q_D_s2_s[s2.name])
                            # print("Q_D_s2_s["+s2.name+"]: "+str(Q_D_s2_s[s2.name]))
                            # if s2.name.find("q5") > -1:
                            #    return

                        n2 = len(s.actionsTransitions[a])

                        '''
                        Determine if there is any need for sampling or not 
                        '''

                        '''
                        if maxValsPerState > 0 and maxNumOfVals > maxValsPerState:
                            print(f"Maximum value per states has reached for Q[{s.name}][{a}] and it is {maxNumOfVals}")
                            sizes = [0]*n2
                            for j in range(0, n2):
                                tran = s.actionsTransitions[a][j]
                                s2 = tran.dstState
                                sizes[j] = len(Q_D_s2_s[s2.name])
                            print(f"Sizes before sampling: {sizes}")
                            stopDivision = False
                            while not stopDivision:
                                numOfValues = 1
                                for j in range(0, n2):
                                    sizes[j] = math.ceil(sizes[j]/2)
                                    numOfValues = numOfValues *sizes[j]
                                if  numOfValues <= maxValsPerState:
                                    stopDivision = True
                            print(f"Sizes after sampling: {sizes}")
                            for j in range(0, n2):
                                tran = s.actionsTransitions[a][j]
                                s2 = tran.dstState
                                Q_D_s2_s[s2.name] = sample(Q_D_s2_s[s2.name], sizes[j])
                                
                        '''

                        Q_Hull = []
                        if n2 > 0:
                            tran = s.actionsTransitions[a][0]
                            s2 = tran.dstState
                            for q in Q_D_s2_s[s2.name]:
                                q2 = deepcopy(q)
                                q_ls = list(q2)
                                q_ls[0] = q_ls[0] * tran.probability
                                for k in range(m):
                                    q_ls[1][k] = q_ls[1][k] * tran.probability
                                Q_Hull.append(tuple(q_ls))
                            # print(f"Q_Hull[{s2.name}]: {Q_Hull}")

                        #                         for tran in s.actionsTransitions[a]:
                        #                             s2 = tran.dstState
                        #                             for a2 in s2.availableActions:
                        #                                 print(f"Q_D_Before[{s2.name}][{a2}]: {Q_D_Before[s2.name][a2]}")

                        # print("s.name: "+s.name)
                        # print("a:"+str(a))
                        # print("tran:"+str(tran))
                        # print("Q_Hull["+s2.name+"]: "+str(Q_Hull))

                        for j in range(1, n2):
                            tran = s.actionsTransitions[a][j]
                            s2 = tran.dstState
                            Q_Hull2 = Q_D_s2_s[s2.name]
                            # print("tran:"+str(tran))
                            # print("Q_Hull2["+s2.name+"]: "+str(Q_Hull2))
                            Q_Hull = self.mergeSumConvexHullandNextStateConvexHull(tran, Q_Hull, Q_Hull2,
                                                                                   applyLimitedPrecision, precision)

                        Q_Hull_Temp = Q_Hull.copy()
                        Q_Hull.clear()

                        for q in Q_Hull_Temp:
                            q_ls = list(q)
                            q_ls[0] = q_ls[0] + 1
                            q2 = tuple(q_ls)
                            Q_Hull.append(q2)

                        '''
                        
                        '''
                        Q_Hull = self.getNonDominantTimeViolationCostVectors(Q_Hull)

                        '''
                        Determine if there is any need for sampling
                        '''
                        if maxValsPerState > -1 and len(Q_Hull) > maxValsPerState:
                            print(f"Reduced the number of vals from {len(Q_Hull)} to {maxValsPerState}")
                            New_Q_Hull = []
                            New_Q_Hull = random.sample(Q_Hull, maxValsPerState)
                            Q_Hull = New_Q_Hull

                        Q_D[s.name][a] = Q_Hull
                        # if cnt == maxCnt:
                        #    print("Q_D["+str(s.name)+"]["+str(a)+"]:"+str(Q_D[s.name][a]))
                        # print(f"Q_D_Before[{s2.name}][{a2}]: {Q_D_Before[s2.name][a2]}")

                if useConvergance:
                    converged = True
                    for s in scc.states:
                        if converged == False:
                            break
                        for a in s.availableActions:
                            if len(Q_D_Before[s.name][a]) != len(Q_D[s.name][a]):
                                converged = False
                                print(f"len(Q_D_Before[{s.name}][{a}]): {len(Q_D_Before[s.name][a])}")
                                print(f"len(Q_D[{s.name}][{a}]): {len(Q_D[s.name][a])}")
                                break

                    for s in scc.states:
                        if converged == False:
                            break
                        for a in s.availableActions:
                            if not self.QDValuesDifferLess(Q_D_Before[s.name][a], Q_D[s.name][a], epsilonExpcNumSteps,
                                                           epsilonSoftConstSatis):
                                converged = False
                                print(f"Q_D_Before[{s.name}][{a}]: {Q_D_Before[s.name][a]}")
                                print(f"Q_D[{s.name}][{a}]: {Q_D[s.name][a]}")
                                break

                cnt += 1

            for s in scc.states:
                if s.isGoal:
                    continue
                Q_Ds = []
                for a in self.actions:
                    if a not in s.availableActions:
                        continue
                    for vect in Q_D[s.name][a]:
                        tple = (a, vect)
                        Q_Ds.append(tple)
                    # print("Q_DS for "+s.name+": "+str(Q_Ds))
                nonDominants = self.getNonDominantActionsAndVectors(Q_Ds)
                V_O[s.name] = []
                V_O_Actions[s.name] = []
                # print("nonDominants: "+str(nonDominants))
                for action_vect in nonDominants:
                    V_O[s.name].append(action_vect[1])
                    V_O_Actions[s.name].append(action_vect[0])
                # if cnt == maxCnt or (converged == True):
                if (converged == True and useConvergance == True) or cnt >= maxCnt:
                    # print("Update V_O["+s.name+"]="+str(V_O[s.name]))
                    # print("Update V_O_Actions["+s.name+"]="+str(V_O_Actions[s.name]))

                    print("Q_D[" + str(s.name) + "][" + str(a) + "]:" + str(Q_D[s.name][a]))

                    f = open(fileName2Output, "a")
                    f.seek(0)
                    f.write("V_O[" + s.name + "]=" + str(V_O[s.name]) + "\n")
                    # f.write("Update V_O["+s.name+"]="+str(self.roundValueVect(V_O[s.name], 3)))
                    f.write("V_O_Actions[" + s.name + "]=" + str(V_O_Actions[s.name]) + "\n")
                    f.close()

            print(f"Number of iterations for SCC {scc.name}: {cnt}")

            # self.recomputeParetoPoliciesValueFunctions(V_O, V_O_Actions)

        time_elapsed = (time.time() - time_start)

        if chooseRandomPolicy == True:
            policy, policyvalues = self.selectARandomPolicy(V_O, V_O_Actions)
            print(f"Selected random policy: {policy}")
            print(f"Values of the selected policy: {policyvalues}")
            return (policy, policyvalues)
        if choosePoliciesbaseOnWeights == True:
            policies = self.chooseBestPolicesBaseOnWeights(Q_D, weights, precision)
            policyValues = []
            for policy in policies:
                print(f"computed policy for weights = {weights}")
                print(policy)
                policyValues.append(self.computeNumstepsAndSatisProbOfAPolicy(self.states, policy, [], True))
            expectedVectCosts = []
            for policyVal in policyValues:
                expectedVectCosts.append(policyVal[self.initialState.name])
            policycomptime = time_elapsed
            return policies, policyValues, expectedVectCosts, policycomptime

    def chooseBestPolicesBaseOnWeights(self, Q_D, weights, numOfDigits):
        Q = {}
        m = len(self.goalStates[0].weightBVector)
        policies = []
        for weightVect in weights:
            # if len(weightVect) == 2 and weightVect[0] == 0 and weightVect[1] == 1:
            #    weightVect[0] == 0.00000001
            #    weightVect[1] == 0.99999999
            # print("success")
            # raise Exception("")
            for s in self.states:
                if s.isGoal:
                    continue
                Q[s.name] = {}
                for a in s.availableActions:
                    if self.availableActionsComputed and (a in s.avoidActions):
                        continue
                    minCost = float_info.max
                    for i in range(len(Q_D[s.name][a])):
                        valVect = Q_D[s.name][a][i]
                        violationCost = 0
                        for j in range(m):
                            violationCost += (1 - valVect[1][j]) * math.pow(10, m - j - 1)
                        cost = weightVect[0] * valVect[0] + weightVect[1] * violationCost * 3
                        # violationCost = violationCost*10
                        if cost < minCost:
                            minCost = cost
                    Q[s.name][a] = minCost

            P = {}
            C = {}
            for s in self.states:
                if s.isGoal:
                    P[s.name] = "STOP"
                    C[s.name] = 0
                    continue
                minCost = float_info.max
                for a in s.availableActions:
                    if self.availableActionsComputed and (a in s.avoidActions):
                        continue
                    if Q[s.name][a] < minCost:
                        minCost = Q[s.name][a]
                        P[s.name] = a
                        C[s.name] = minCost
                    elif Q[s.name][a] == minCost and a[1] != self.special_noEvent and P[s.name][
                        1] == self.special_noEvent:
                        P[s.name] = a
                        C[s.name] = minCost

            policies.append(P)
            print(f"C: {C}")
        return policies

    def paretoOptimalPolicies_InfiniteHorizon_ConvexHullValueIteration(self, epsilonOfConvergance=0.01,
                                                                       printPolicy=True, printPolicy2File=True,
                                                                       fileName2Output="ParetoOptimalPolcies.txt",
                                                                       computeAvoidableActions=False,
                                                                       applyLimitedPrecision=False, precision=3):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)

        if computeAvoidableActions == True:
            self.computeAvoidableActions()

        m = len(self.goalStates[0].weightBVector)

        V_O = {}
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.isGoal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            V_O[s.name] = [vect]

        V_O_Actions = {}
        for s in self.states:
            if s.isGoal:
                V_O_Actions[s] = ["STOP"]
            else:
                V_O_Actions[s] = []

        # Set of Q-vectors which might contain the ones that are dominated. 
        Q_D = {}
        for s in self.states:
            Q_D[s.name] = {}
            for a in self.actions:
                Q_D[s.name][a] = []
                softConstProbVect = [0] * m
                if s.isGoal:
                    for k in range(m):
                        softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
                Q_D[s.name][a].append(vect)

        maxCnt = 120
        cnt = 1
        stateQueue = self.getOrder4ComputingPolicy()
        if self.initialSCC == None:
            self.decomposeToStrngConnComponents()
            self.computeSCCTopologicalOrder()

        for scc in self.sccTopologicalOrder:
            cnt = 1
            while cnt <= maxCnt:

                for s in scc.states:
                    if s.isGoal:
                        continue
                    # for a in s.availableActions:
                    #    Q_D[s.name][a] = []

                for s in scc.states:
                    if s.isGoal:
                        continue
                    for a in s.availableActions:
                        # print("s: "+s.name+", a: "+a)
                        """
                        First obtain the successor states of s by a, then obtain a list containing a list
                        of Pareto optimal values for each of those successor states
                        """
                        Q_D_s2_s = {}
                        for tran in s.actionsTransitions[a]:
                            s2 = tran.dstState
                            Q_D_s2_s[s2.name] = []
                            for a2 in s2.availableActions:
                                for vc in Q_D[s2.name][a2]:
                                    Q_D_s2_s[s2.name].append(vc)
                            Q_D_s2_s[s2.name] = self.getConvexHull(Q_D_s2_s[s2.name])
                            # print("Q_D_s2_s["+s2.name+"]: "+str(Q_D_s2_s[s2.name]))
                            # if s2.name.find("q5") > -1:
                            #    return

                        n2 = len(s.actionsTransitions[a])
                        V_S2s = []
                        indices = []

                        for j in range(n2):
                            tran = s.actionsTransitions[a][j]
                            s2 = tran.dstState
                            V_S2s.append(Q_D_s2_s[s2.name])
                            indices.append([k for k in range(len(Q_D_s2_s[s2.name]))])

                        Q_D[s.name][a] = []

                        for indexList in itertools.product(*indices):

                            Vlist = []
                            for k in range(n2):
                                Vlist.append(V_S2s[k][indexList[k]])
                                # print("Vlist: "+str(Vlist))
                            steps = 1
                            soft_probs = [0] * m
                            # print("n2: "+str(n2))
                            for j in range(n2):
                                tran = s.actionsTransitions[a][j]
                                s2 = tran.dstState
                                v_vect = Vlist[j]
                                # print("v_vect: "+str(v_vect))
                                steps += v_vect[0] * tran.probability
                                for k in range(m):
                                    soft_probs[k] += v_vect[1][k] * tran.probability
                                # print("soft_probs: "+str(soft_probs))
                            tple = (steps, soft_probs)

                            if applyLimitedPrecision:
                                tple = self.roundValueVect(tple, precision)
                                if not self.containtsValTuple(Q_D[s.name][a], tple):
                                    Q_D[s.name][a].append(tple)
                            # if tple not in Q_D[s.name][a]: 
                            else:
                                Q_D[s.name][a].append(tple)

                        Q_D[s.name][a] = self.getConvexHull(Q_D[s.name][a])
                        print("Q_D[" + str(s.name) + "][" + str(a) + "]:" + str(Q_D[s.name][a]))

                for s in scc.states:
                    if s.isGoal:
                        continue
                    Q_Ds = []
                    for a in self.actions:
                        for vect in Q_D[s.name][a]:
                            tple = (a, vect)
                            Q_Ds.append(tple)
                    # print("Q_DS for "+s.name+": "+str(Q_Ds))
                    nonDominants = self.getNonDominantActionsAndVectors(Q_Ds)
                    V_O[s.name] = []
                    V_O_Actions[s.name] = []
                    # print("nonDominants: "+str(nonDominants))
                    for action_vect in nonDominants:
                        V_O[s.name].append(action_vect[1])
                        V_O_Actions[s.name].append(action_vect[0])
                    if cnt == maxCnt:
                        # print("Update V_O["+s.name+"]="+str(V_O[s.name]))
                        # print("Update V_O_Actions["+s.name+"]="+str(V_O_Actions[s.name]))
                        f = open(fileName2Output, "a")
                        f.seek(0)
                        f.write("V_O[" + s.name + "]=" + str(V_O[s.name]) + "\n")
                        # f.write("Update V_O["+s.name+"]="+str(self.roundValueVect(V_O[s.name], 3)))
                        f.write("V_O_Actions[" + s.name + "]=" + str(V_O_Actions[s.name]) + "\n")
                        f.close()

                cnt += 1

    def paretoOptimalPolicies__PreferencePlanning_InfiniteHorizon(self, epsilonOfConvergance=0.01, printPolicy=True,
                                                                  printPolicy2File=True,
                                                                  fileName2Output="ParetoOptimalPolcies.txt",
                                                                  computeAvoidableActions=False,
                                                                  applyLimitedPrecision=False, precision=3):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)

        if computeAvoidableActions == True:
            self.computeAvoidableActions()

        m = len(self.goalStates[0].weightBVector)

        V_O = {}
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.isGoal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            V_O[s.name] = [vect]

        V_O_Actions = {}
        for s in self.states:
            if s.isGoal:
                V_O_Actions[s] = ["STOP"]
            else:
                V_O_Actions[s] = []

        # Set of Q-vectors which might contain the ones that are dominated. 
        Q_D = {}
        for s in self.states:
            Q_D[s.name] = {}
            for a in self.actions:
                Q_D[s.name][a] = []

        maxCnt = 20
        cnt = 1
        stateQueue = self.getOrder4ComputingPolicy()
        if self.initialSCC == None:
            self.decomposeToStrngConnComponents()
            self.computeSCCTopologicalOrder()

        for scc in self.sccTopologicalOrder:
            cnt = 1
            while cnt <= maxCnt:

                for s in scc.states:
                    if s.isGoal:
                        continue
                    for a in s.availableActions:
                        Q_D[s.name][a] = []

                for s in scc.states:
                    if s.isGoal:
                        continue
                    for a in s.availableActions:
                        # print("s: "+s.name+", a: "+a)
                        """
                        First obtain the successor states of s by a, then obtain a list containing a list
                        of Pareto optimal values for each of those successor states
                        """
                        n2 = len(s.actionsTransitions[a])
                        V_S2s = []
                        indices = []
                        # print("n2: "+str(n2))
                        # print("len(V_S2s): "+str(len(V_S2s)))
                        for j in range(n2):
                            tran = s.actionsTransitions[a][j]
                            # print("tran: "+str(tran))
                            s2 = tran.dstState
                            V_S2s.append(V_O[s2.name])
                            indices.append([k for k in range(len(V_O[s2.name]))])
                            # print("s2: "+s2.name)
                            # print("V_S2s[j] : "+str(V_S2s[j]))

                        for indexList in itertools.product(*indices):
                            # print("indexList: "+str(indexList));
                            Vlist = []
                            for k in range(n2):
                                Vlist.append(V_S2s[k][indexList[k]])
                                # print("Vlist: "+str(Vlist))
                            steps = 1
                            soft_probs = [0] * m
                            # print("n2: "+str(n2))
                            for j in range(n2):
                                tran = s.actionsTransitions[a][j]
                                s2 = tran.dstState
                                v_vect = Vlist[j]
                                # print("v_vect: "+str(v_vect))
                                steps += v_vect[0] * tran.probability
                                for k in range(m):
                                    soft_probs[k] += v_vect[1][k] * tran.probability
                                # print("soft_probs: "+str(soft_probs))
                            tple = (steps, soft_probs)
                            if applyLimitedPrecision:
                                tple = self.roundValueVect(tple, precision)
                                if not self.containtsValTuple(Q_D[s.name][a], tple):
                                    Q_D[s.name][a].append(tple)
                            # if tple not in Q_D[s.name][a]: 
                            else:
                                Q_D[s.name][a].append(tple)

                for s in scc.states:
                    if s.isGoal:
                        continue
                    Q_Ds = []
                    for a in self.actions:
                        for vect in Q_D[s.name][a]:
                            tple = (a, vect)
                            Q_Ds.append(tple)
                    # print("Q_DS for "+s.name+": "+str(Q_Ds))
                    nonDominants = self.getNonDominantActionsAndVectors(Q_Ds)
                    V_O[s.name] = []
                    V_O_Actions[s.name] = []
                    # print("nonDominants: "+str(nonDominants))
                    for action_vect in nonDominants:
                        V_O[s.name].append(action_vect[1])
                        V_O_Actions[s.name].append(action_vect[0])
                    if cnt == maxCnt:
                        # print("Update V_O["+s.name+"]="+str(V_O[s.name]))
                        # print("Update V_O_Actions["+s.name+"]="+str(V_O_Actions[s.name]))
                        f = open(fileName2Output, "a")
                        f.seek(0)
                        f.write("V_O[" + s.name + "]=" + str(V_O[s.name]) + "\n")
                        # f.write("Update V_O["+s.name+"]="+str(self.roundValueVect(V_O[s.name], 3)))
                        f.write("V_O_Actions[" + s.name + "]=" + str(V_O_Actions[s.name]) + "\n")
                        f.close()

                cnt += 1

    def paretoOptimalPolicies__PreferencePlanning_InfiniteHorizon2(self, epsilonOfConvergance=0.01, printPolicy=True,
                                                                   printPolicy2File=True,
                                                                   fileName2Output="ParetoOptimalPolcies.txt",
                                                                   computeAvoidableActions=False,
                                                                   applyLimitedPrecision=False, precision=3):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)

        if computeAvoidableActions == True:
            self.computeAvoidableActions()

        m = len(self.goalStates[0].weightBVector)

        V_O = {}
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.isGoal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            V_O[s.name] = [vect]

        V_O_Actions = {}
        for s in self.states:
            if s.isGoal:
                V_O_Actions[s.name] = ["STOP"]
            else:
                V_O_Actions[s.name] = []

        # Set of Q-vectors which might contain the ones that are dominated. 
        Q_D = {}
        for s in self.states:
            Q_D[s.name] = {}
            for a in self.actions:
                Q_D[s.name][a] = []

        maxCnt = 6
        cnt = 1
        stateQueue = self.getOrder4ComputingPolicy()
        while cnt <= maxCnt:

            for s in self.states:
                if s.isGoal:
                    continue
                for a in s.availableActions:
                    Q_D[s.name][a] = []

            for s in stateQueue:
                if s.isGoal:
                    continue
                for a in s.availableActions:
                    # print("s: "+s.name+", a: "+a)
                    """
                    First obtain the successor states of s by a, then obtain a list containing a list
                    of Pareto optimal values for each of those successor states
                    """
                    n2 = len(s.actionsTransitions[a])
                    V_S2s = []
                    indices = []
                    # print("n2: "+str(n2))
                    # print("len(V_S2s): "+str(len(V_S2s)))
                    for j in range(n2):
                        tran = s.actionsTransitions[a][j]
                        # print("tran: "+str(tran))
                        s2 = tran.dstState
                        V_S2s.append(V_O[s2.name])
                        indices.append([k for k in range(len(V_O[s2.name]))])
                        # print("s2: "+s2.name)
                        # print("V_S2s[j] : "+str(V_S2s[j]))

                    for indexList in itertools.product(*indices):
                        # print("indexList: "+str(indexList));
                        Vlist = []
                        for k in range(n2):
                            Vlist.append(V_S2s[k][indexList[k]])
                        # print("Vlist: "+str(Vlist))
                        steps = 1
                        soft_probs = [0] * m
                        # print("n2: "+str(n2))
                        for j in range(n2):
                            tran = s.actionsTransitions[a][j]
                            s2 = tran.dstState
                            v_vect = Vlist[j]
                            # print("v_vect: "+str(v_vect))
                            steps += v_vect[0] * tran.probability
                            for k in range(m):
                                soft_probs[k] += v_vect[1][k] * tran.probability
                            # print("soft_probs: "+str(soft_probs))
                        tple = (steps, soft_probs)
                        if applyLimitedPrecision:
                            tple = self.roundValueVect(tple, precision)
                            if not self.containtsValTuple(Q_D[s.name][a], tple):
                                Q_D[s.name][a].append(tple)
                        # if tple not in Q_D[s.name][a]:
                        else:
                            Q_D[s.name][a].append(tple)

                    # if cnt == maxCnt:
                    # print("Q_D["+s.name+"]["+str(a)+"]="+str(Q_D[s.name][a]))
                    # f = open(fileName2Output, "a")
                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str(round(Q_D[s.name][a], 3)))
                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str(self.roundValueVect(Q_D[s.name][a], 3)))

                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str([round(num, 3) for num in Q_D[s.name][a]]))
                    # f.close()

            for s in self.states:
                if s.isGoal:
                    continue
                Q_Ds = []
                for a in self.actions:
                    for vect in Q_D[s.name][a]:
                        tple = (a, vect)
                        Q_Ds.append(tple)
                # print("Q_DS for "+s.name+": "+str(Q_Ds))
                nonDominants = self.getNonDominantActionsAndVectors(Q_Ds)
                V_O[s.name] = []
                V_O_Actions[s.name] = []
                # print("nonDominants: "+str(nonDominants))
                for action_vect in nonDominants:
                    V_O[s.name].append(action_vect[1])
                    V_O_Actions[s.name].append(action_vect[0])
                if cnt == maxCnt:
                    # print("Update V_O["+s.name+"]="+str(V_O[s.name]))
                    # print("Update V_O_Actions["+s.name+"]="+str(V_O_Actions[s.name]))
                    f = open(fileName2Output, "a")
                    f.write("Update V_O[" + s.name + "]=" + str(V_O[s.name]) + "\n")
                    # f.write("Update V_O["+s.name+"]="+str(self.roundValueVect(V_O[s.name], 3)))
                    f.write("Update V_O_Actions[" + s.name + "]=" + str(V_O_Actions[s.name]) + "\n")
                    f.close()

            if cnt < maxCnt:
                max_actions_cnt = 0
                print("V_O_Actions: " + str(V_O_Actions))
                print("V_O_Actions[q5_s_l_w_l__q1_q1]: " + str(V_O_Actions['q5_s_l_w_l__q1_q1']))
                print("q5_s_l_w_l__q1_q1")
                for s in self.states:
                    action_cnt_list = [(x, V_O_Actions[s.name].count(x)) for x in V_O_Actions[s.name]]
                    print("action_cnt_list[" + s.name + "]: " + str(action_cnt_list))
                    for tpl in action_cnt_list:
                        if tpl[1] > max_actions_cnt:
                            max_actions_cnt = tpl[1]
                print("max_actions_cnt: " + str(max_actions_cnt))
                if max_actions_cnt >= 8:
                    self.recomputeParetoPoliciesValueFunctions(V_O, V_O_Actions)

            cnt += 1

    def paretoOptimalPolicies__Consistentcy_PreferencePlanning_InfiniteHorizon2(self, epsilonOfConvergance=0.01,
                                                                                printPolicy=True, printPolicy2File=True,
                                                                                fileName2Output="ParetoOptimalPolcies.txt",
                                                                                computeAvoidableActions=False,
                                                                                applyLimitedPrecision=False,
                                                                                precision=3):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)

        if computeAvoidableActions == True:
            self.computeAvoidableActions()

        m = len(self.goalStates[0].weightBVector)

        V_O = {}
        for s in self.states:
            softConstProbVect = [0] * m
            # vect = None
            if s.isGoal:
                for k in range(m):
                    softConstProbVect[k] = 1.0 if s.weightBVector[k] else 0.0
                vect = (0, softConstProbVect)
            else:
                vect = (0, softConstProbVect)
            V_O[s.name] = [vect]

        V_O_Actions = {}
        for s in self.states:
            if s.isGoal:
                V_O_Actions[s.name] = ["STOP"]
            else:
                V_O_Actions[s.name] = []

        V_O_Policies = {}
        for s in self.states:
            V_O_Policies[s.name] = {}

        # Set of Q-vectors which might contain the ones that are dominated. 
        Q_D = {}
        for s in self.states:
            Q_D[s.name] = {}
            for a in self.actions:
                Q_D[s.name][a] = []

        maxCnt = 6
        cnt = 1
        stateQueue = self.getOrder4ComputingPolicy()
        while cnt <= maxCnt:

            for s in self.states:
                if s.isGoal:
                    continue
                for a in s.availableActions:
                    Q_D[s.name][a] = []

            for s in stateQueue:
                if s.isGoal:
                    continue
                for a in s.availableActions:
                    # print("s: "+s.name+", a: "+a)
                    """
                    First obtain the successor states of s by a, then obtain a list containing a list
                    of Pareto optimal values for each of those successor states
                    """
                    n2 = len(s.actionsTransitions[a])
                    V_S2s = []
                    V_S2s_policies = []
                    indices = []
                    # print("n2: "+str(n2))
                    # print("len(V_S2s): "+str(len(V_S2s)))
                    for j in range(n2):
                        tran = s.actionsTransitions[a][j]
                        # print("tran: "+str(tran))
                        s2 = tran.dstState
                        V_S2s.append(V_O[s2.name])
                        V_S2s_policies.append(V_O_Policies[s2.name])
                        indices.append([k for k in range(len(V_O[s2.name]))])
                        # print("s2: "+s2.name)
                        # print("V_S2s[j] : "+str(V_S2s[j]))

                    for indexList in itertools.product(*indices):
                        # print("indexList: "+str(indexList));
                        Vlist = []
                        VList_Policies = []
                        for k in range(n2):
                            Vlist.append(V_S2s[k][indexList[k]])
                            VList_Policies.append(V_S2s_policies[k][indexList[k]])
                        # print("Vlist: "+str(Vlist))
                        steps = 1
                        soft_probs = [0] * m
                        # print("n2: "+str(n2))
                        for j in range(n2):
                            tran = s.actionsTransitions[a][j]
                            s2 = tran.dstState
                            v_vect = Vlist[j]
                            # print("v_vect: "+str(v_vect))
                            steps += v_vect[0] * tran.probability
                            for k in range(m):
                                soft_probs[k] += v_vect[1][k] * tran.probability
                            # print("soft_probs: "+str(soft_probs))
                        tple = (steps, soft_probs)
                        if applyLimitedPrecision:
                            tple = self.roundValueVect(tple, precision)
                            if not self.containtsValTuple(Q_D[s.name][a], tple):
                                Q_D[s.name][a].append(tple)
                        # if tple not in Q_D[s.name][a]:
                        else:
                            Q_D[s.name][a].append(tple)

                    # if cnt == maxCnt:
                    # print("Q_D["+s.name+"]["+str(a)+"]="+str(Q_D[s.name][a]))
                    # f = open(fileName2Output, "a")
                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str(round(Q_D[s.name][a], 3)))
                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str(self.roundValueVect(Q_D[s.name][a], 3)))

                    # f.write("Q_D["+s.name+"]["+str(a)+"]="+str([round(num, 3) for num in Q_D[s.name][a]]))
                    # f.close()

            for s in self.states:
                if s.isGoal:
                    continue
                Q_Ds = []
                for a in self.actions:
                    for vect in Q_D[s.name][a]:
                        tple = (a, vect)
                        Q_Ds.append(tple)
                # print("Q_DS for "+s.name+": "+str(Q_Ds))
                nonDominants = self.getNonDominantActionsAndVectors(Q_Ds)
                V_O[s.name] = []
                V_O_Actions[s.name] = []
                V_O_Policies[s.name] = {}
                # print("nonDominants: "+str(nonDominants))
                for action_vect in nonDominants:
                    V_O[s.name].append(action_vect[1])
                    V_O_Actions[s.name].append(action_vect[0])

                if cnt == maxCnt:
                    # print("Update V_O["+s.name+"]="+str(V_O[s.name]))
                    # print("Update V_O_Actions["+s.name+"]="+str(V_O_Actions[s.name]))
                    f = open(fileName2Output, "a")
                    f.write("Update V_O[" + s.name + "]=" + str(V_O[s.name]) + "\n")
                    # f.write("Update V_O["+s.name+"]="+str(self.roundValueVect(V_O[s.name], 3)))
                    f.write("Update V_O_Actions[" + s.name + "]=" + str(V_O_Actions[s.name]) + "\n")
                    f.close()

            if cnt < maxCnt:
                max_actions_cnt = 0
                print("V_O_Actions: " + str(V_O_Actions))
                print("V_O_Actions[q5_s_l_w_l__q1_q1]: " + str(V_O_Actions['q5_s_l_w_l__q1_q1']))
                print("q5_s_l_w_l__q1_q1")
                for s in self.states:
                    action_cnt_list = [(x, V_O_Actions[s.name].count(x)) for x in V_O_Actions[s.name]]
                    print("action_cnt_list[" + s.name + "]: " + str(action_cnt_list))
                    for tpl in action_cnt_list:
                        if tpl[1] > max_actions_cnt:
                            max_actions_cnt = tpl[1]
                print("max_actions_cnt: " + str(max_actions_cnt))
                if max_actions_cnt >= 8:
                    self.recomputeParetoPoliciesValueFunctions(V_O, V_O_Actions)

            cnt += 1

    def are_policies_consistent(self, policies, fromState, fromAction):
        for s in self.states:
            if s.isGoal:
                continue

    def optimalPolicy_PreferencePlanning_InfiniteHorizon(self, epsilonOfConvergance=0.0001, printPolicy=True,
                                                         fileName2Output="OptimalPolicies_MinimumViolation.txt",
                                                         computeAvoidableActions=False):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)

        if computeAvoidableActions == True:
            self.computeAvoidableActions()

        # for state in self.mdp.states:
        # state.computeAvailableActions()

        m = len(self.goalStates[0].weightBVector)

        G = [[[0.0 for m in range(m)] for j in [0, 1]] for i in range(n)]
        A = ["" for j in range(n)]

        for j in range(n):
            if (self.states[j].isGoal):
                for k in range(m):
                    G[j][0][k] = 1.0 if self.states[j].weightBVector[k] else 0.0
                    G[j][1][k] = 1.0 if self.states[j].weightBVector[k] else 0.0
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
                if self.states[j].isGoal == True:
                    continue

                if self.states[j].reachable == False:
                    continue

                if computeAvoidableActions and self.states[j].aGoalIsReachable == False:
                    continue

                maxVal = -1
                maxTerm = [0.0 for _ in range(m)]
                optAction = ""
                state = self.states[j]

                # print("r="+str(r))

                # for action in self.eventList:
                # print("#Available actions for "+str(state)+": "+str(len(state.availableActions)))

                if len(state.availableActions) == 0:
                    print(f"State {self.states[j].name} has no availableActions")
                    raise Exception("")

                for action in state.availableActions:
                    if computeAvoidableActions:
                        if action in state.avoidActions:
                            continue
                    val = 0.0
                    # for k in range(n):
                    #    term = G[k][1]*self.mdp.conditionalProbability(k, j, action)
                    #    val += term
                    term = [0.0 for _ in range(m)]
                    for tran in state.actionsTransitions[action]:
                        for k in range(m):
                            term[k] += G[tran.dstState.index][1][k] * tran.probability
                    val = self.getPreferenceValueVector(term)

                    # print("term = "+str(term))
                    # print("val="+str(val))
                    # r += 1
                    # print("r="+str(r))
                    if val > maxVal:
                        maxVal = val
                        maxTerm = term
                        optAction = action

                        # if minVal-G[j][0] > maxDif:
                    # maxDif = minVal-G[j][0]

                if j == 0:
                    dif = maxVal - self.getPreferenceValueVector(G[j][0])

                maxDif = max(maxDif, maxVal - self.getPreferenceValueVector(G[j][0]))

                G[j][0] = maxTerm
                A[j] = optAction

            for j in range(n):
                G[j][1] = G[j][0]

            dif = maxDif

        optPolicy = {}

        f = open(fileName2Output, "w")
        # f.seek(0)

        for j in range(n):
            optPolicy[self.states[j].name] = A[j]
            if printPolicy == True:
                print("\pi(" + self.states[j].name + ")=" + str(A[j]))
                print("M(" + self.states[j].name + ")=" + str(G[j][0]))
            # f.write("\pi("+self.states[j].name+")="+str(A[j]) + "\n")
            f.write(f"\pi({self.states[j].name})={str(A[j])} \n")
            # f.write("M("+self.states[j].name+")="+str(G[j][0]) + "\n")
            f.write(f"M({self.states[j].name})= {str(G[j][0])} \n")

        f.close()

        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed in " + str(numIterations) + " iterations")

        self.computeNumstepsAndSatisProbOfAPolicy(self.states, optPolicy, [], True)

        time_elapsed = (time.time() - time_start)

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.initialState.index], time_elapsed)

    def write_POMDPX_XML(self, filePath):
        f = open(filePath, "w+")
        f.write(self.getPOMDPX_XML2())
        f.close()

    def write_POMDPX_4_EXMDP_XML(self, filePath):
        f = open(filePath, "w+")
        f.write(self.getPOMDPX_4_EXMDP_XML())
        f.close()
