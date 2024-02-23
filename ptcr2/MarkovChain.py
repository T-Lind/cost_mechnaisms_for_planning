import random

import numpy


class MarkovState:
    # name = ""
    # events = set()
    # index = -1
    def __init__(self, name="", events=set(), evidenceDistribution=[]):
        self.name = name
        self.events = events
        self.evidenceDistribution = evidenceDistribution
        self.index = -1

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name
        return self.name + ":" + str(self.events)
        s = "(" + self.name + ", " + str(self.events) + ")"
        return s


class MarkovChain:

    def __init__(self, state_names, state_events, transitionMatrix, initial_distribution, initialStateIndex=0,
                 hasEvidence=False, evidenceList=[]):
        self.states = []
        self.stateNames = state_names
        self.stateEvents = state_events
        self.initialStateIndex = initialStateIndex
        self.__create_states(state_names, state_events)
        self.initialDistribution = initial_distribution
        self.transitionMatrix = transitionMatrix
        self.events = set()
        for s in self.states:
            for e in s.events:
                if not (e in self.events):
                    self.events.add(e)
        self.initial_state = self.states[initialStateIndex]
        self.null_state = MarkovState("none", "none")
        self.has_evidence = hasEvidence
        self.evidence_list = evidenceList

    def __create_states(self, stateNames, stateEvents):
        self.states = []

        i = 0

        for name in stateNames:
            state = MarkovState(name, stateEvents[i])
            state.index = len(self.states)
            self.states.append(state)
            i += 1

    def setDefault_EvidenceDistribution(self, evidenceDistribution):
        for state in self.states:
            state.evidenceDistribution = evidenceDistribution

    def stateIndex(self, state):
        return self.states.index(state)

    def getInitialProbability(self, state):
        return self.initialDistribution[state.index]

    def getTransitionProbability(self, srcState, dstState):
        return self.transitionMatrix[srcState.index][dstState.index]

    def printAll(self):
        print("---------------------------------Markov Chain-------------------------------------")
        print(self.states)
        print(self.transitionMatrix)
        print(self.initialDistribution)
        print("----------------------------------------------------------------------------------")

    """
    Randomly raise an evidence based on the evidence distribution of the current state
    """

    def pomdp_raiseEvidence(self, currentState):
        # print("currentState.evidenceDistribution: "+str(currentState.evidenceDistribution))
        return numpy.random.choice(self.evidence_list, p=currentState.evidenceDistribution)

    def nextState(self, currentState):
        if currentState == self.null_state:
            return numpy.random.choice(self.states, p=self.initialDistribution)
        # print("len(states)="+str(len(self.states)))
        # print("len(self.transitionMatrix[currentState.index])="+str(len(self.transitionMatrix[currentState.index])))
        # print("self.transitionMatrix[currentState.index]="+str(currentState.index))
        return numpy.random.choice(self.states, p=self.transitionMatrix[currentState.index])

    def getSuccessorsHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and (event in self.states[j].events):
                succ.add(self.states[j])
        return succ

    def getSuccessorsNotHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and not (event in self.states[j].events):
                succ.add(self.states[j])
        return succ

    def getSetSuccessorsHavingEvent(self, stateSet, event):
        succ = set()
        for state in stateSet:
            scc = self.getSuccessorsHavingEvent(state, event)
            for s in scc:
                if not (s in succ):
                    succ.add(s)
        return succ

    def getSetSuccessorsNotHavingEvent(self, stateSet, event):
        succ = set()
        for state in stateSet:
            scc = self.getSuccessorsNotHavingEvent(state, event)
            for s in scc:
                if not (s in succ):
                    succ.add(s)
        return succ

    """
    The probablity that the given event happens in the next time step given that the event model is currently in state currentState
    """

    def possbilityOfHappeningInNextStep(self, currentState, event):
        result = 0.0
        for state in self.states:
            if event not in state.events:
                continue
            result += self.getTransitionProbability(currentState, state)
        return result

    def getNexTimeMostPlausibleEvent(self, eventList, currentState):
        maxProb = -1
        mostPlauEv = None
        probs = [0] * len(eventList)
        i = 0
        for ev in eventList:
            prob = self.getNextTimeProbabilityOfEvent(ev, currentState)
            probs[i] = prob
            # print("(event, probability)=("+ev+", "+str(prob)+")")
            if prob > maxProb:
                maxProb = prob
                # mostPlauEv = ev
            i += 1

        selectedEvents = []
        for i in range(len(eventList)):
            if probs[i] == maxProb:
                selectedEvents.append(eventList[i])
                # print("selectedEvents="+str(selectedEvents))

        if len(selectedEvents) > 0:
            mostPlauEv = random.choice(selectedEvents)
        elif len(eventList) > 0:
            mostPlauEv = eventList[0]

            # print("selectedEvents = "+mostPlauEv)

        return mostPlauEv

    def getNextTimeProbabilityOfEvent(self, event, currentState):
        result = 0
        for state in self.states:
            if self.getTransitionProbability(currentState, state) == 0:
                continue
            if event not in state.events:
                continue
            result += self.getTransitionProbability(currentState, state)

        return result

    def product_singleInitialState(self, markovChain, pairEventsList=[]):
        prodStates = []

        stateNames = []

        stateEvents = []

        initialDistribution = []

        transitionMatrix = [[]]

        initialStateIndex = -1

        k = 0

        for i in range(len(self.states)):
            for j in range(len(markovChain.states)):
                if self.states[i] == self.initial_state and markovChain.states[j] != markovChain.initial_state:
                    print("not to make")
                    continue
                if markovChain.states[j] == markovChain.initial_state and self.states[i] != self.initial_state:
                    print("not to make")
                    continue
                if self.states[i] == self.initial_state and markovChain.states[j] == markovChain.initial_state:
                    initialStateIndex = k
                s1 = self.states[i]
                s2 = markovChain.states[j]
                initialDistribution.append(self.initialDistribution[i] * markovChain.initialDistribution[j])
                stateName = s1.name + "_" + s2.name
                stateNames.append(stateName)
                eventSet = s1.events.union(s2.events)
                for t in pairEventsList:
                    appeared = True
                    ev = ""
                    for e in t[0]:
                        appeared = appeared and (e in eventSet)
                        ev = ev + e
                    if len(t[0]) > 0 and appeared:
                        eventSet.add(t[1])
                stateEvents.append(eventSet)
                prodState = MarkovState(stateName, eventSet)
                prodState.anchor = (s1, s2)
                prodStates.append(prodState)
                k = k + 1

        numStates = k
        # numStates = len(markov_chain.states)*len(self.states)

        transitionMatrix = [[0 for j in range(numStates)] for i in range(numStates)]

        for k in range(0, len(prodStates)):
            state = prodStates[k]
            s1 = state.anchor[0]
            s2 = state.anchor[1]

            for t in range(0, len(prodStates)):
                statePrime = prodStates[t]
                s1prime = statePrime.anchor[0]
                s2prime = statePrime.anchor[1]
                p = self.transitionMatrix[s1.index][s1prime.index] * markovChain.transitionMatrix[s2.index][
                    s2prime.index]
                transitionMatrix[k][t] = p

        state = prodStates[0]
        s1 = state.anchor[0]
        s2 = state.anchor[1]

        statePrime = prodStates[0]
        s1prime = statePrime.anchor[0]
        s2prime = statePrime.anchor[1]

        p = self.transitionMatrix[s1.index][s1prime.index] * markovChain.transitionMatrix[s2.index][s2prime.index]

        prodMC = MarkovChain(stateNames, stateEvents, transitionMatrix, initialDistribution, initialStateIndex)

        return prodMC

    def product(self, markovChain, pairEventsList=[]):
        prodStates = []

        stateNames = []

        stateEvents = []

        initialDistribution = []

        transitionMatrix = [[]]

        for i in range(len(self.states)):
            for j in range(len(markovChain.states)):
                s1 = self.states[i]
                s2 = markovChain.states[j]
                # initial_distribution.append(self.initial_distribution[i]*markov_chain.initial_distribution[j])
                stateName = s1.name + "_" + s2.name
                stateNames.append(stateName)
                eventSet = s1.events.union(s2.events)
                for t in pairEventsList:
                    appeared = True
                    ev = ""
                    for e in t[0]:
                        appeared = appeared and (e in eventSet)
                        ev = ev + e
                    if len(t[0]) > 0 and appeared:
                        eventSet.add(t[1])
                stateEvents.append(eventSet)
                prodState = MarkovState(stateName, eventSet)
                prodState.anchor = (s1, s2)
                prodStates.append(prodState)

        numStates = len(markovChain.states) * len(self.states)

        transitionMatrix = [[0 for j in range(numStates)] for i in range(numStates)]

        for k in range(0, len(prodStates)):
            state = prodStates[k]
            s1 = state.anchor[0]
            s2 = state.anchor[1]

            for t in range(0, len(prodStates)):
                statePrime = prodStates[t]
                s1prime = statePrime.anchor[0]
                s2prime = statePrime.anchor[1]
                p = self.transitionMatrix[s1.index][s1prime.index] * markovChain.transitionMatrix[s2.index][
                    s2prime.index]
                transitionMatrix[k][t] = p

        state = prodStates[0]
        s1 = state.anchor[0]
        s2 = state.anchor[1]

        statePrime = prodStates[0]
        s1prime = statePrime.anchor[0]
        s2prime = statePrime.anchor[1]

        p = self.transitionMatrix[s1.index][s1prime.index] * markovChain.transitionMatrix[s2.index][s2prime.index]

        prodMC = MarkovChain(stateNames, stateEvents, transitionMatrix, initialDistribution)

        return prodMC
