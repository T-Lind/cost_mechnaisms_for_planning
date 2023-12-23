import random

import numpy

from planner.Markov.MarkovChain2 import MarkovChain2
from planner.Markov.MarkovChain3 import MarkovChain3


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
                 has_evidence=False, evidence_list=None):
        if evidence_list is None:
            evidence_list = []
        self.states = []
        self.state_names = state_names
        self.state_events = state_events
        self.initial_state_index = initial_state_index
        self.__create_states(state_names, state_events)
        self.initial_distribution = initial_distribution
        self.transition_matrix = transition_matrix
        self.events = set()
        for s in self.states:
            for e in s.events:
                if not (e in self.events):
                    self.events.add(e)
        self.initialState = self.states[initial_state_index]
        self.nullState = MarkovState("none", "none")
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

    def setDefault_EvidenceDistribution(self, evidence_distribution):
        for state in self.states:
            state.evidence_distribution = evidence_distribution

    def stateIndex(self, state):
        return self.states.index(state)

    def getInitialProbability(self, state):
        return self.initial_distribution[state.index]

    def getTransitionProbability(self, src_state, dstState):
        return self.transition_matrix[src_state.index][dstState.index]

    def printAll(self):
        print("---------------------------------Markov Chain-------------------------------------")
        print(self.states)
        print(self.transition_matrix)
        print(self.initial_distribution)
        print("----------------------------------------------------------------------------------")

    """
    Randomly raise an evidence based on the evidence distribution of the current state
    """

    def pomdp_raiseEvidence(self, current_state):
        # print("current_state.evidence_distribution: "+str(current_state.evidence_distribution))
        return numpy.random.choice(self.evidence_list, p=current_state.evidence_distribution)

    def next_state(self, current_state):
        if current_state == self.nullState:
            return numpy.random.choice(self.states, p=self.initial_distribution)
        # print("len(states)="+str(len(self.states)))
        # print("len(self.transition_matrix[current_state.index])="+str(len(self.transition_matrix[current_state.index])))
        # print("self.transition_matrix[current_state.index]="+str(current_state.index))
        return numpy.random.choice(self.states, p=self.transition_matrix[current_state.index])

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

    def get_set_successors_not_having_event(self, stateSet, event):
        succ = set()
        for state in stateSet:
            scc = self.get_successors_not_having_event(state, event)
            for s in scc:
                if not (s in succ):
                    succ.add(s)
        return succ

    """
    The probablity that the given event happens in the next time step given that the event model is currently in state current_state
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

    def getMarkovChain2(self):
        newStateEvents = []
        for i in range(len(self.state_events)):
            tples = []
            for ev in self.state_events[i]:
                tple = (ev, 1.0)
                tples.append(tple)
            newStateEvents.append(tples)
        markovChain2 = MarkovChain2(self.state_names, newStateEvents, self.transition_matrix, self.initial_distribution,
                                    self.initial_state_index, self.has_evidence, self.evidence_list)
        return markovChain2

    def convertMarkovChain3(self):
        newStateEvents = []
        for i in range(len(self.state_events)):
            tples = []
            # for ev in self.state_events[i]:
            #    tple = (ev, 1.0)
            #    tples.append(tple)
            newStateEvents.append(tples)
        markovChain3 = MarkovChain3(self.state_names, newStateEvents, self.transition_matrix, self.initial_distribution,
                                    self.initial_state_index, self.has_evidence, self.evidence_list)
        return markovChain3

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
                if self.states[i] == self.initialState and markovChain.states[j] != markovChain.initialState:
                    print("not to make")
                    continue
                if markovChain.states[j] == markovChain.initialState and self.states[i] != self.initialState:
                    print("not to make")
                    continue
                if self.states[i] == self.initialState and markovChain.states[j] == markovChain.initialState:
                    initialStateIndex = k
                s1 = self.states[i]
                s2 = markovChain.states[j]
                initialDistribution.append(self.initial_distribution[i] * markovChain.initial_distribution[j])
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
        # numStates = len(markovChain.states)*len(self.states)

        transitionMatrix = [[0 for j in range(numStates)] for i in range(numStates)]

        for k in range(0, len(prodStates)):
            state = prodStates[k]
            s1 = state.anchor[0]
            s2 = state.anchor[1]

            for t in range(0, len(prodStates)):
                statePrime = prodStates[t]
                s1prime = statePrime.anchor[0]
                s2prime = statePrime.anchor[1]
                p = self.transition_matrix[s1.index][s1prime.index] * markovChain.transition_matrix[s2.index][
                    s2prime.index]
                transitionMatrix[k][t] = p

        state = prodStates[0]
        s1 = state.anchor[0]
        s2 = state.anchor[1]

        statePrime = prodStates[0]
        s1prime = statePrime.anchor[0]
        s2prime = statePrime.anchor[1]

        p = self.transition_matrix[s1.index][s1prime.index] * markovChain.transition_matrix[s2.index][s2prime.index]

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
                # initial_distribution.append(self.initial_distribution[i]*markovChain.initial_distribution[j])
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
                p = self.transition_matrix[s1.index][s1prime.index] * markovChain.transition_matrix[s2.index][
                    s2prime.index]
                transitionMatrix[k][t] = p

        state = prodStates[0]
        s1 = state.anchor[0]
        s2 = state.anchor[1]

        statePrime = prodStates[0]
        s1prime = statePrime.anchor[0]
        s2prime = statePrime.anchor[1]

        p = self.transition_matrix[s1.index][s1prime.index] * markovChain.transition_matrix[s2.index][s2prime.index]

        prodMC = MarkovChain(stateNames, stateEvents, transitionMatrix, initialDistribution)

        return prodMC
