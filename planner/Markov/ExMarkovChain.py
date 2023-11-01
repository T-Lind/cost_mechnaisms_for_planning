import numpy 
import random



class ExMarkovState:
    """
    Parameter events is a list of pairs (e, p) where e is the event name 
    and p is the probability that event e happens at the state.
    """
    def __init__(self, name="", events=[], eventProbabilites=[], eventCaptureProbabilities=[]):
        self.name = name
        self.events = events
        self.eventProbabilites = eventProbabilites
        self.eventCaptureProbabilities = eventCaptureProbabilities
        self.index = -1
        
    def __str__(self):
        return self.name
    
    def __repr__(self):

        return self.name

        
    def hasEvent(self, e):
        i = self.events.index(e)
        if i < 0:
            return False
        
        if self.eventProbabilites[i]>0:
            return True
        
        return False
    
    def getEventProbability(self, e):
        i = self.events.index(e)
        
        if i < 0:
            return 0
        
        return self.eventProbabilites[i]
    
    def getEventCaptureProbabilities(self, e):
        i = self.events.index(e)
        
        if i < 0:
            return 0
        
        
        return self.eventCaptureProbabilities[i]    
        
    def simulateEventsOccring(self):
        eventsOccuring = set()
        for i in range(len(self.eventProbabilites)):
            r = random.uniform(0, 1)
            if r < self.eventProbabilites[i]:
                eventsOccuring.add(self.events[i])
        return eventsOccuring
    
    def simulateEventCapturing(self, event):
        r = random.uniform(0, 1)
        i = self.events.index(event)
        if r < self.eventCaptureProbabilities[i]:
            return True
        return False
    


class ExMarkovChain:
    """
    Parameter stateEvents is a collection of lists, where for each state 's' there is a list of pairs (e, p), in which 'e' is the event name and
    'p' is the probability that event 'e' happens at state 's'  
    """
    def __init__(self, stateNames, events, stateEventProbabilites, stateEventCaptureProbabilities, transitionMatrix, initialStateIndex=0):
        self.states = []
        self.__createStates(stateNames, events, stateEventProbabilites, stateEventCaptureProbabilities)
        self.transitionMatrix = transitionMatrix 
        self.events = events
        self.initialState = self.states[initialStateIndex]
        
    
    def __createStates(self, stateNames, events, stateEventProbabilites, stateEventCaptureProbabilities):
        self.states = []
        
        i = 0
        
        for name in stateNames:
            state = ExMarkovState(name, events, stateEventProbabilites[i], stateEventCaptureProbabilities[i])
            state.index = len(self.states)
            self.states.append(state)
            i += 1
            
    def stateIndex(self, state):
        return self.states.index(state)
    
    
    def printAll(self):
        print("---------------------------------Markov Chain-------------------------------------")
        print(self.states)
        print(self.transitionMatrix)
        print(self.initialDistribution)
        print("----------------------------------------------------------------------------------")
        
    def nextState(self, currentState):
        return numpy.random.choice(self.states, p=self.transitionMatrix[currentState.index]) 
        
    def getSuccessorsHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and self.states[j].hasEvent(event):
                succ.add(self.states[j])
        return succ
    
    def getSuccessorsNotHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and not self.states[j].hasEvent(event):
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

    def getTransitionProbability(self, srcState, dstState):
        return self.transitionMatrix[srcState.index][dstState.index]
    
       