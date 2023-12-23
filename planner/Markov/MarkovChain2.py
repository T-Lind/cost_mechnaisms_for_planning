import numpy 


class MarkovState2:
    """
    Parameter events is a list of pairs (e, p) where e is the event name 
    and p is the probability that event e happens at the state.
    """
    def __init__(self, name="", events=set(), evidenceDistribution=[]):
        self.name = name
        self.events = events
        self.evidenceDistribution = evidenceDistribution
        self.index = -1
        
    def __str__(self):
        return self.name
    
    def __repr__(self):

        return self.name
        #return self.name+":"+str(self.events)
        #s = "("+self.name+", "+str(self.events)+")"
        #return s
        
    def getFullName(self):
        return self.name+":"+str(self.events)
        
    def hasEvent(self, e):
        for ep in self.events:
            if e == ep[0]:
                return True
        return False
    
    def probabilityOfEvent(self, e):
        for ep in self.events:
            if e == ep[0]:
                return ep[1]
        return 0
    
    
    def getEventPair(self, e):
        for ep in self.events:
            if ep[0] == e:
                return ep
        
        return None


class MarkovChain2:
    """
    Parameter state_events is a collection of lists, where for each state 's' there is a list of pairs (e, p), in which 'e' is the event name and
    'p' is the probability that event 'e' happens at state 's'  
    """
    def __init__(self, stateNames, stateEvents, transitionMatrix, initialDistribution, initialStateIndex=0, hasEvidence=False, evidenceList=[]):
        self.states = []
        self.__createStates(stateNames, stateEvents)
        self.transitionMatrix = transitionMatrix 
        self.events = set()
        for s in self.states:
            for ep in s.events:   # ep is a pair (e, p) where e is the event and p is the probability that event e happens at the state
                if not(ep[0] in self.events):    # ep[0] is the event e
                    self.events.add(ep[0])
        self.initialState = self.states[initialStateIndex]
        self.nullState = MarkovState2("none", "none")
        self.hasEvidence = hasEvidence
        self.evidenceList = evidenceList
        
    
    def __createStates(self, stateNames, stateEvents):
        self.states = []
        
        i = 0
        
        for name in stateNames:
            state = MarkovState2(name, stateEvents[i])
            state.index = len(self.states)
            self.states.append(state)
            i += 1
            
    def setDefault_EvidenceDistribution(self, evidenceDistribution):
        for state in self.states:
            state.evidence_distribution = evidenceDistribution

    def stateIndex(self, state):
        return self.states.index(state)
    
    def getInitialProbability(self, state):
        return self.initialDistribution[state.index]
    
    def getTransitionProbability(self, srcState, dstState):
        return self.transitionMatrix[srcState.index][dstState.index]
    
    def printAll(self):
        print("---------------------------------Markov Chain-------------------------------------")
        print("States = [")
        for s in self.states:
            print(s)
        print("]")
        print("TransitionMatrix = [")
        for i in range(len(self.transitionMatrix)):
            print(self.states[i].name+":"+str(self.transitionMatrix[i]))
        print("]")
        print("----------------------------------------------------------------------------------")
        
    def printToFile(self, fileName):
        
        strP = "---------------------------------Markov Chain-------------------------------------"+"\n"
        strP += "States = ["+"\n"
        for s in self.states:
            strP += s.getFullName()+"\n"
        strP += "]"+"\n"
        strP +=  "TransitionMatrix = ["+"\n"
        for i in range(len(self.transitionMatrix)):
            strP += self.states[i].name+":"+str(self.transitionMatrix[i])+"\n"
        strP += "]"+"\n"
        strP += "----------------------------------------------------------------------------------"+"\n"
        
        f = open(fileName, "w")
        f.write(strP)
        f.close()

        
    def nextState(self, currentState):
        if currentState == self.nullState:
            return numpy.random.choice(self.states, p=self.initialDistribution)
        #print("len(states)="+str(len(self.states)))
        #print("len(self.transition_matrix[current_state.index])="+str(len(self.transition_matrix[current_state.index])))
        #print("self.transition_matrix[current_state.index]="+str(current_state.index))
        return numpy.random.choice(self.states, p=self.transitionMatrix[currentState.index]) 
        
    def getSuccessorsHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and self.states[j].has_event(event):
                succ.add(self.states[j])
        return succ
    
    def getSuccessorsNotHavingEvent(self, state, event):
        succ = set()
        for j in range(len(self.states)):
            if self.transitionMatrix[state.index][j] > 0 and not self.states[j].has_event(event):
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


    def getNexTimeMostPlausibleEvent(self, eventList, currentState):
        maxProb  =  -1
        mostPlauEv = None
        probs = [0]*len(eventList)
        i = 0
        for ev in eventList:
            prob = self.getNextTimeProbabilityOfEvent(ev, currentState)
            probs[i] = prob
            #print("(event, probability)=("+ev+", "+str(prob)+")")
            if prob > maxProb:
                maxProb = prob
                #mostPlauEv = ev 
            i += 1
        
        selectedEvents = []
        for i in range(len(eventList)):
            if probs[i] == maxProb:
                selectedEvents.append(eventList[i])   
        print("selectedEvents="+str(selectedEvents))
        
        if len(selectedEvents) > 0:
            mostPlauEv = random.choice(selectedEvents)
        elif len(eventList) > 0:
            mostPlauEv = eventList[0]  
            
        print("selectedEvents = "+mostPlauEv)     
        
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

    
       