from automata.fa.dfa import DFA
from Markov.ExMarkovChain import ExMarkovChain
from Markov.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from sys import float_info
import queue 
from automata.fa import dfa


class ExEventPredictor:
       
    def __init__(self, dfa, exMarkovChain, eventList, verbose):
        self.dfa = dfa
        self.exMarkovChain = exMarkovChain
        self.eventList = eventList
        self.verbose = verbose
        self.mdp = None  
        self.__createProductAutomaton()
        
                
    def __createProductAutomaton(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.exMarkovChain
        mdp = MDP()
        self.mdp = mdp
        
        mdp.actions = self.eventList
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState, False)
        v0 = MDPState(dfa.initial_state+"_"+mc.initialState.name+"_False", t0)
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0
        
        Obs = {True, False}
        """
        Create the state space of the MDP. 
        The state space of the MDP is V = Q \times S where 'Q' is the state space of the DFA specification and 'S' is the state space of the Markov Chain 
        """
        for q in dfa.states:
            for s in mc.states:
                if q == dfa.initial_state and s == mc.initialState:
                    continue
                for ob in Obs:
                    t = (q, s, ob)
                    v = MDPState(q+"_"+s.name+"_"+str(ob), t)
                    mdp.addState(v)
                    if q in dfa.final_states:
                        v.isGoal = True
                        mdp.setAsGoal(v)
                
        
        """
        Create transitions
        """           
        for v1 in mdp.states:
            #if v1 == v0:
            #    continue
            q = v1.anchor[0]
            s = v1.anchor[1]
            
            for v2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """              
                if v2 == v0:    
                    continue
                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                o2 = v2.anchor[2]
                
                if mc.getTransitionProbability(s, s2) == 0:
                    continue
                
                w = mc.getTransitionProbability(s, s2)
                
                for e in self.eventList:
                    if dfa.transitions[q][e] == q2 and q2 == q:
                        if s2.hasEvent(e) and o2 == True:
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*eventCaptureProbability+w*eventProbability*(1-eventCaptureProbability))
                            mdp.addTransition(trans)
                        if s2.hasEvent(e) and o2 == False:
                            trans = MDPTransition(v1, v2, e, s2.events, w*(1-eventProbability))
                            mdp.addTransition(trans)                            
                        if not s2.hasEvent(e) and q == q2 and o2 == False:
                            trans = MDPTransition(v1, v2, e, s2.events, w)
                            mdp.addTransition(trans)
                    else:                       
                        if s2.hasEvent(e) and dfa.transitions[q][e] == q2 and o2 == True:
                            eventProbability = s2.getEventProbability(e)
                            eventCaptureProbability = s2.getEventCaptureProbabilities(e)
                        #print("w:"+str(w))
                        #print("eventProbability:"+str(eventProbability))
                        #print("eventCaptureProbability:"+str(eventCaptureProbability))
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*eventCaptureProbability)
                            mdp.addTransition(trans)
                        elif s2.hasEvent(e) and q == q2 and o2 == False:
                            eventProbability = s2.getEventProbability(e)
                            eventCaptureProbability = s2.getEventCaptureProbabilities(e)
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*(1-eventCaptureProbability)+w*(1-eventProbability))
                            mdp.addTransition(trans)
                        if not s2.hasEvent(e) and q == q2 and o2 == False:
                            trans = MDPTransition(v1, v2, e, s2.events, w)
                            mdp.addTransition(trans)
        
        
        
        """
        Create set of observations
        """
        for s in mc.states:
            o1 = (True, s)
            o2 = (False, s)
            mdp.observations.append(o1)
            mdp.observations.append(o2)
            
        mdp.createObservationFunctionDict()
        
        for e in mdp.actions:
            for x in mdp.states:
                s = x.anchor[1] 
                b = x.anchor[2]
                for o in mdp.observations:
                    s2 = o[1]
                    r = o[0] 
                    if r == b and s2 == s:
                        mdp.observationFunction[o][x][e] = 1
                    elif r != b and s2 == s:
                        mdp.observationFunction[o][x][e] = 0
                    if s2 != s:
                        mdp.observationFunction[o][x][e] = 0
                    
        
        #self.mdp.makeGoalStatesAbsorbing()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states and "+str(len(mdp.goalStates))+" goal states")
            print(str(mdp.observationFunction))
            print("----------------------------------------------") 
                        
               
    def __createProductAutomaton2(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.exMarkovChain
        mdp = MDP()
        self.mdp = mdp
        
        mdp.actions = self.eventList
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state+"_"+mc.initialState.name, t0)
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0
        
        """
        Create the state space of the MDP. 
        The state space of the MDP is V = Q \times S where 'Q' is the state space of the DFA specification and 'S' is the state space of the Markov Chain 
        """
        for q in dfa.states:
            for s in mc.states:
                if q == dfa.initial_state and s == mc.initialState:
                    continue
                t = (q, s)
                v = MDPState(q+"_"+s.name, t)
                mdp.addState(v)
                if q in dfa.final_states:
                    v.isGoal = True
                    mdp.setAsGoal(v)
                
        
        """
        Create transitions
        """           
        for v1 in mdp.states:
            #if v1 == v0:
            #    continue
            q = v1.anchor[0]
            s = v1.anchor[1]
            
            for v2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """              
                if v2 == v0:    
                    continue
                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                
                if mc.getTransitionProbability(s, s2) == 0:
                    continue
                
                w = mc.getTransitionProbability(s, s2)
                
                for e in self.eventList:
                    if dfa.transitions[q][e] == q2 and q2 == q:
                        if s2.hasEvent(e):
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*eventCaptureProbability+w*eventProbability*(1-eventCaptureProbability)+w*(1-eventProbability))
                            mdp.addTransition(trans)
                        if not s2.hasEvent(e) and q == q2:
                            trans = MDPTransition(v1, v2, e, s2.events, w)
                            mdp.addTransition(trans)
                    else:                       
                        if s2.hasEvent(e) and dfa.transitions[q][e] == q2:
                            eventProbability = s2.getEventProbability(e)
                            eventCaptureProbability = s2.getEventCaptureProbabilities(e)
                        #print("w:"+str(w))
                        #print("eventProbability:"+str(eventProbability))
                        #print("eventCaptureProbability:"+str(eventCaptureProbability))
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*eventCaptureProbability)
                            mdp.addTransition(trans)
                        elif s2.hasEvent(e) and q == q2:
                            eventProbability = s2.getEventProbability(e)
                            eventCaptureProbability = s2.getEventCaptureProbabilities(e)
                            trans = MDPTransition(v1, v2, e, s2.events, w*eventProbability*(1-eventCaptureProbability)+w*(1-eventProbability))
                            mdp.addTransition(trans)
                        if not s2.hasEvent(e) and q == q2:
                            trans = MDPTransition(v1, v2, e, s2.events, w)
                            mdp.addTransition(trans)
        
        """
        for v1 in mdp.states:
            for a in mdp.actions:
                for v2 in mdp.states:
                    trans = []
                    for t in mdp.transitions:
                        if t.srcState == v1 and t.dstState == v2 and t.action == a:
                            trans.append(t)
                    if len(trans)>0:
                        s = 0 
                        i = 1
                        while i < len(trans):
                            s += trans[i].probability
                            mdp.transitions.remove(trans[i])
                            t.srcState.transitions.remove(trans[i])
                        trans[0].probability += s
        """
        
        """
        Create set of observations
        """
        for s in mc.states:
            o1 = (True, s)
            o2 = (False, s)
            mdp.observations.append(o1)
            mdp.observations.append(o2)
            
        mdp.createObservationFunctionDict()
        
        for e in mdp.actions:
            for x in mdp.states:
                s = x.anchor[1] 
                for o in mdp.observations:
                    s2 = o[1]
                    r = o[0] 
                    if r == True and s2 == s:
                        mdp.observationFunction[o][x][e] = s.getEventProbability(e)
                    if r == False and s2 == s:
                        mdp.observationFunction[o][x][e] = 1-s.getEventProbability(e)
                    if s2 != s:
                        mdp.observationFunction[o][x][e] = 0
                    
        
        #self.mdp.makeGoalStatesAbsorbing()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states and "+str(len(mdp.goalStates))+" goal states")
            print(str(mdp.observationFunction))
            print("----------------------------------------------") 

    
    
    def optimalPolicyToMaximizeProbOfGoals(self, H, printPolicy):
        self.mdp.makeBeliefTree(H)
        
        self.beliefTree = self.mdp.beliefTree
        
        self.beliefTree.computeOptimalPolicyToMaxProbToGoal(H, self.eventList)
        
        if self.verbose == True:
            print("Optimal policy to maximize the probability of reaching goal states has been computed. The maximum probability is: "+str(self.beliefTree.root.expcetedProbToGoal))
            print("Number of belief nodes of the tree with a nonzero goal value is: "+str(self.beliefTree.numberOfNodesWithNonzeroGoalValue()))
            #print(str(self.beliefTree))
                     
        #return G[0][self.mdp.initialState.index]
        return (self.mdp.beliefTree, self.mdp.beliefTree.root.bestActionToMaxExpectedToGoal, self.mdp.beliefTree.root.expcetedProbToGoal)
         
        
    
    def simulateMaximizeProbToGoals(self, beliefTree, H, printOutput =True):
        story = ""
        #s = self.markovChain.nullState
        s = self.exMarkovChain.initialState
        q = self.dfa.initial_state
        node = self.beliefTree.root
        q_previous = q
        i = 0
        k = 0
        while i < H:
             
            predictedEvent = node.bestActionToMaxExpectedToGoal
            
            #print("bestAction="+node.bestActionToMaxExpectedToGoal)
            
            s2 = self.exMarkovChain.nextState(s)
            
            occuredEvents = s2.simulateEventsOccring()
            
            eventOccured = False
            if predictedEvent in occuredEvents:
                eventOccured = True
            
            eventCaptured = False
            if predictedEvent in occuredEvents:
                if s2.simulateEventCapturing(predictedEvent):
                    eventCaptured = True 
            
            q_previous = q
            if eventCaptured == True:
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            i += 1
            
            observation = self.mdp.getObservation(s2, eventOccured)
            
            #print("obs="+str(observation))
            
            #print("node="+str(node))
            
            node = node.getChild(observation, predictedEvent)
            
            #print("node="+str(node))
            
            
            if printOutput == True:
                #print(str(node.beliefState))
                print(str(i)+". "+"q="+q_previous+", s="+s.name+", predicted="+predictedEvent+", occurred="+str(occuredEvents)+", Captured="+str(eventCaptured)+", sNext="+s2.name+", recorded story="+story+", b="+str(node.beliefState))  
            
            s = s2
        succeed = False
        if q in self.dfa.final_states:
            succeed = True
        return (i, succeed, story)
    


    def optimalPolicyMaximizingRewards(self, horizon, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        

        G = [[0.0 for j in [0, 1]] for i in range(n)] 
        A = ["" for j in range(n)]
        
        for j in range(n):
            if (self.mdp.states[j].isGoal):
                G[j][0] = 1
                G[j][1] = 1
        
        dif = float_info.max
        
        
        numIterations = 0
        
        r = 0
        
        i = 0
        
        while i < horizon:
            numIterations += 1
                        
            for j in range(n):
                
                if self.mdp.states[j].reachable == False:
                    continue
                                
                maxVal = float_info.min
                optAction = ""
                state = self.mdp.states[j]
                               
                for action in self.eventList:
                    val = 0.0
                    
                    for k in range(n):
                        reward = 0
                        if state.isGoal:
                            reward = 1
                        term = self.mdp.conditionalProbability(k, j, action)*(G[k][1])
                        val += term
                        #r += 1
                        #print("r="+str(r))
                    if val > maxVal:
                        maxVal = val
                        optAction = action    
                
                #if minVal-G[j][0] > maxDif:
                    #maxDif = minVal-G[j][0]
                
                
                
                G[j][0] = maxVal
                A[j] = optAction
                
            for j in range(n):
                G[j][1] = G[j][0]
                
            i += 1
            
        
        optPolicy = {}
        for q in self.dfa.states:
            optPolicy[q] = {}
        
        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][str(self.mdp.states[j].anchor[1])] = A[j]
            if printPolicy == True:
                    print("\pi("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+A[j])
                    print("M("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j][0]))
              
        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed in "+str(numIterations)+" iterations")
                      
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index])    
        