from automata.fa.dfa import DFA
from Markov.MarkovChain2 import MarkovChain2
from Markov.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from sys import float_info
import queue 


import time

class EventPredictor2:
       
    def __init__(self, dfa, markovChain, eventList, verbose, currentMarkovStateVisibile=True):
        self.dfa = dfa
        self.markovChain = markovChain
        self.eventList = eventList
        self.verbose = verbose
        self.mdp = None  
        self.currentMarkovStateVisibile = currentMarkovStateVisibile
        if currentMarkovStateVisibile == True: 
            #self.__createProductAutomaton_singleInitialState()
            self.__createProductAutomaton_singleInitialState_onlyreachables()
        else:
            self.__createProductAutomaton_WhereCurrentMarkovStateInvisible()    
        
                
    def __createProductAutomaton_singleInitialState(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.has_evidence = mc.has_evidence
        mdp.evidence_list = mc.evidence_list
        self.mdp = mdp
        
        mdp.actions = self.eventList
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state+"_"+mc.initialState.name, t0)
        v0.evidence_distribution = mc.initialState.evidence_distribution
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
                v.evidence_distribution = s.evidence_distribution
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
            #if v1 == v0:
            #    continue
            q = v1.anchor[0]
            s = v1.anchor[1]
            
            for v2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """              
                #if v2 == v0:    
                 #   continue
                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                
                if mc.getTransitionProbability(s, s2) == 0:
                    continue
                
                #print("Checking MDP states: "+str(v1)+", "+str(v2))
                
                w = mc.getTransitionProbability(s, s2)
                
                for e in self.eventList:
                    if s2.has_event(e) and dfa.transitions[q][e] == q2 and (q != q2) and (q not in dfa.final_states):
                        ep = s2.getEventPair(e)
                        p = ep[1]
                        trans = MDPTransition(v1, v2, e, s2.events, w*p)
                        mdp.addTransition(trans)
                    elif s2.has_event(e) and dfa.transitions[q][e] == q2 and (q == q2) and (q not in dfa.final_states):
                        ep = s2.getEventPair(e)
                        p = ep[1]
                        trans = MDPTransition(v1, v2, e, s2.events, w)
                        mdp.addTransition(trans)                        
                    elif s2.has_event(e) and q == q2 and (q not in dfa.final_states):
                        ep = s2.getEventPair(e)
                        p = ep[1]
                        if w*(1-p)>0:
                            trans = MDPTransition(v1, v2, e, s2.events, w*(1-p))
                            mdp.addTransition(trans)
                    elif (not s2.has_event(e)) and q == q2 and (q not in dfa.final_states):
                        trans = MDPTransition(v1, v2, e, s2.events, w)
                        mdp.addTransition(trans)
                    
                        
        """
        Make goal states absorbing
        """
        for v in mdp.goalStates:
            for e in self.eventList:
                trans = MDPTransition(v, v, e, v.anchor[1].events, 1)
                mdp.addTransition(trans)
        
            
                        #selfLoopAdded[q] = True
        #self.mdp.recognizeReachableStates()
        
        #self.mdp.makeGoalStatesAbsorbing()
        
        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()
        
        self.mdp.makeObservationFunction()
        
        self.mdp.checkTransitionFunction()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states")
            print("----------------------------------------------") 
                        

    def __createProductAutomaton_singleInitialState_onlyreachables(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.has_evidence = mc.has_evidence
        mdp.evidence_list = mc.evidence_list
        self.mdp = mdp
        
        mdp.actions = self.eventList
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state+"_"+mc.initialState.name, t0)
        v0.evidence_distribution = mc.initialState.evidence_distribution
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
            print("Iteration "+str(i))
            v = queue.pop(0)
            vName = queueNames.pop(0)
            t = v.anchor
            q = t[0]
            s = t[1]
            if v.isGoal == True:
                for e in self.eventList:
                    trans = MDPTransition(v, v, e, s.events, 1)
                    mdp.addTransition(trans)
                continue
                    
            for s2 in mc.states:
                w =probability = mc.getTransitionProbability(s, s2)
                if probability == 0:
                   continue
                for e in self.eventList:
                    #if s2.probabilityOfEvent(e) == 0:
                    #    continue
                    #print("Entereded the for loop")
                    q2 = self.dfa.transitions[q][e]
                    
                    #tran = MDPTransition(v, v2, e, s2.events, probability)
                    #mdp.addTransition(tran)
                    #ep = s2.getEventPair(e)
                    #p = ep[1]
                    p = s2.probabilityOfEvent(e)
                    if (p > 0) and (q != q2) and (q not in dfa.final_states):
                        v2Name = q2+"_"+s2.name
                        v2wasInMDP = False
                        if v2Name in mdp.statesDictByName.keys():
                            v2 = mdp.statesDictByName[v2Name]
                            v2wasInMDP = True
                        else:
                            t2 = (q2, s2)
                            v2 = MDPState(v2Name, t2)
                            v2.evidence_distribution = s2.evidence_distribution
                            mdp.addState(v2)
                            cnt += 1
                            if q2 in dfa.final_states:
                                v2.isGoal = True
                                mdp.setAsGoal(v2)
                        if v2Name not in queueNames and v2wasInMDP == False:
                            queue.append(v2)       
                            queueNames.append(v2Name) 
                        trans = MDPTransition(v, v2, e, s2.events, w*p)
                        mdp.addTransition(trans)
                    elif (q == q2) and (q not in dfa.final_states):
                        v2Name = q2+"_"+s2.name
                        v2wasInMDP = False
                        if v2Name in mdp.statesDictByName.keys():
                            v2 = mdp.statesDictByName[v2Name]
                            v2wasInMDP = True
                        else:
                            t2 = (q2, s2)
                            v2 = MDPState(v2Name, t2)
                            v2.evidence_distribution = s2.evidence_distribution
                            mdp.addState(v2)
                            cnt += 1
                            if q2 in dfa.final_states:
                                v2.isGoal = True
                                mdp.setAsGoal(v2)
                        if v2Name not in queueNames and v2wasInMDP == False:
                            queue.append(v2)       
                            queueNames.append(v2Name) 
                        trans = MDPTransition(v, v2, e, s2.events, w)
                        mdp.addTransition(trans)                        
                    #elif s2.hasEvent(e) and q == q2 and (q not in dfa.final_states):
                    #    if w*(1-p)>0:
                    #        trans = MDPTransition(v, v2, e, s2.events, w*(1-p))
                    #        mdp.addTransition(trans)
                    #elif p == 0 and q == q2 and (q not in dfa.final_states):
                    #    trans = MDPTransition(v, v2, e, s2.events, w)
                    #    mdp.addTransition(trans)                    
                    
                for e in self.eventList:
                    p = s2.probabilityOfEvent(e)
                    if p == 1:
                        continue
                    if dfa.transitions[q][e] == q:
                        continue
                    v2Name = q+"_"+s2.name
                    v2wasInMDP = False
                    if v2Name in mdp.statesDictByName.keys():
                        v2 = mdp.statesDictByName[v2Name]
                        v2wasInMDP = True
                    else:
                        t2 = (q, s2)
                        v2 = MDPState(v2Name, t2)
                        v2.evidence_distribution = s2.evidence_distribution
                        mdp.addState(v2)
                        cnt += 1
                        if q in dfa.final_states:
                            v2.isGoal = True
                            mdp.setAsGoal(v2)                        
                    if v2Name not in queueNames and v2wasInMDP == False:
                        queue.append(v2)       
                        queueNames.append(v2Name) 
                    tran = MDPTransition(v, v2, e, s2.events, probability*(1-p))
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

        print("Number of added states = "+str(cnt))
        
        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()
        
        self.mdp.makeObservationFunction()
        
        self.mdp.checkTransitionFunction()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states")
            print("----------------------------------------------") 
            
                     
    def optimalPolicyFiniteHorizon(self, F, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        G = [[0.0 for j in range(n)] for i in range(F+1)] 
        A = [["" for j in range(n)] for i in range(F+1)]
        
        for j in range(n):
            if (self.mdp.states[j].isGoal):
                G[F][j] = 0.0
                A[F][j] = "STOP"
            else:
                G[F][j] = 10000.0
            
        for i in range(F-1, -1, -1):
            #print(i)
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
                    if  state.isGoal == False:
                        val += 1
                    for k in range(n):
                        term = G[i+1][k]*self.mdp.conditionalProbability(k, j, action)
                        val += term
                    if val < minVal:
                        minVal = val
                        optAction = action    
                G[i][j] = minVal
                A[i][j] = optAction
                
        
        optPolicy = {}
        for q in self.dfa.states:
            optPolicy[q] = {}
        
        print("mdp.initialState=["+self.mdp.initialState.anchor[0]+","+self.mdp.initialState.anchor[1].name+"]")
        
        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][self.mdp.states[j].anchor[1]] = A[0][j]
            if printPolicy == True:
                    print("\pi("+self.mdp.states[j].anchor[0]+","+self.mdp.states[j].anchor[1].name+")="+A[0][j])
                    print("M("+self.mdp.states[j].anchor[0]+","+self.mdp.states[j].anchor[1].name+")="+str(G[0][j]))
              
        if self.verbose == True:
            print("optimal policy for finite horizon has been computed")
                     
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index])
    

    def optimalPolicyInfiniteHorizonForDAG(self, epsilonOfConvergance, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        
        G = [0.0 for i in range(n)] 
        A = ["" for j in range(n)]
        
        time_start = time.time()
        
        #for state in self.mdp.states:
            #state.computeAvailableActions()
        if self.mdp.topologicalOrder == []:
            self.mdp.computeTopologicalOrder()
        
        for state in self.mdp.topologicalOrder:
            if state.isGoal:
                G[state.index] = 0.0
                A[state.index] = "STOP"
                continue
            minValue = float_info.max
            bestAction = None
            #print("Num of actions of "+str(state.name)+": "+str(len(state.availableActions)))
            for action in state.availableActions:
                selfFactor = 1.0
                sumSuccessorsG = 0.0
                #print("Num of transitions of ("+state.name+", "+action+"): "+str(len(state.actionsTransitions[action])))
                for tran in state.actionsTransitions[action]:
                    if tran.dstState == state:
                        selfFactor -= tran.probability
                        sumSuccessorsG += tran.probability*1
                    else:
                        sumSuccessorsG += tran.probability*(G[tran.dstState.index]+1)
                if selfFactor == 0:  # self loop and that action is not an optimal action for that state
                    continue
                value = sumSuccessorsG/selfFactor
                if value < minValue:
                    minValue = value
                    bestAction = action
            G[state.index] = minValue
            A[state.index] = bestAction 
        
        optPolicy = {}
        for q in self.dfa.states:
            optPolicy[q] = {}
        
        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][str(self.mdp.states[j].anchor[1])] = A[j]
            if printPolicy == True:
                if self.mdp.states[j].anchor[0] == None:
                    print("self.mdp.states[j].anchor[0] is None")
                if self.mdp.states[j].anchor[1] == None:
                    print("self.mdp.states[j].anchor[1] is None")
                if A[j] == None:
                    print("A[j] is None")
                    A[j] = "None"
                print("\pi("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+A[j])
                print("M("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j]))
              
        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed")
                      
        time_elapsed = (time.time() - time_start)              
                      
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[self.mdp.initialState.index], time_elapsed)
    
    
    def optimalPolicyInfiniteHorizonForLayeredDAG(self, epsilonOfConvergance, printPolicy, computeAvoidableActions = False):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        
        #for state in self.mdp.states:
            #state.computeAvailableActions()
            
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
                    G[j][0] = G[j][1] =  float("inf")
                    A[j] = "DeadEnd"  
                    print("Set optimal action for dead end state "+self.mdp.states[j].name)
        
        
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
            #print("Start Computing Optimal Policy for states within SCC "+scc.name)
            while dif > epsilonOfConvergance:
                numIterations += 1
                maxDif = 0
                for state in scc.states:
                    if self.mdp.states[state.index].isGoal == True:
                        continue
                
                    if self.mdp.states[state.index].reachable == False:
                        continue
                    
                    if computeAvoidableActions == True and self.mdp.states[j].aGoalIsReachable == False:
                        A[j] = "DeadEnd"
                        G[j][0] = G[j][1] = float("inf")
                        continue
                                
                    minVal = float_info.max
                    optAction = ""
                    
                    if len(state.availableActions) == 0:
                        print("Number of available actions of state "+str(state)+" is "+str(len(state.availableActions)))
                                   
                    for action in state.availableActions:
                        if computeAvoidableActions == True:
                            if action in state.avoidActions:
                                continue
                        
                        val = 0.0
                        
                        if  state.isGoal == False:
                            val += 1
                        for tran in state.actionsTransitions[action]:
                            #term = G[tran.dstState.index][1]*self.mdp.conditionalProbability(tran.dstState.index, state.index, action)
                            term = G[tran.dstState.index][1]*tran.probability
                            val += term
                        #r += 1
                        #print("r="+str(r))
                        if val < minVal:
                            minVal = val
                            optAction = action    
                
                #if minVal-G[j][0] > maxDif:
                    #maxDif = minVal-G[j][0]
                
                    maxDif = max(maxDif, minVal-G[state.index][0])
                
                    if state == scc.Leader:
                        dif = minVal-G[state.index][0]
                
                    G[state.index][0] = minVal
                    A[state.index] = optAction
                
                for s in scc.states:
                    G[s.index][1] = G[s.index][0]
                dif = maxDif
                print("numIterations="+str(numIterations)+", dif = "+str(dif))
            print("Optimal Policy for states within SCC "+scc.name+" was computed")
                
   
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
            print("Initial state is: "+str(self.mdp.initialState.name))
            
        time_elapsed = (time.time() - time_start)
                      
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)
    

    def optimalPolicyInfiniteHorizon(self, epsilonOfConvergance, printPolicy, computeAvoidableActions = False):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        
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
        
        dif = float_info.max
                
        numIterations = 0
        
        r = 0
        
        time_start = time.time()
        
        while dif > epsilonOfConvergance:
            numIterations += 1
            
            print("dif="+str(dif))
            
            #print("r="+str(r))
            
            #if numIterations > 1000:
                #break
            maxDif = 0            
            
            for j in range(n):
                if self.mdp.states[j].isGoal == True:
                    continue
                
                if self.mdp.states[j].reachable == False:
                    continue
                
                if computeAvoidableActions == True and self.mdp.states[j].aGoalIsReachable == False:
                    continue
                 
                                
                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]
                
                #print("r="+str(r))
                
                for action in state.availableActions:
                    if computeAvoidableActions == True:
                        if action in state.avoidActions:
                            continue
                    val = 0.0
                    if  state.isGoal == False:
                        val += 1
                    #for k in range(n):
                    #    term = G[k][1]*self.mdp.conditionalProbability(k, j, action)
                    #    val += term
                        
                    for tran in state.actionsTransitions[action]:
                        term = G[tran.dstState.index][1]*tran.probability
                        val += term
                        #r += 1
                        #print("r="+str(r))
                    if val < minVal:
                        minVal = val
                        optAction = action    
                
                #if minVal-G[j][0] > maxDif:
                    #maxDif = minVal-G[j][0]
                
                if j == 0:
                    dif = minVal-G[j][0]
                
                maxDif = max(maxDif, minVal-G[j][0])
                
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
                    print("\pi("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+A[j])
                    print("M("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j][0]))
              
        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed in "+str(numIterations)+" iterations")
            print("InitialState: "+self.mdp.initialState.name)
        
        time_elapsed = (time.time() - time_start)       
                      
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)
    
    
        
    
    def simulate(self, policy, printOutput =False):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible( policy, printOutput)
        else:
            return self.__simulate_markovStateInvisible( policy, printOutput)
        
    def simulate_greedyAlgorithm(self, printOutput =True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible_greedyalgorithm(printOutput)
        #else:
        #    return self.__simulate_markovStateInvisible( policy, printOutput)           
    
    def __simulate_markovStateVisible(self, policy, printOutput =True):
        story = ""
        #s = self.markovChain.nullState
        s = self.markovChain.initialState
        q = self.dfa.initial_state
        q_previous = q
        i = 0
        while True:
            if q in self.dfa.final_states:
                return (i, story)
            if (s.name in policy[q].keys()) == False:
                print("q="+q+", s="+s.name) 
            predictedEvent = policy[q][s.name] 
            s2 = self.markovChain.next_state(s)
            
            q_previous = q
            if s2.has_event(predictedEvent):
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            i += 1
            
            if printOutput == True:
                print(str(i)+". "+"q="+q_previous+", s="+s.name+", predicted="+predictedEvent+", actual="+str(s2.events)+", sNext="+s2.name+", recorded story="+story)  
            
            s = s2
        return (i, story)
    
    def __simulate_markovStateVisible_greedyalgorithm(self, printOutput =True, ):
        story = ""
        #s = self.markovChain.nullState
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
                print(str(i)+". "+"q="+q_previous+", s="+s.name+", predicted="+predictedEvent+", actual="+str(s2.events)+", sNext="+s2.name+", recorded story="+story)  
            
            s = s2
        return (i, story)    
    
    
    def __simulate_markovStateInvisible(self, policy, printOutput =True):
        story = ""
        sts = set()
        for i in range(len(self.markovChain.states)):
            if self.markovChain.initial_distribution[i]>0:
                sts.add(self.markovChain.states[i])
        q = self.dfa.initial_state
        q_previous = q
        i = 1
        s = self.markovChain.next_state(self.markovChain.nullState)
        m = self.mdp.initialState
        while True:
            if q in self.dfa.final_states:
                return (i-1, story)

            predictedEvent = policy[q][str(sts)] 
            s2 = self.markovChain.next_state(s)
            
            q_previous = q
            if s2.has_event(predictedEvent):
                q = self.dfa.transitions[q][predictedEvent]
                if q != q_previous:
                    story += predictedEvent
            if printOutput == True:
                print(str(i)+". "+"q="+q_previous+", s="+s.name+", predicted="+predictedEvent+", actual="+str(s2.events)+", possibleStates="+str(sts)+", sNext="+s2.name+", recorded story="+story)  
            if predictedEvent in s2.events:
                m = self.mdp.getNextStateForEventPositive(m, predictedEvent)
                sts = m.anchor[1]
            else:
                m = self.mdp.getNextStateForEventNegative(m, predictedEvent)
                sts = m.anchor[1]            
            i += 1
            s = s2
        return (i-1, story)
    
        