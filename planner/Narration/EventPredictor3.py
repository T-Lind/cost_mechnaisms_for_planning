from automata.fa.dfa import DFA
from Markov.MarkovChain3 import MarkovChain3
from Markov.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from sys import float_info
import queue 


import time





class EventPredictor3:
       
    def __init__(self, markovChain, transitionSystem, dfa, filter, eventList, verbose):
        self.dfa = dfa
        self.transitionSystem  = transitionSystem
        self.markovChain = markovChain
        self.filter = filter
        self.dfa = dfa
        self.eventList = eventList
        self.verbose = verbose
        self.mdp = None  
        self.special_noEvent = "E"
        self.createActionsList()
        #self.__createProductAutomaton_singleInitialState()
        self.__createProductAutomaton_singleInitialState_onlyreachables()
        
        #else:
            #self.__createProductAutomaton_WhereCurrentMarkovStateInvisible()    
    
    def createActionsList(self):
        actions = []
        for e in self.eventList:
            for w in self.transitionSystem.states:
                a = (w, e)  
                actions.append(a)
        for w in self.transitionSystem.states:
            a = (w, self.special_noEvent)    
            actions.append(a)
        self.actions = actions   
    
               
    def __createProductAutomaton_singleInitialState(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        ts = self.transitionSystem
        ft = self.filter
        mdp = MDP()
        self.mdp = mdp
        
        mdp.actions = self.actions
        mdp.special_noEvent = self.special_noEvent
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState, ts.initial_state, ft.initial_state)
        x0 = MDPState(dfa.initial_state+"_"+mc.initialState.name +"_"+ ts.initial_state +"_"+ ft.initial_state, t0)
        x0.is_initial = True
        mdp.addState(x0)
        mdp.initialState = x0
        
        """
        Create the state space of the MDP. 
        The state space of the MDP is V = S \times W \times Q \times V where 'Q' is the state space of the DFA specification and 'S' is the state space of the Markov Chain,
        'W' is the state space of the transition system, and V is the state space of the filter 
        """
        for w in ts.states:
            for s in mc.states:
                for q in dfa.states:
                    for v in ft.states:
                        if w == ts.initial_state and s == mc.initialState and q == dfa.initial_state and v == ft.initial_state:
                            continue
                        if w == ts.initial_state and s == mc.initialState:
                            continue
                        t = (q, s, w, v)
                        x = MDPState(q+"_"+s.name+"_"+w+"_"+v, t)
                        mdp.addState(x)
                        if q in dfa.final_states:
                            x.is_goal = True
                            mdp.setAsGoal(x)
                            x.weightBVector = ft.output_function[v]
        
        selfLoopAdded = {}
        for q in dfa.states:
            selfLoopAdded[q] = False      

        print("len(mdp.states): "+str(len(mdp.states)))
        print("len(mdp.actions): "+str(len(mdp.actions)))

        """
        Create transitions
        """           
        for x1 in mdp.states:
            #if v1 == v0:
            #    continue
            
            if x1.is_goal:
                continue
            
            q = x1.anchor[0]
            s = x1.anchor[1]
            w = x1.anchor[2]
            v = x1.anchor[3]
            
            for x2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """              
                if x2 == x0:    
                   continue
                
                q2 = x2.anchor[0]
                s2 = x2.anchor[1]
                w2 = x2.anchor[2]
                v2 = x2.anchor[3]
                
                if mc.getTransitionProbability(s, s2) == 0:
                    continue
                
                if not ts.hasTransition(w, w2):
                    continue
                
                #print("Checking MDP states: "+str(v1)+", "+str(v2))
                
                transProb = mc.getTransitionProbability(s, s2)
                
                for a in self.actions:
                    w3 = a[0] 
                    e = a[1]
                    if w3 != w2:
                        continue
                    
                    #ts.labeling_func[w2].add(e)
                    #print("e: "+e)
                    #print("ts.labeling_func[w2]: "+str(ts.labeling_func[w2]))
                    ex_label = ts.labeling_func[w2].copy()
                    ex_label.add(e)
                    #print("ex_label: "+str(ex_label))
                    letterCaptured = ts.getLetter(ex_label)
                    #letterNotCaptured = ts.getLetter(ts.labeling_func[w2])
                    letterNotCaptured = ts.getLetter(ts.labeling_func[w2].copy())
                    #print("letterCaptured: "+str(letterCaptured))
                    #print("letterNotCaptured: "+str(letterNotCaptured))
                    if e != self.special_noEvent and s2.hasEventLoc(e, w2) and dfa.transitions[q][letterCaptured] == q2 and ft.transitions[v][letterCaptured] == v2 and (q not in dfa.final_states):
                        evProb = s2.getEventLocPair(e, w3)[2]
                        trans = MDPTransition(x1, x2, a, set(), transProb*evProb)
                        mdp.addTransition(trans)
                        if dfa.transitions[q][letterNotCaptured] == q2 and ft.transitions[v][letterNotCaptured] == v2:
                            trans.probability = transProb
                        #print("Case 1: "+str(trans))
                    elif e != self.special_noEvent and  dfa.transitions[q][letterNotCaptured] == q2 and ft.transitions[v][letterNotCaptured] == v2 and (q not in dfa.final_states):
                        evProb = s2.probabilityOfEventLoc(e, w2) 
                        if evProb != 1: 
                            trans = MDPTransition(x1, x2, a, set(), transProb*(1-evProb))
                            mdp.addTransition(trans)
                            #print("Case 2: "+str(trans)+", transProb: "+ str(transProb)+", evProb: "+ str(evProb))
                    elif e == self.special_noEvent and dfa.transitions[q][letterNotCaptured] == q2 and ft.transitions[v][letterNotCaptured] == v2 and (q not in dfa.final_states):
                        if transProb > 0:
                            trans = MDPTransition(x1, x2, a, set(), transProb)
                            mdp.addTransition(trans)
                        #print("Case 3: "+str(trans))
                    #elif q == q2 and v == v2 and (q in dfa.final_states):
                    #    trans = MDPTransition(x1, x2, a, set(), 1)
                    #    mdp.addTransition(trans)                        
                                
        """
        Make goal states absorbing
        """
        for x in mdp.goalStates:
            if not x.is_goal:
                continue
            for a in mdp.actions:
                trans = MDPTransition(x, x, a, set(), 1)
                mdp.addTransition(trans)
        
            
                        #selfLoopAdded[q] = True
        #self.mdp.recognizeReachableStates()
        
        #self.mdp.makeGoalStatesAbsorbing()
        
        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()
        
        self.mdp.makeObservationFunction()
        
        self.mdp.check_transition_function()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states")
            print("----------------------------------------------") 
                        
                 

    def __createProductAutomaton_singleInitialState_onlyreachables(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        ft = self.filter
        mc = self.markovChain
        ts = self.transitionSystem
        dfa = self.dfa
        mdp = MDP()
        self.mdp = mdp
        locs =  ts.states
        
        mdp.actions = self.actions
        
        mdp.special_noEvent = self.special_noEvent
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState, ts.initial_state, ft.initial_state)
        x0 = MDPState(dfa.initial_state+"_"+mc.initialState.name +"_"+ ts.initial_state +"_"+ ft.initial_state, t0)
        x0.weightBVector = ft.output_function[ft.initial_state]
        #v0.evidence_distribution = mc.initialState.evidence_distribution
        #q0 = dfa.initial_state
        x0.is_initial = True
        mdp.addState(x0)
        mdp.initialState = x0
        
        queue = []
        queueNames = []
        queue.append(x0)
        queueNames.append(x0.name)
        
        cnt = 0
        i = 0
        while queue:
            i += 1
            print("Iteration "+str(i))
            x = queue.pop(0)
            xName = queueNames.pop(0)
            t = x.anchor
            q = t[0]
            s = t[1]
            w = t[2]
            v = t[3]
            if x.is_goal == True:
                for a in self.actions:
                    trans = MDPTransition(x, x, a, set(), 1)
                    mdp.addTransition(trans)
                continue
                    
            for s2 in mc.states:
                prob = mc.getTransitionProbability(s, s2)
                if prob == 0:
                   continue
                for a in self.actions:
                    w2 = a[0]
                    e = a[1]
                    if not ts.hasTransition(w, w2):
                        continue
                    
                    
                    
                    
                    ex_label = ts.labeling_func[w2].copy()
                    
                    #print(f"e: {e}")
                    #print(f"ex_label: {ex_label}")
                    
                    ex_label.add(e)
                    
                    #print(f"ex_label: {ex_label}")
                    
                    #print(f"ts.alphabet: {ts.alphabet}")
                    
                    
                    
                    letterCaptured = ts.getLetter(ex_label)
                    
                    letterNotCaptured = ts.getLetter(ts.labeling_func[w2].copy())
                    
                    if e != self.special_noEvent and s2.hasEventLoc(e, w2):
                        #print(f"letterCaptured: {letterCaptured}")
                        q2 = dfa.transitions[q][letterCaptured]
                        v2 = ft.transitions[v][letterCaptured] 
                        x2Name = q2+"_"+s2.name+"_"+w2+"_"+v2
                        x2wasInMDP = False
                        if x2Name in mdp.statesDictByName.keys():
                            x2 = mdp.statesDictByName[x2Name]
                            x2wasInMDP = True
                        else:
                            t2 = (q2, s2, w2, v2)
                            x2 = MDPState(x2Name, t2)
                            x2.weightBVector = ft.output_function[v2]
                            mdp.addState(x2)
                            cnt += 1
                            if q2 in dfa.final_states:
                                x2.is_goal = True
                                mdp.setAsGoal(x2)
                        if x2Name not in queueNames and x2wasInMDP == False:
                            queue.append(x2)       
                            queueNames.append(x2Name)
                        evProb = s2.getEventLocPair(e, w2)[2]
                        trans = MDPTransition(x, x2, a, set(), prob*evProb)
                        mdp.addTransition(trans)
                        #if dfa.transitions[q][letterNotCaptured] == q2 and ft.transitions[v][letterNotCaptured] == v2:
                        #    trans.probability = prob
                    if e != self.special_noEvent:# and (dfa.transitions[q][letterNotCaptured] != q2 or ft.transitions[v][letterNotCaptured] != v2):  
                        q2 = dfa.transitions[q][letterNotCaptured]
                        v2 = ft.transitions[v][letterNotCaptured] 
                        x2Name = q2+"_"+s2.name+"_"+w2+"_"+v2
                        x2wasInMDP = False
                        if x2Name in mdp.statesDictByName.keys():
                            x2 = mdp.statesDictByName[x2Name]
                            x2wasInMDP = True
                        else:
                            t2 = (q2, s2, w2, v2)
                            x2 = MDPState(x2Name, t2)
                            x2.weightBVector = ft.output_function[v2]
                            mdp.addState(x2)
                            cnt += 1
                            if q2 in dfa.final_states:
                                x2.is_goal = True
                                mdp.setAsGoal(x2)
                        if x2Name not in queueNames and x2wasInMDP == False:
                            queue.append(x2)       
                            queueNames.append(x2Name)
                        evProb = s2.probabilityOfEventLoc(e, w2) 
                        if evProb != 1: 
                            trans = mdp.getTransition(x, x2, a)
                            if trans == None:
                                trans = MDPTransition(x, x2, a, set(), prob*(1-evProb))
                                mdp.addTransition(trans)
                            else:
                                trans.probability += prob*(1-evProb)
                            
                            #print("Case 2: "+str(trans)+", transProb: "+ str(transProb)+", evProb: "+ str(evProb))
                    if e == self.special_noEvent: 
                        q2 = dfa.transitions[q][letterNotCaptured]
                        v2 = ft.transitions[v][letterNotCaptured] 
                        x2Name = q2+"_"+s2.name+"_"+w2+"_"+v2
                        x2wasInMDP = False
                        if x2Name in mdp.statesDictByName.keys():
                            x2 = mdp.statesDictByName[x2Name]
                            x2wasInMDP = True
                        else:
                            t2 = (q2, s2, w2, v2)
                            x2 = MDPState(x2Name, t2)
                            x2.weightBVector = ft.output_function[v2]
                            mdp.addState(x2)
                            cnt += 1
                            if q2 in dfa.final_states:
                                x2.is_goal = True
                                mdp.setAsGoal(x2)
                        if x2Name not in queueNames and x2wasInMDP == False:
                            queue.append(x2)       
                            queueNames.append(x2Name)
                        if prob > 0:
                            trans = MDPTransition(x, x2, a, set(), prob)
                            mdp.addTransition(trans)
                    #elif s2.hasEvent(e) and q == q2 and (q not in dfa.final_states):
                    #    if w*(1-p)>0:
                    #        trans = MDPTransition(v, v2, e, s2.events, w*(1-p))
                    #        mdp.addTransition(trans)
                    #elif p == 0 and q == q2 and (q not in dfa.final_states):
                    #    trans = MDPTransition(v, v2, e, s2.events, w)
                    #    mdp.addTransition(trans)                    
                    
                

#    
        print("Number of added states = "+str(cnt))
        
        print(f"Number of goal states: {len(mdp.goalStates)}")
        
        self.mdp.removeUnReachableStates()
        self.mdp.computeStatesAvailableActions()
        
        self.mdp.makeObservationFunction()
        
        self.mdp.check_transition_function()
        
        
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states and "+str(len(mdp.transitions))+" transitions")
            print("----------------------------------------------") 
            
     
                        
                        
    def optimalPolicyFiniteHorizon(self, F, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        G = [[0.0 for j in range(n)] for i in range(F+1)] 
        A = [["" for j in range(n)] for i in range(F+1)]
        
        for j in range(n):
            if (self.mdp.states[j].is_goal):
                G[F][j] = 0.0
                A[F][j] = "STOP"
            else:
                G[F][j] = 10000.0
            
        for i in range(F-1, -1, -1):
            #print(i)
            for j in range(n):
                if self.mdp.states[j].is_goal == True:
                    A[i][j] = "STOP"
                    G[i][j] = 0.0
                    continue
                                
                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]
                
                for action in self.eventList:
                    val = 0.0
                    if  state.is_goal == False:
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
    
    
    
    def optimalPolicyInfiniteHorizonForDAG(self, epsilonOfConvergance, printPolicy, outputFileName="computedPolicyMinExpNumOfSteps.txt"):
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
            if state.is_goal:
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
        
        f = open(outputFileName, "w")  
        
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
                
                f.write("\pi("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+A[j])
                f.write("M("+self.mdp.states[j].anchor[0]+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j]))
              
        
        f.close()
        
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
            if (self.mdp.states[j].is_goal):
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
            if scc.states[0].is_goal:
                continue
            firstTimeComputeDif = True
            #print("Start Computing Optimal Policy for states within SCC "+scc.name)
            while dif > epsilonOfConvergance:
                numIterations += 1
                maxDif = 0
                for state in scc.states:
                    if self.mdp.states[state.index].is_goal == True:
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
                        
                        if  state.is_goal == False:
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
    

    def optimalPolicyInfiniteHorizon(self, epsilonOfConvergance, printPolicy, computeAvoidableActions = False, outputFileName="optimalPolicyMinNumOfSteps.txt"):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        
        if computeAvoidableActions == True:
            if self.mdp.availableActionsComputed == False:
                self.mdp.computeAvoidableActions()

        G = [[0.0 for j in [0, 1]] for i in range(n)] 
        A = ["" for j in range(n)]
        
        for j in range(n):
            if (self.mdp.states[j].is_goal):
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
                if self.mdp.states[j].is_goal == True:
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
                    if  state.is_goal == False:
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
            
        f = open(outputFileName, "w")
        
        for j in range(n):
            optPolicy[self.mdp.states[j].anchor[0]][str(self.mdp.states[j].anchor[1])] = A[j]
            if printPolicy == True:
                    print("\pi("+str(self.mdp.states[j].anchor[0])+","+str(self.mdp.states[j].anchor[1])+")="+str(A[j]))
                    print("M("+str(self.mdp.states[j].anchor[0])+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j][0]))
                    f.write("\pi("+str(self.mdp.states[j].anchor[0])+","+str(self.mdp.states[j].anchor[1])+")="+str(A[j])+"\n")
                    f.write("M("+str(self.mdp.states[j].anchor[0])+","+str(self.mdp.states[j].anchor[1])+")="+str(G[j][0])+"\n")
                    
        f.close()
              
        if self.verbose == True:
            print("optimal policy for infinite horizon has been computed in "+str(numIterations)+" iterations")
            print("InitialState: "+self.mdp.initialState.name)
        
        time_elapsed = (time.time() - time_start)       
                      
        #return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index], time_elapsed)
    
    
        
    
    def simulate(self, policy, printOutput =False):
        #if self.current_markov_state_visible == True:
        return self.__simulate_markovStateVisible( policy, printOutput)
        #else:
        #    return self.__simulate_markovStateInvisible( policy, printOutput)
        
    def simulate_greedyAlgorithm(self, printOutput =True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible_greedyalgorithm(printOutput)
        #else:
        #    return self.__simulate_markovStateInvisible( policy, printOutput)           
    
    def __simulate_markovStateVisible(self, policy, printOutput =True):
        story = ""
        storyLocation = ""
        #s = self.markovChain.nullState
        s = self.markovChain.initialState
        q = self.dfa.initial_state
        v = self.filter.initial_state
        w = self.transitionSystem.initial_state
        q_previous = q
        i = 0
        while True:
            if i > 2000:
                return (2000, story, "not successful", self.filter.output_function[v])
            if q in self.dfa.final_states:
                return (i, story, storyLocation, self.filter.output_function[v])
            mdp_state = q+"_"+s.name +"_"+ w +"_"+ v
            #if (mdp_state in policy[q].keys()) == False:
            #    print("q="+q+", s="+s.name) 
            #location, predictedEvent = policy[q][s.name][w.name]
            location, predictedEvent = policy[mdp_state] 
            s2 = self.markovChain.next_state(s)
            occuredEvents = s2.occurEvents(location)
            
            occured = False
            
            ex_label = self.transitionSystem.labeling_func[location].copy()
            if predictedEvent != self.special_noEvent:
                if predictedEvent in occuredEvents:
                    ex_label.add(predictedEvent)
                    occured = True
            #ex_label.add(e)
            letter = self.transitionSystem.getLetter(ex_label)
                                    
            q_previous = q
            q = self.dfa.transitions[q][letter]
            v = self.filter.transitions[v][letter]
            #if q != q_previous:
            #    story += predictedEvent
            if occured:
                if q != q_previous:
                    story += predictedEvent
                    storyLocation += "{"+predictedEvent+","+location+"}"
                else:
                    storyLocation += "{"+location+"}"
            else:
                storyLocation += "{"+location+"}"
            
            i += 1
            
            if printOutput == True:
                print(str(i)+". "+"q="+q_previous+", s="+s.name+", tranSystemState="+w+", locationToGo = "+location+", predicted="+predictedEvent+", occured="+str(occured)+", sNext="+s2.name+", recorded story="+story+", recorded story location="+storyLocation)  
            
            s = s2
            w = location
        return (i, story, storyLocation, self.filter.output_function[v])
    
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
    
        