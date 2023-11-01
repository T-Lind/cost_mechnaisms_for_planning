from automata.fa.dfa import DFA
from hazhar.Markov.MarkovChain2 import MarkovChain2
from hazhar.Markov.MarkovDecisionProcess import MDP, MDPState, MDPTransition
from sys import float_info
import queue 

import math

class EventPredictor:
       
    def __init__(self, dfa, markovChain, style, eventList, verbose, story,
                 currentMarkovStateVisibile=True, monolithic = False):
        self.dfa = dfa
        self.markovChain = markovChain
        self.story = story
        self.eventList = eventList
        self.verbose = verbose
        self.mdp = None
        self.mdp_actions = None
        self.style = style
        self.discount = 0.8
        self.currentMarkovStateVisibile = currentMarkovStateVisibile
        if monolithic:
            self.__createProductAutomaton_singleInitialState_monolithic()
        else:
            if currentMarkovStateVisibile == True:
                self.__createProductAutomaton_singleInitialState()
                # self.__createProductAutomaton_singleInitialState_monolithic()
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
        v0 = MDPState(dfa.initial_state+","+"_"+mc.initialState.name, t0)
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
                v = MDPState(q+"_"+s.name, t)
                mdp.addState(v)
                if q in dfa.final_states:
                    v.isGoal = True
                    mdp.setAsGoal(v)
                
              
        """
        Create transitions from the initial state of the MDP, which is (q0, _)
        """
        for i in range(0, len(mc.states)):
            if mc.initialDistribution[i] == 0:
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
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states")
            print("----------------------------------------------")
            
    def __createProductAutomaton_singleInitialState(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.hasEvidence = mc.hasEvidence
        mdp.evidenceList = mc.evidenceList
        self.mdp = mdp
        
        mdp.actions = self.eventList
        
        """
        Create the initial state of the MDP
        """
        t0 = (dfa.initial_state, mc.initialState)
        v0 = MDPState(dfa.initial_state+"_"+mc.initialState.name, t0)
        v0.evidenceDistribution = mc.initialState.evidenceDistribution
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
                v = MDPState(q+"_"+s.name, t)
                v.evidenceDistribution = s.evidenceDistribution
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
                    elif not s2.has_event(e) and q == q2 and (q not in dfa.final_states):
                        trans = MDPTransition(v1, v2, e, s2.events, w)
                        mdp.addTransition(trans)
                
        #         if v1 == v2:
        #             for e1 in self.eventList:
        #                 p = mc.getTransitionProbability(s, s2)
        #                 trans = MDPTransition(v1, v2, e1, s2.events, p)
        #                 mdp.addTransition(trans)
        #             continue
        #
        #         for e in s2.events:
        #             if dfa.transitions[q][e] == q2:
        #                 p = mc.getTransitionProbability(s, s2)
        #                 trans = MDPTransition(v1, v2, e, s2.events, p)
        #                 mdp.addTransition(trans)
        #         if q == q2:
        #             for e1 in self.eventList:
        #                 if e1 in s2.events:
        #                     continue
        #                 #if selfLoopAdded[q]:
        #                 #    continue
        #                 #selfLoopAdded[q] = True
        #                 p = mc.getTransitionProbability(s, s2)
        #                 trans = MDPTransition(v1, v2, e1, s2.events, p)
        #                 mdp.addTransition(trans)
        #                 #selfLoopAdded[q] = True
        # #self.mdp.recognizeReachableStates()
        #
        # #self.mdp.makeGoalStatesAbsorbing()
        
        if self.verbose == True:
            print("the product automata has been computed. It has "+str(len(mdp.states))+" states")
            print("----------------------------------------------")

    def __createProductAutomaton_singleInitialState_monolithic(self):
        if self.verbose == True:
            print("---------------creating product automata-----------------")
        dfa = self.dfa
        mc = self.markovChain
        mdp = MDP()
        mdp.hasEvidence = mc.hasEvidence
        mdp.evidenceList = mc.evidenceList
        self.mdp = mdp
        # print('Got till here')
        #
        # print(self.eventList)
        # print(self.style.dictionary)

        actions = set()

        for event in self.style.dictionary:
            for style in self.style.dictionary[event]:
                # print(event + '_' + style)
                actions.add(event + '_' + style)

        # print(actions)
        mdp.actions = actions
        self.mdp_actions = list(actions)
        # print(self.mdp_actions)

        """
        Create the initial state of the MDP
        """
        style_initial = '.'.join([self.style.blank for i in range(self.style.k - 1)])
        # print(style_initial)
        t0 = (dfa.initial_state, mc.initialState, style_initial)
        # print(t0)
        # print(dfa.initial_state + "_" + mc.initialState.name + "_" + style_initial)
        v0 = MDPState(dfa.initial_state + "_" + mc.initialState.name + "_" + style_initial, t0)
        v0.evidenceDistribution = mc.initialState.evidenceDistribution
        q0 = dfa.initial_state
        v0.isInitial = True
        mdp.addState(v0)
        mdp.initialState = v0

        """
        Create the state space of the MDP.
        The state space of the MDP is V = (Q \cup S) \cup {(q0, _)}.
        """
        style_k = [self.style.styles for i in range(self.style.k -1)]
        style_states_t = list(itertools.product(*style_k))
        style_states = []
        for ss in style_states_t:
            sst = '.'.join(ss)
            if self.style.is_valid(sst):
                style_states.append(sst)

        # print(style_states)
        # print(style_initial)

        for q in dfa.states:
            for s in mc.states:
                for sty in style_states:
                    if q == dfa.initial_state and s == mc.initialState and sty == style_initial:
                        continue
                    t = (q, s, sty)
                    v = MDPState(q + "_" + s.name + "_" + sty, t)
                    v.evidenceDistribution = s.evidenceDistribution
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
            st = v1.anchor[2]

            if q == '*':
                continue

            valid_evnts = self.story.get_outgoing_events(q)
            # print(q)
            # print(valid_evnts)

            for v2 in mdp.states:
                """
                There should be no transition to the initial state of the MDP
                """
                # if v2 == v0:
                #   continue
                q2 = v2.anchor[0]
                s2 = v2.anchor[1]
                st2 = v2.anchor[2]
                if mc.getTransitionProbability(s, s2) == 0:
                    continue

                w = mc.getTransitionProbability(s, s2)

                for e_s in self.mdp_actions:
                    e_s_split = e_s.split("_")
                    act_e = e_s_split[0]
                    act_s = e_s_split[1]
                    cur_sty = '.'.join(st.split('.')[1:]) + '.' + act_s
                    # print(act_e)
                    if act_e in valid_evnts:
                        # print(q, act_e)

                        if s2.has_event(act_e) and dfa.transitions[q][act_e] == q2 and (q != q2) and (q not in dfa.final_states):
                            if cur_sty == st2:
                                ep = s2.getEventPair(act_e)
                                p = ep[1]
                                # print(w*p)
                                if w * p > 0:
                                    trans = MDPTransition(v1, v2, e_s, s2.events, w * p)
                                    mdp.addTransition(trans)

                        elif s2.has_event(act_e) and dfa.transitions[q][act_e] == q2 and (q == q2) and (q not in dfa.final_states):
                            if cur_sty == st2:
                                ep = s2.getEventPair(act_e)
                                p = ep[1]
                                # print(w*p)
                                trans = MDPTransition(v1, v2, e_s, s2.events, w)
                                mdp.addTransition(trans)
                    #         ep = s2.getEventPair(e)
                    #         p = ep[1]
                    #         trans = MDPTransition(v1, v2, e, s2.events, w * p)
                    #         mdp.addTransition(trans)
                        elif s2.has_event(act_e) and q == q2 and (q not in dfa.final_states):
                            ep = s2.getEventPair(act_e)
                            p = ep[1]
                            # print(w * (1-p))
                            if w * (1 - p) > 0:
                                trans = MDPTransition(v1, v2, e_s, s2.events, w * (1 - p))
                                mdp.addTransition(trans)
                        elif not s2.has_event(act_e) and q == q2 and (q not in dfa.final_states):
                            # print(w)
                            trans = MDPTransition(v1, v2, e_s, s2.events, w)
                            mdp.addTransition(trans)


        if self.verbose == True:
            print("the product automata has been computed. It has " + str(len(mdp.states)) + " states")
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
    

    
    def optimalPolicyInfiniteHorizon(self, epsilonOfConvergance, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)
        # print(self.mdp.transitionsDict)
        

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
        
        while dif > epsilonOfConvergance:
            numIterations += 1
            
            print("dif="+str(dif))
            
            #print("r="+str(r))
            
            #if numIterations > 1000:
                #break
                        
            
            for j in range(n):
                if self.mdp.states[j].isGoal == True:
                    continue
                
                if self.mdp.states[j].reachable == False:
                    continue
                                
                minVal = float_info.max
                optAction = ""
                state = self.mdp.states[j]
                
                #print("r="+str(r))
                
                for action in self.eventList:
                    val = 0.0
                    if  state.isGoal == False:
                        val += 1
                    for k in range(n):
                        term = G[k][1]*self.mdp.conditionalProbability(k, j, action)
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
                
                G[j][0] = minVal
                A[j] = optAction
                
            for j in range(n):
                G[j][1] = G[j][0]
                
            
        
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


    def optimalPolicyInfiniteHorizon_monolithic(self, epsilonOfConvergance, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.mdp.states)

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

        while dif > epsilonOfConvergance:
            numIterations += 1

            print("dif=" + str(dif))

            # print("r="+str(r))

            # if numIterations > 1000:
            # break

            for j in range(n):
                if self.mdp.states[j].isGoal == True:
                    continue

                if self.mdp.states[j].reachable == False:
                    continue

                maxVal = -float_info.max
                optAction = ""
                state = self.mdp.states[j]

                # print("r="+str(r))

                for action in self.mdp_actions:
                    val = 0.0
                    # if state.isGoal == False:
                    #     val += 1
                    for k in range(n):
                        # print(state.anchor[2], action.split('_')[1])
                        term =  self.mdp.conditionalProbability_monolithic(k, j, action) * \
                                (G[k][1] + self.style.get_transition_weight(action.split('_')[1], state.anchor[2]))
                        val += term
                        # r += 1
                        # print("r="+str(r))
                    # print('##################################')
                    # print(val, maxVal)
                    if val > maxVal:
                        maxVal = val
                        optAction = action

                        # if minVal-G[j][0] > maxDif:
                    # maxDif = minVal-G[j][0]

                if j == 0:
                    dif = maxVal - G[j][0]
                    # print(G[j][0], maxVal)
                    # dif = G[j][0] - maxVal

                G[j][0] = maxVal
                A[j] = optAction

            for j in range(n):
                G[j][1] = G[j][0]

            # print("TTTTTTTTTTTTTTTTTTTT")
            # print(dif)

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

        # return G[0][self.mdp.initialState.index]
        return (optPolicy, G, G[0][self.mdp.initialState.index])
    
    
        
    
    def simulate(self, policy, printOutput =True):
        if self.currentMarkovStateVisibile == True:
            return self.__simulate_markovStateVisible( policy, printOutput)
        else:
            return self.__simulate_markovStateInvisible( policy, printOutput)   
    
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
            s2 = self.markovChain.nextState(s)
            
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
            if self.markovChain.initialDistribution[i]>0:
                sts.add(self.markovChain.states[i])
        q = self.dfa.initial_state
        q_previous = q
        i = 1
        s = self.markovChain.nextState(self.markovChain.nullState)
        m = self.mdp.initialState
        while True:
            if q in self.dfa.final_states:
                return (i-1, story)

            predictedEvent = policy[q][str(sts)] 
            s2 = self.markovChain.nextState(s)
            
            q_previous = q
            if predictedEvent in s2.events:
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

    def expected_utility(self, a, s, u):
        n = len(self.mdp.states)
        sum_val = 0.0
        j = self.mdp.states.index(s)
        print("J = ", j)
        for i in range(n):
            sum_val = sum_val + u[self.mdp.states[i]] * self.mdp.conditionalProbability_monolithic(i, j, a)
        # return sum([p * u[s1] for (s1, p) in self.spa.transitions[s][a]])

    def optimalPolicyInfiniteHorizon_monolithic2(self, epsilonOfConvergence, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")

        n = len(self.mdp.states)
        u1 = dict([(s, 0) for s in self.mdp.states])
        pi = dict([(s, "") for s in self.mdp.states])

        while True:
            u = u1.copy()
            delta = 0

            # print("#######################################################")

            for j, s in enumerate(self.mdp.states):
                if self.mdp.states[j].anchor[0] == "*":
                    continue
                max_val = - math.inf
                temp = False
                # print(self.mdp.states[j])
                for a in self.mdp_actions:
                    if a.split("_")[0] in self.story.get_outgoing_events(self.mdp.states[j].anchor[0]):
                        val = 0.0
                        for k in range(n):
                            val = val + self.mdp.conditionalProbability_monolithic(k, j, a) * \
                                  (self.style.get_transition_weight(a.split('_')[1], s.anchor[2]) +
                                   self.discount * u[self.mdp.states[k]])

                        if val > max_val:
                            pi[s] = a
                        max_val = max(max_val, val)
                        temp = True
                if temp:
                    u1[s] = max_val
                    # print(max_val, temp)
                    delta = max(delta, abs(u1[s] - u[s]))
                # print(delta)
            print(delta)
            # print(epsilonOfConvergence * (1 - self.discount) / self.discount)
            if delta < epsilonOfConvergence * (1 - self.discount) / self.discount:
                # print("Completed calculating value iteration")
                # pi = {}
                # for s in self.mdp.states:
                #     print(s)
                #     pi[s] = argmax(self.mdp_actions, lambda a: self.expected_utility(a, s, u))
                # print(pi)
                return pi

    def optimalPolicyInfiniteHorizon_joint2(self, epsilonOfConvergence, printPolicy):
        if self.verbose == True:
            print("------computing optimal policy for finite horizon--------------")

        n = len(self.mdp.states)
        u1 = dict([(s, 1.0) for s in self.mdp.states])
        pi = dict([(s, "") for s in self.mdp.states])

        while True:
            u = u1.copy()
            delta = 0

            for j in range(n):
                if self.mdp.states[j].anchor[0] == "*":
                    continue

                if self.mdp.states[j].isGoal == True:
                    continue

                if self.mdp.states[j].reachable == False:
                    continue

                min_val = math.inf
                temp = False
                for a in self.eventList:
                    if a in self.story.get_outgoing_events(self.mdp.states[j].anchor[0]):
                        val = 0.0
                        if self.mdp.states[j].isGoal == False:
                            val = val + 1.0
                        for k in range(n):
                            val = val + self.mdp.conditionalProbability(k, j, a) * u[self.mdp.states[k]]

                        if val < min_val:
                            pi[self.mdp.states[j]] = a
                            min_val = val
                        temp = True
                if temp:
                    u1[self.mdp.states[j]] = min_val

                    delta = min(delta, abs(u1[self.mdp.states[j]] - u[self.mdp.states[j]]))
            print(delta)
            if delta < epsilonOfConvergence * (1 - self.discount) / self.discount:
                return pi