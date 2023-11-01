from automata.fa.dfa import DFA

class Filter(DFA):
    def __init__(self, states, input_symbols, transitions, initial_state, colors, output_function):
        super().__init__(states=states, input_symbols=input_symbols, transitions=transitions, initial_state=initial_state, final_states=set())
        self.colors = colors
        self.output_function = output_function
        
    def printAll(self):
        print("------------------------------------- print DFA ------------------------------------------------")
        print("Initial State: "+self.initial_state)
        print("States and Transitions:")
        for t in self.transitions.items():
            print(t)
        print("Colors="+str(self.colors))
        print("Output function:")
        for f in self.output_function.items():
            print(f)
        print("-------------------------------------------------------------------------------------------------")
        
    def convertOutpus2BooleanVectors(self):
        for v in self.states:
            color = self.output_function[v]
            booleanVals  = color.split('_')
            booleanVector = []
            for i in range(len(booleanVals)):
                if booleanVals[i] == "True":
                    booleanVector.append(True)
                else:
                    booleanVector.append(False)
            self.output_function[v]   =  booleanVector 
        
    def fromDFA(dfa):
        colors = [True, False]
        output_function = {}
        for s in dfa.states:
            if s in dfa.final_states:
                output_function[s] = True
            else:
                 output_function[s] = False
        filter = Filter(dfa.states, dfa.input_symbols, dfa.transitions, dfa.initial_state, colors, output_function)
        return filter
        
    def Product(filter1, filter2):
        states = []
        input_symbols = filter1.input_symbols
        initial_state = filter1.initial_state+"_"+filter2.initial_state
        print("initial_state of product: "+initial_state)
        anchorStates = [] 
        for s1 in filter1.states:
            for s2 in filter2.states:
                s = s1+"_"+s2
                anchor = (s1, s2) 
                states.append(s)
                anchorStates.append(anchor) 
        
        transitions = {}
        for i in range(len(states)):
            s = states[i]
            transitions[s] = {}
            anchor = anchorStates[i]
            for a in input_symbols:
                s2 = filter1.transitions[anchor[0]][a] +"_"+filter2.transitions[anchor[1]][a]
                transitions[s][a] = s2
        
        
        colors = []
        for c1 in filter1.colors:
            for c2 in filter2.colors:
                c = str(c1) +"_" +str(c2)
                colors.append(c)
        print("colors of product: "+str(colors))
        output_function = {}
        for i in range(len(states)):
            s = states[i]
            anchor = anchorStates[i]
            c = str(filter1.output_function[anchor[0]]) +"_"+str(filter2.output_function[anchor[1]])
            output_function[s] = c
        
        f = Filter(set(states), input_symbols, transitions, initial_state, colors, output_function)
        
        return f
    
    def ProductList(filterList):
        n = len(filterList)
        if n == 0:
            return None
        if n == 1:
            return filterList[0]
        f1 = filterList[0]
        for i in range(1, n):
            f2 = filterList[i]
            print("Product to "+str(i)+"'th filter is done")
            f1 = Filter.Product(f1, f2) 
        return f1
        
    






    