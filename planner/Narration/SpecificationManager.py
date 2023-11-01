from automata.fa.dfa import DFA
from automata.fa.nfa import NFA
import Utility.AutomataUtility as AU


class SpecificationManager:
    def __init__(self, specificationDFA):
        self.specificationDFA = specificationDFA
        
    
    @classmethod
    def dfa4SuperSequenceOf(cls, dfa):   
        states = dfa.states.copy()
        input_symbols = dfa.input_symbols.copy()
        transitions = dfa.transitions
        initial_state = dfa.initial_state
        final_states = dfa.final_states.copy()
        nfa_transitions = {}
        for q in states:
            nfa_transitions[q] = {}
            for a in input_symbols:
                nfa_transitions[q][a] = {transitions[q][a], q}
        
        nfa = NFA(states=set(states), input_symbols=input_symbols, transitions=nfa_transitions, initial_state=initial_state, final_states=final_states)
        
        superSeqDFA = DFA.from_nfa(nfa)
        
        return superSeqDFA
        
    @classmethod
    def dfaWithEditDistanceOf(cls, dfa, k):
        dfa2 = AU.levenshtein_automaton_for_a_dfa(dfa, k, dfa.input_symbols)
        dfaMin = dfa2.minify()
        return dfaMin
    
    @classmethod
    def dfaForUnion(cls, dfa1, dfa2):
        dfa = AU.union(dfa1, dfa2)
        dfaMin = dfa.minify()
        return dfaMin
    
    @classmethod
    def dfa4Intersection(cls, dfa1, dfa2):
        dfa = AU.intersection(dfa1, dfa2)
        dfaMin = dfa.minify()
        return dfaMin
    
    @classmethod
    def dfa4SuperSequenceOfIntersection(cls, dfa1, dfa2):
        dfaS1 = cls.dfa4SuperSequenceOf(dfa1)
        dfaS2 = cls.dfa4SuperSequenceOf(dfa2)
        dfa = cls.dfa4Intersection(dfaS1, dfaS2)
        dfaMin = dfa.minify()
        return dfaMin
    
    @classmethod
    def dfa4ContaintGoodShots(cls, dfa, alphabetMap, goodShotNeededList):
        states = []
        transitions = {}
        input_symbols = dfa.input_symbols
        
        for s in dfa.states:
            states.append(s)
            s_shdw = s+"_shdw"
            states.append(s_shdw)
            transitions[s] = {}
            transitions[s_shdw] = {}
            for a in input_symbols:
                transitions[s][a] = set()
                transitions[s][a].add(dfa.transitions[s][a]) 
                
                transitions[s][alphabetMap[a]] = set()
                transitions[s][alphabetMap[a]].add(s)
                if a in goodShotNeededList:
                    transitions[s][alphabetMap[a]].add(dfa.transitions[s][a]+"_shdw")
                
                transitions[s_shdw][a] = set()
                transitions[s_shdw][a].add(dfa.transitions[s][a]+"_shdw") 
                
                transitions[s_shdw][alphabetMap[a]] = set()
                transitions[s_shdw][alphabetMap[a]].add(dfa.transitions[s][a]+"_shdw")
        
        initial_state = dfa.initial_state
        final_states = set()
        for s in dfa.final_states:
            final_states.add(s+"_shdw")
   
        alphabet = set()
        for a in input_symbols:
            alphabet.add(a)
            alphabet.add(alphabetMap[a])
            
        nfa = NFA(states=set(states), input_symbols=alphabet, transitions=transitions, initial_state=initial_state, final_states=final_states)
        dfaOutput = DFA.from_nfa(nfa)
        return dfaOutput
    
    @classmethod
    def dfa4replaceletter(cls, dfa, a1, a2):
        states = []
        transitions = {}
        input_symbols = dfa.input_symbols
        
        for s in dfa.states:
            states.append(s)
            s_shdw = s+"_shdw"
            states.append(s_shdw)
            transitions[s] = {}
            transitions[s_shdw] = {}
            for a in input_symbols:
                transitions[s][a] = set()
                transitions[s][a].add(dfa.transitions[s][a])
                
                if a == a2:
                    transitions[s][a2].add(dfa.transitions[s][a1]+"_shdw")     
                
                transitions[s_shdw][a] = set()
                transitions[s_shdw][a].add(dfa.transitions[s][a]+"_shdw") 
                
        
        initial_state = dfa.initial_state
        final_states = set()
        for s in dfa.final_states:
            final_states.add(s)
            final_states.add(s+"_shdw")
   
        alphabet = set()
        for a in input_symbols:
            alphabet.add(a)
            
        nfa = NFA(states=set(states), input_symbols=alphabet, transitions=transitions, initial_state=initial_state, final_states=final_states)
        dfaOutput = DFA.from_nfa(nfa)
        return dfaOutput