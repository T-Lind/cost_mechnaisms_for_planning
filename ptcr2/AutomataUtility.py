from typing import Union

from automata.fa.dfa import DFA
from automata.fa.nfa import NFA


def dfa_accepting_a_sequence(sequence, alphabet):
    states = []

    transitions = {}
    current_state = "q0"
    prev_state = current_state
    states.append(current_state)

    q_trap = "q_trap"
    states.append(q_trap)

    i = 1

    for a in sequence:
        current_state = "q_" + a + "_" + str(i)
        states.append(current_state)
        transitions[prev_state] = {}
        transitions[prev_state][a] = current_state
        for b in alphabet:
            if b != a:
                transitions[prev_state][b] = q_trap
        prev_state = current_state
        i = i + 1

    q_final = prev_state
    transitions[q_trap] = {}
    transitions[q_final] = {}

    for a in alphabet:
        transitions[q_trap][a] = q_trap
        transitions[q_final][a] = q_trap

    initial_state = states[0]
    final_states = set()
    final_states.add(q_final)

    dfa = DFA(states=set(states), input_symbols=alphabet, transitions=transitions, initial_state=initial_state,
              final_states=final_states)

    return dfa


def union(dfa_1, dfa_2):
    states = []
    s0 = "q0"
    states.append(s0)
    for s in dfa_1.states:
        states.append(s + "_1")
    for s in dfa_2.states:
        states.append(s + "_2")
    alphabet = dfa_1.input_symbols
    transitions = {}
    for s in dfa_1.states:
        transitions[s + "_1"] = {}
        for a in alphabet:
            transitions[s + "_1"][a] = set()
            transitions[s + "_1"][a].add(dfa_1.transitions[s][a] + "_1")
    for s in dfa_2.states:
        transitions[s + "_2"] = {}
        for a in alphabet:
            transitions[s + "_2"][a] = set()
            transitions[s + "_2"][a].add(dfa_2.transitions[s][a] + "_2")
    transitions[s0] = {}
    transitions[s0][''] = set()
    transitions[s0][''].add(dfa_1.initial_state + "_1")
    transitions[s0][''].add(dfa_2.initial_state + "_2")

    final_states = set()
    for s in dfa_1.final_states:
        final_states.add(s + "_1")
    for s in dfa_2.final_states:
        final_states.add(s + "_2")

    nfa = NFA(states=set(states), input_symbols=alphabet, transitions=transitions, initial_state=s0,
              final_states=final_states)
    dfa = DFA.from_nfa(nfa)
    return dfa


def intersection(dfa1, dfa2):
    states = []
    transitions = {}
    alphabet = dfa1.input_symbols
    for q1 in dfa1.states:
        for q2 in dfa2.states:
            s = q1 + "_" + q2
            states.append(s)
            transitions[s] = {}
            for a in alphabet:
                transitions[s][a] = dfa1.transitions[q1][a] + "_" + dfa2.transitions[q2][a]
    initial_state = dfa1.initial_state + "_" + dfa2.initial_state
    final_states = set()
    for q1 in dfa1.final_states:
        for q2 in dfa2.final_states:
            final_states.add(q1 + "_" + q2)
    dfa = DFA(states=set(states), input_symbols=alphabet, transitions=transitions, initial_state=initial_state,
              final_states=final_states)
    return dfa


def super_sequence(dfa):
    # Copy over the various states form the DFA to create the NFA
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

    nfa = NFA(states=set(states), input_symbols=input_symbols, transitions=nfa_transitions, initial_state=initial_state,
              final_states=final_states)

    return DFA.from_nfa(nfa)


def closure_plus(dfa):
    states = dfa.states.copy()
    input_symbols = dfa.input_symbols.copy()
    transitions = dfa.transitions
    initial_state = dfa.initial_state
    final_states = dfa.final_states.copy()
    nfa_transitions = {}
    for q in states:
        nfa_transitions[q] = {}
        for a in input_symbols:
            nfa_transitions[q][a] = {transitions[q][a]}
        if q in final_states:
            nfa_transitions[q][''] = {initial_state}

    nfa = NFA(states=set(states), input_symbols=input_symbols, transitions=nfa_transitions, initial_state=initial_state,
              final_states=final_states)

    super_seq_dfa = DFA.from_nfa(nfa)

    return super_seq_dfa


def concatenate(dfa1, dfa2):
    states = dfa1.states.copy()
    input_symbols = dfa1.input_symbols.copy()
    transitions = dfa1.transitions
    initial_state = dfa1.initial_state

    transitions2 = dfa2.transitions.copy()

    initial_state2 = dfa1.initial_state

    final_states = dfa1.final_states.copy()
    nfa_transitions = {}
    for q in states:
        nfa_transitions[q] = {}
        for a in input_symbols:
            nfa_transitions[q][a] = {transitions[q][a]}
        if q in final_states:
            nfa_transitions[q][''] = {initial_state2}

    new_states = []
    new_states_dict = {}

    for q in dfa2.states:
        s = q
        while s in states:
            s = s + "_2"
        new_states.append((s, q))
        new_states_dict[q] = s
        states.add(s)

    for t in new_states:
        s = t[0]
        q = t[1]
        nfa_transitions[s] = {}
        for a in input_symbols:
            nfa_transitions[s][a] = {new_states_dict[transitions2[q][a]]}

    for q in states:
        if q in final_states:
            nfa_transitions[q][''] = {new_states_dict[dfa2.initial_state]}

    final_states2 = set()
    for q in dfa2.final_states:
        final_states2.add(new_states_dict[q])

    nfa = NFA(states=set(states), input_symbols=input_symbols, transitions=nfa_transitions, initial_state=initial_state,
              final_states=final_states2)

    return DFA.from_nfa(nfa)


def new_state_names(dfa):
    input_symbols = dfa.input_symbols.copy()
    transitions = dfa.transitions

    new_states = set()

    pairs = []
    dict_previous_new = {}
    i = 0
    for q in dfa.states:
        s = str(i)
        pairs.append((s, q))
        dict_previous_new[q] = s
        new_states.add(s)
        i = i + 1

    new_transitions = {}
    for q in dfa.states:
        new_transitions[dict_previous_new[q]] = {}
        for a in input_symbols:
            new_transitions[dict_previous_new[q]][a] = dict_previous_new[transitions[q][a]]

    final_states = set()

    for q in dfa.final_states:
        if q in dict_previous_new.keys():
            final_states.add(dict_previous_new[q])
        else:
            print("Key " + q + " was not found in the dictionary 'dictOldNew'")

    initial_state = dict_previous_new[dfa.initial_state]

    return DFA(states=set(new_states), input_symbols=input_symbols, transitions=new_transitions,
               initial_state=initial_state, final_states=final_states)


def print_dfa(automaton: Union[DFA, NFA]):
    print(f"{type(automaton).__name__}:".center(40, "-"))
    print("Initial State: " + automaton.initial_state)
    print("States and Transitions:")
    for t in automaton.transitions.items():
        print(t)
    print("Final States:")
    for f in automaton.final_states:
        print(f)
    print("-" * 40)


def get_non_self_loop_letters(dfa: DFA, state):
    result = []
    for a in dfa.input_symbols:
        if dfa.transitions[state][a] != state:
            result.append(a)
    return result
