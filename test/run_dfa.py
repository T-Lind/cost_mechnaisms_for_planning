from ptcr import DeterministicFiniteAutomaton

# Example usage:
states = {'q0', 'q1', 'q2'}
alphabet = {'h', 'h', 't', 'c'}
transitions = {
    ('start', 'h'): 'q(h)',
    ('start', 'k'): 'q(k)',
    ('start', 't'): 'q(c-t)',
    ('start', 'c'): 'q(c-t)',
    ('q(h)', 'h'): 'q(h)',
    ('q(h)', 'k'): 'q(k,h)',
    ('q(h)', 't'): 'q(h,c-t)',
    ('q(h)', 'c'): 'q(h,c-t)',
    ('q(k)', 'k'): 'q(k)',
    ('q(k)', 'h'): 'q(k,h)',
    ('q(k)', 't'): 'q(k,c-t)',
    ('q(k)', 'c'): 'q(k,c-t)',
    ('q(c-t)', 't'): 'q(c-t)',
    ('q(c-t', 'c'): 'q(c-t)',
    ('q(c-t)', 'h'): 'q(h,c-t)',
    ('q(c-t)', 'k'): 'q(k,c-t)',
    ('q(k,h)', 'k'): 'q(k,h)',
    ('q(k,h)', 'h'): 'q(k,h)',
    ('q(k,h)', 't'): 'q(k,h,c-t)',
    ('q(k,h)', 'c'): 'q(k,h,c-t)',
    ('q(h,c-t)', 'h'): 'q(h,c-t)',
    ('q(h,c-t)', 't'): 'q(h,c-t)',
    ('q(h,c-t)', 'c'): 'q(h,c-t)',
    ('q(h,c-t)', 'k'): 'q(k,h,c-t)',
    ('q(k,c-t)', 'k'): 'q(k,c-t)',
    ('q(k,c-t)', 't'): 'q(k,c-t)',
    ('q(k,c-t)', 'c'): 'q(k,c-t)',
    ('q(k,c-t)', 'h'): 'q(k,h,c-t)',
    ('q(k,h,c-t)', 'k'): 'q(k,h,c-t)',
    ('q(k,h,c-t)', 'h'): 'q(k,h,c-t)',
    ('q(k,h,c-t)', 't'): 'q(k,h,c-t)',
    ('q(k,h,c-t)', 'c'): 'q(k,h,c-t)'
}
start_state = 'start'
accept_states = {'q(k,h,c-t)'}

dfa = DeterministicFiniteAutomaton(states, alphabet, transitions, start_state, accept_states)

# Print the DFA
print(dfa)

# Test the DFA with input strings
input_strings = ['hkt', 'khh', 'chhk']  # Should be True, False, True

for input_str in input_strings:
    result = dfa.run(input_str)
    print(f"Input: {input_str}, Accepted: {result}")
