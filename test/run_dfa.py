from ptcr import DeterministicFiniteAutomaton

# Example usage:
states = {'q0', 'q1', 'q2'}
alphabet = {'0', '1'}
transitions = {('q0', '0'): 'q1',
               ('q0', '1'): 'q2',
               ('q1', '0'): 'q0',
               ('q1', '1'): 'q2',
               ('q2', '0'): 'q2',
               ('q2', '1'): 'q1'}
start_state = 'q0'
accept_states = {'q1'}

dfa = DeterministicFiniteAutomaton(states, alphabet, transitions, start_state, accept_states)

# Print the DFA
print(dfa)

# Test the DFA with input strings
input_strings = ['0', '001', '10', '1100']

for input_str in input_strings:
    result = dfa.run(input_str)
    print(f"Input: {input_str}, Accepted: {result}")