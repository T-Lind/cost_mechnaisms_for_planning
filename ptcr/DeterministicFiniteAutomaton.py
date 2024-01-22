from typing import Set, Tuple, Dict


class DeterministicFiniteAutomaton:
    def __init__(self,
                 states: Set[str],
                 alphabet: Set[str],
                 transitions: Dict[Tuple[str, str], str],
                 start_state: str,
                 accept_states: Set[str]):
        self.states = states
        self.alphabet = alphabet
        self.transitions = transitions
        self.start_state = start_state
        self.accept_states = accept_states

    def __str__(self) -> str:
        result = "States: {}\n".format(self.states)
        result += "Alphabet: {}\n".format(self.alphabet)
        result += "Transitions:\n{}\n".format(self.transitions)
        result += "Start State: {}\n".format(self.start_state)
        result += "Accept States: {}\n".format(self.accept_states)
        return result

    def run(self, input_string: str, verbose=False) -> bool:
        current_state = self.start_state

        for symbol in input_string:
            if (current_state, symbol) in self.transitions:
                current_state = self.transitions[(current_state, symbol)]
            else:
                # If there is no transition for the current symbol, reject the input
                return False

        if verbose:
            print("Final state:", current_state)

        # Check if the final state is an accept state
        return current_state in self.accept_states


# Example usage:
states = {'q0', 'q1', 'q2'}
alphabet = {'0', '1'}
transitions = {('q0', '0'): 'q1', ('q0', '1'): 'q2', ('q1', '0'): 'q0', ('q1', '1'): 'q2', ('q2', '0'): 'q2',
               ('q2', '1'): 'q2'}
start_state = 'q0'
accept_states = {'q1'}

dfa = DeterministicFiniteAutomaton(states, alphabet, transitions, start_state, accept_states)

# Print the DFA
print(dfa)

# Test the DFA with input strings
input_strings = ['01', '001', '101', '1001']

for input_str in input_strings:
    result = dfa.run(input_str)
    print(f"Input: {input_str}, Accepted: {result}")
