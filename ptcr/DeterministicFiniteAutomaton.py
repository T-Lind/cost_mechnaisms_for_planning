import json
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

    @classmethod
    def load_model(cls, model_text_raw: str):
        model_text = json.loads(model_text_raw)

        start_state = model_text['start_state']
        accept_states = model_text['accept_states']
        states = model_text['states']
        alphabet = model_text['alphabet']

        transitions = {}
        for key, value in model_text['transitions'].items():
            for symbol, next_state in value.items():
                transitions[(key, symbol)] = next_state

        return cls(states, alphabet, transitions, start_state, accept_states)
