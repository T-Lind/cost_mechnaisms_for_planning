from ptcr import DeterministicFiniteAutomaton

with open("../samples/oulu/dfa.json", "r") as f:
    text = f.read()
dfa = DeterministicFiniteAutomaton.load_model(text)

# Print the DFA
print(dfa)

# Test the DFA with input strings
input_strings = ['hkt', 'khh', 'chhk']  # Should be True, False, True

for input_str in input_strings:
    result = dfa.run(input_str)
    print(f"Input: {input_str}, Accepted: {result}")
