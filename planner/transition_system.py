from itertools import chain, combinations


class TransitionSystem:

    def __init__(self, states, transitions, initial_state, atomic_props, labeling_func, events):
        self.alphabet = None
        self.states = states
        self.transitions = transitions
        self.initial_state = initial_state
        self.atomic_props = atomic_props
        self.labeling_func = labeling_func

        self.events = events

        self.stateTransitions = {}
        for s in self.states:
            self.stateTransitions[s] = set()
        for t in transitions:
            self.stateTransitions[t[0]].add(t)

        self.alphabets_sets = {}
        self.ex_atomic_props = atomic_props.copy()
        self.ex_atomic_props.extend(events)
        # print("self.ex_atomic_props: "+str(self.ex_atomic_props))
        self.create_alphabet()

    def has_transition(self, from_state, to_state):
        for t in self.stateTransitions[from_state]:
            if t[1] == to_state:
                return True
        return False

    def create_alphabet(self):
        xs = list(self.ex_atomic_props)
        sets = chain.from_iterable(combinations(xs, n) for n in range(len(xs) + 1))
        self.alphabet = []
        for s in sets:
            self.alphabet.append(s)
            self.alphabets_sets[s] = set(s)

    def get_letter(self, letter):
        for a in self.alphabet:
            if a == letter:
                return a
        for a in self.alphabet:
            # print("a: "+str(a))
            # print("letter: "+str(letter))
            # print("self.alphabets_sets[a]: "+str(self.alphabets_sets[a]))
            if self.alphabets_sets[a] == letter:
                # print("found set "+str(letter))
                return a

    def __str__(self):
        strP = "---------------------------------Transition System-------------------------------------"
        strP += "States = ["
        for s in self.states:
            strP += str(s)
        strP += "]"
        strP += "TransitionRelation = ["
        for t in self.transitions:
            strP += str(t)
        strP += "]"
        strP += "LabelingFunction = ["
        strP += str(self.labeling_func)
        strP += "]"
        strP += "----------------------------------------------------------------------------------"
        return strP

    def print_all(self):
        print("---------------------------------Transition System-------------------------------------")
        print("States = [")
        for s in self.states:
            print(s)
        print("]")
        print("TransitionRelation = [")
        for t in self.transitions:
            print(str(t))
        print("]")
        print("LabelingFunction = [")
        print(self.labeling_func)
        print("]")
        print("----------------------------------------------------------------------------------")
