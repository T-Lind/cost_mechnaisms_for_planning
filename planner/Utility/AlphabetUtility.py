
def getLetters(alphabet, cnf_formula):
    result = alphabet
    conditions = cnf_formula.split('&')
    for c in conditions:
        c = c.replace('(', '').replace(')', '')
        props = c.split('|') 
        temp = []
        for prop in props:
            prop = prop.strip()
            mustContaint = True
            if prop.startswith("~"):
                mustContaint = False
                prop = prop[1:]
            for a in result:
                contains = False
                for i in range(len(a)):
                    if a[i] == prop:
                        contains = True
                        break 
                if contains == mustContaint:
                    if a not in temp:
                        temp.append(a)
        result = temp
    return result

def addMissingTransitions(transitions, alphabet, states, q_trap):
    for q in states:
        for a in alphabet:
            if a not in transitions[q].keys():
                transitions[q][a]= q_trap
    return transitions

def isComplete(transitions, alphabet, states):
    for q in states:
        for a in alphabet:
            if a not in transitions[q]:
                return False
    return True
        
def test_getLetters():
    alphabet = [tuple(), tuple('a'), tuple('b'), tuple(('a', 'b'))]
    cnf_formula  = 'a|b'
    letters =  getLetters(alphabet, cnf_formula) 
    print(letters)
    
#test_getLetters()
        