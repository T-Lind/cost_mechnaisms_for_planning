import ptcr2.AutomataUtility as AutomataUtility
from ptcr2.BaseCaseStudy import BaseCaseStudy
from ptcr2.EventPredictor import EventPredictor
from ptcr2.MarkovChain import MarkovChain


class WeddingFOM(BaseCaseStudy):
    def __init__(self):
        super().__init__()
        self.verbose = True
        self.computed_policy = None
        self.ep = None

    def make_eventPredictor(self):
        stateNames1 = ["I", "E", "B", "C", "D", "S"]
        stateEvents1 = [set([]), set(["e1"]), set(["b1"]), set(["c1"]), set(["d1"]), set(["s1"])]
        transitionMatrix = [
            [0, 0.1, 0.3, 0.1, 0.2, 0.3],
            [0, 0.1, 0.2, 0.1, 0.3, 0.3],
            [0, 0.2, 0.1, 0.1, 0.3, 0.3],
            [0, 0.1, 0.2, 0.2, 0.3, 0.2],
            [0, 0.2, 0.3, 0.1, 0.1, 0.3],
            [0, 0.4, 0.2, 0.2, 0.2, 0.0]]
        initialDistribution = [0, 0.1, 0.3, 0.2, 0.2, 0.2]
        mc1 = MarkovChain(stateNames1, stateEvents1, transitionMatrix, initialDistribution, 0)

        stateNames2 = ["I", "E", "B", "C", "D", "S"]
        stateEvents2 = [set([]), set(["e2"]), set(["b2"]), set(["c2"]), set(["d2"]), set(["s2"])]
        mc2 = MarkovChain(stateNames2, stateEvents2, transitionMatrix, initialDistribution, 0)

        stateNames3 = ["I", "E", "B", "C", "D", "S"]
        stateEvents3 = [set([]), set(["e3"]), set(["b3"]), set(["c3"]), set(["d3"]), set(["s3"])]
        mc3 = MarkovChain(stateNames3, stateEvents3, transitionMatrix, initialDistribution, 0)

        mc12 = mc1.product_singleInitialState(mc2, [[("d1", "d2"), "d12"]])

        mc = mc12.product_singleInitialState(mc3, [[("d2, d3"), "d23"]])

        dfa = self.getDFA()

        self.ep = EventPredictor(dfa, mc, self.alphabetS, self.verbose)

        return self.ep

    def getDFA(self):
        alphabetS = {"e1", "b1", "c1", "d1", "s1", "e2", "b2", "c2", "d2", "s2", "d12", "e3", "b3", "c3", "d3", "s3",
                     "d23"}
        self.alphabetS = alphabetS

        dfa111 = AutomataUtility.dfaAcceptingASequence(["c3"], alphabetS)
        dfa112 = AutomataUtility.dfaAcceptingASequence(["s3"], alphabetS)
        dfa11 = AutomataUtility.union(dfa111, dfa112)

        dfa11 = AutomataUtility.newStateNames(dfa11)
        dfa11 = AutomataUtility.closurePlus(dfa11)
        dfa11 = dfa11.minify()
        dfa11 = AutomataUtility.newStateNames(dfa11)

        dfa12 = AutomataUtility.dfaAcceptingASequence(["d12"], alphabetS)
        dfa1 = AutomataUtility.concatenate(dfa11, dfa12)
        dfa1 = dfa1.minify()
        dfa1 = AutomataUtility.newStateNames(dfa1)

        dfa31 = AutomataUtility.dfaAcceptingASequence(["s3"], alphabetS)
        dfa32 = AutomataUtility.dfaAcceptingASequence(["c3"], alphabetS)
        dfa33 = AutomataUtility.union(dfa31, dfa32)

        dfa34 = AutomataUtility.concatenate(dfa33, dfa33)
        dfa35 = AutomataUtility.closurePlus(dfa33)
        dfa3 = AutomataUtility.concatenate(dfa34, dfa35)
        dfa3.minify()

        dfa211 = AutomataUtility.dfaAcceptingASequence(["d2"], alphabetS)
        dfa212 = AutomataUtility.dfaAcceptingASequence(["d12"], alphabetS)
        dfa213 = AutomataUtility.dfaAcceptingASequence(["d23"], alphabetS)
        dfa21 = AutomataUtility.union(dfa211, dfa212)
        dfa22 = AutomataUtility.union(dfa21, dfa213)
        dfa22 = AutomataUtility.newStateNames(dfa22)
        dfa23 = AutomataUtility.closurePlus(dfa22)
        dfa24 = AutomataUtility.dfaAcceptingASequence(["d12"], alphabetS)
        dfa25 = AutomataUtility.concatenate(dfa23, dfa24)
        dfa2 = dfa25.minify()
        dfa2 = AutomataUtility.newStateNames(dfa2)

        dfa1 = AutomataUtility.superSequence(dfa1)
        dfa3 = AutomataUtility.superSequence(dfa3)
        dfa2 = AutomataUtility.superSequence(dfa2)
        dfa2 = AutomataUtility.newStateNames(dfa2)

        dfa = AutomataUtility.intersection(dfa1, dfa3)
        dfa = dfa.minify()
        dfa = AutomataUtility.newStateNames(dfa)
        dfa = AutomataUtility.intersection(dfa, dfa2)
        dfa = AutomataUtility.newStateNames(dfa)
        dfa = dfa.minify()

        # May need to remove 
        dfa = AutomataUtility.newStateNames(dfa)

        AutomataUtility.printDFA(dfa)

        return dfa
