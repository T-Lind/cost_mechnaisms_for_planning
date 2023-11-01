from Narration.EventPredictor import EventPredictor

"""
This class used to implement the case study of 'Party' 
"""


class BaseCaseStudy:

    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep = None

    def make_event_predictor(self):

        dfa = self.get_dfa()

        self.ep = EventPredictor(dfa, self.mc, self.alphabetS, self.verbose)

        self.ep.mdp.checkTransitionFunction()

    def get_dfa(self):

        return None

    def compute_optimal_policy(self):
        self.make_event_predictor()
        self.computed_policy = self.ep.optimalPolicyInfiniteHorizon(0.01, True)
        return self.computed_policy

    def make_pomdpx_file(self, filePath):
        if self.ep == None:
            self.make_event_predictor()
        self.ep.mdp.write_POMDPX_XML(filePath)
        print("POMDPX file created in path '" + filePath + "'")

    def simulate(self, show_results=True):
        if self.computed_policy is None:
            self.compute_optimal_policy()
        tple = self.computed_policy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate(policy, True)
        avg = tple2[0]
        if show_results:
            print("expc=" + str(expc) + ", avg=" + str(avg) + ", recorded=" + tple2[1])
        return expc, avg, tple2[1], tple[3]

    def simulate_greedy_algorithm(self, showResults=True):
        if self.ep == None:
            self.make_event_predictor()
        tple2 = self.ep.simulate_greedy_algorithm(False)
        avg = tple2[0]
        if showResults:
            # print("expc="+str(expc)+", avg="+str(avg)+", recorded="+tple2[1])
            print("avg=" + str(avg) + ", recorded=" + tple2[1])
        return (avg, tple2[1])

    def simulate_greedy_algorithm_pomdp(self, show_results=True):
        if self.ep is None:
            self.make_event_predictor()
        # print("simulate_greedyAlgorithm_pomdp")
        tple2 = self.ep.simulate_greedy_algorithm_pomdp(False)
        avg = tple2[0]
        if show_results:
            # print("expc="+str(expc)+", avg="+str(avg)+", recorded="+tple2[1])
            print("avg=" + str(avg) + ", recorded=" + tple2[1])
        return (avg, tple2[1])

    def simulate_general_and_greedy_algorithms(self, showResults=True):
        if self.computed_policy is None:
            self.compute_optimal_policy()
        tple = self.computed_policy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate_general_and_greedy_algorithms(policy, False)
        avg = tple2[0]
        avg2 = tple2[2]
        if showResults:
            print("expc=" + str(expc) + ", avgGeneralAlg=" + str(avg) + ", recordedGeneralAlg=" + tple2[
                1] + ", avgGreedyAlg=" + str(avg2) + ", recordedGreedyAlg=" + tple2[3])
        return (expc, avg, tple2[1], avg2, tple2[3], tple[3])
