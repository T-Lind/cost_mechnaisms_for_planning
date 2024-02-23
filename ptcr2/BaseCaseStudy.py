from ptcr2.EventPredictor import EventPredictor

"""
This class used to implement the case study of 'Party' 
"""


class BaseCaseStudy:

    def __init__(self):
        self.verbose = True
        self.computed_policy = None
        self.ep = None

    def make_eventPredictor(self):

        dfa = self.getDFA()

        self.ep = EventPredictor(dfa, self.mc, self.alphabetS, self.verbose)

        self.ep.mdp.checkTransitionFunction()

    def getDFA(self):
        return None

    def compute_optimalPolicy(self):
        self.make_eventPredictor()
        self.computed_policy = self.ep.optimalPolicyInfiniteHorizon(0.01, True)
        return self.computed_policy

    def make_POMDPX_file(self, filePath):
        if self.ep == None:
            self.make_eventPredictor()
        self.ep.mdp.write_POMDPX_XML(filePath)
        print("POMDPX file created in path '" + filePath + "'")

    def simulate(self, showResults=True):
        if self.computed_policy == None:
            self.compute_optimalPolicy()
        tple = self.computed_policy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate(policy, True)
        avg = tple2[0]
        if showResults == True:
            print("expc=" + str(expc) + ", avg=" + str(avg) + ", recorded=" + tple2[1])
        return (expc, avg, tple2[1], tple[3])

    def simulate_greedyAlgorithm(self, showResults=True):
        if self.ep == None:
            self.make_eventPredictor()
        tple2 = self.ep.simulate_greedyAlgorithm(False)
        avg = tple2[0]
        if showResults == True:
            # print("expc="+str(expc)+", avg="+str(avg)+", recorded="+tple2[1])
            print("avg=" + str(avg) + ", recorded=" + tple2[1])
        return (avg, tple2[1])

    def simulate_greedyAlgorithm_pomdp(self, showResults=True):
        if self.ep == None:
            self.make_eventPredictor()
        # print("simulate_greedyAlgorithm_pomdp")
        tple2 = self.ep.simulate_greedyAlgorithm_pomdp(False)
        avg = tple2[0]
        if showResults == True:
            # print("expc="+str(expc)+", avg="+str(avg)+", recorded="+tple2[1])
            print("avg=" + str(avg) + ", recorded=" + tple2[1])
        return (avg, tple2[1])

    def simulate_generalAndGreedyAlgorithms(self, showResults=True):
        if self.computed_policy == None:
            self.compute_optimalPolicy()
        tple = self.computed_policy
        policy = tple[0]
        expc = tple[2]
        tple2 = self.ep.simulate_generalAndGreedyAlgorithms(policy, False)
        avg = tple2[0]
        avg2 = tple2[2]
        if showResults == True:
            print("expc=" + str(expc) + ", avgGeneralAlg=" + str(avg) + ", recordedGeneralAlg=" + tple2[
                1] + ", avgGreedyAlg=" + str(avg2) + ", recordedGreedyAlg=" + tple2[3])
        return (expc, avg, tple2[1], avg2, tple2[3], tple[3])
