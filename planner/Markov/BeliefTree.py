from array import array


class BeliefTreeNode:
    def __init__(self, beliefState):
        self.beliefState = beliefState
        self.probabilityTo = 0
        self.edges = []
        self.height = -1
        self.goalAvgValue = 0
        self.expcetedProbToGoal = 0
        self.bestActionToMaxExpectedToGoal = None
        
    
    def __str__(self):
        return str(self.goalAvgValue)+", h="+str(self.height)
    
    def addEdge(self, edge):
        self.edges.append(edge)
        
    def computeExpectedProbToGoal(self, action):
        total = 0
        for e in self.edges:
            if e.action == action:
                total += e.probability*e.beleifTreeNodeTo.expcetedProbToGoal
        return total
    
    
    def computeBestActionToMaxExpectedToGoal(self, H, actions):
        if self.height == H:
            self.expcetedProbToGoal = self.goalAvgValue
            self.bestActionToMaxExpectedToGoal = None
            return
            
        maxVal = 0
        bestAction = None
        for a in actions:
            val = self.computeExpectedProbToGoal(a)
            if val > maxVal:
                maxVal = val
                bestAction = a
        self.expcetedProbToGoal = maxVal
        self.bestActionToMaxExpectedToGoal = bestAction
        
    def getChild(self, observation, action):
        print(len(self.edges))
        #print("action="+action)
        #print("observation="+str(observation))
        for e in self.edges:
            if e.action == action:
                if e.observation == observation:
         #           print("found child")
                    return e.beleifTreeNodeTo
        return None
        
        
class BeleifTreeEdge:
    def __init__(self, beleifTreeNodeFrom, beleifTreeNodeTo, action, observation, probability):
        self.beleifTreeNodeFrom = beleifTreeNodeFrom
        self.beleifTreeNodeTo = beleifTreeNodeTo
        self.action = action
        self.observation = observation
        self.probability = probability   


class BeliefTree:
    def __init__(self, root=None):
        self.root = root
    
    def setRoot(self, root):
        self.root = root
        
    def getRoot(self):
        return self.root
    
    """
    Compute the optimal policy that maximize the  probability of reaching goal states
    """
    def computeOptimalPolicyToMaxProbToGoal(self, H, actions):
        queue = []
        arr = []
        queue.append(self.root)
        while len(queue)>0:
            node = queue.pop(0)
            arr.insert(0, node)
            for e in node.edges:
                queue.append(e.beleifTreeNodeTo)
                
        print("len(arr)="+str(len(arr)))
        for i in range(len(arr)):
            node = arr[i]
            node.computeBestActionToMaxExpectedToGoal(H, actions)
            
    def __str__(self):
        result = ""
        queue = []
        queue.append(self.root)
        while len(queue)>0:
            node = queue.pop(0)
            result += str(node)+"\n"
            for e in node.edges:
                queue.append(e.beleifTreeNodeTo)
        return result
            
    def numberOfNodesWithNonzeroGoalValue(self):
        result = 0
        queue = []
        queue.append(self.root)
        while len(queue)>0:
            node = queue.pop(0)
            if node.goalAvgValue > 0:
                result += 1
            for e in node.edges:
                queue.append(e.beleifTreeNodeTo)
        return result
    
        