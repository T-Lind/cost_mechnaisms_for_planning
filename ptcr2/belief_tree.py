class BeliefTreeNode:
    def __init__(self, belief_state):
        self.belief_state = belief_state
        self.probability_to = 0
        self.edges = []
        self.height = -1
        self.goal_avg_value = 0
        self.excepted_prob_to_goal = 0
        self.best_action_to_max_expected_to_goal = None

    def __str__(self):
        return str(self.goal_avg_value) + ", h=" + str(self.height)

    def add_edge(self, edge):
        self.edges.append(edge)

    def compute_expected_prob_to_goal(self, action):
        total = 0
        for e in self.edges:
            if e.action == action:
                total += e.probability * e.belief_tree_node_to.excepted_prob_to_goal
        return total

    def compute_best_action_to_max_expected_to_goal(self, H, actions):
        if self.height == H:
            self.excepted_prob_to_goal = self.goal_avg_value
            self.best_action_to_max_expected_to_goal = None
            return

        max_val = 0
        best_action = None
        for a in actions:
            val = self.compute_expected_prob_to_goal(a)
            if val > max_val:
                max_val = val
                best_action = a
        self.excepted_prob_to_goal = max_val
        self.best_action_to_max_expected_to_goal = best_action

    def get_child(self, observation, action):
        print(len(self.edges))
        # print("action="+action)
        # print("observation="+str(observation))
        for e in self.edges:
            if e.action == action:
                if e.observation == observation:
                    #           print("found child")
                    return e.belief_tree_node_to
        return None


class BeliefTreeEdge:
    def __init__(self, belief_tree_node_from, belief_tree_node_to, action, observation, probability):
        self.belief_tree_node_from = belief_tree_node_from
        self.belief_tree_node_to = belief_tree_node_to
        self.action = action
        self.observation = observation
        self.probability = probability


class BeliefTree:
    def __init__(self, root=None):
        self.root = root

    """
    Compute the optimal policy that maximize the  probability of reaching goal states
    """

    def compute_optimal_policy_to_max_prob_to_goal(self, H, actions):
        queue = []
        arr = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            arr.insert(0, node)
            for e in node.edges:
                queue.append(e.belief_tree_node_to)

        print("len(arr)=" + str(len(arr)))
        for i in range(len(arr)):
            node = arr[i]
            node.compute_best_action_to_max_expected_to_goal(H, actions)

    def __str__(self):
        result = ""
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            result += str(node) + "\n"
            for e in node.edges:
                queue.append(e.belief_tree_node_to)
        return result

    def number_of_nodes_with_nonzero_goal_value(self):
        result = 0
        queue = []
        queue.append(self.root)
        while len(queue) > 0:
            node = queue.pop(0)
            if node.goal_avg_value > 0:
                result += 1
            for e in node.edges:
                queue.append(e.belief_tree_node_to)
        return result
