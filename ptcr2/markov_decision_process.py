import itertools
import math
import random
from copy import deepcopy
from sys import float_info

from ptcr2.belief_tree import BeliefTree, BeliefTreeNode, BeliefTreeEdge



class MDPState:
    # name = ""
    # anchor = None
    # is_initial = False
    # is_goal = False
    # index = -1

    def __init__(self, name="", anchor=None):
        self.name = name
        self.anchor = anchor
        self.is_initial = False
        self.is_goal = False
        self.index = -1
        self.transitions = []
        self.reachable = True
        self.evidence_distribution = []
        self.sum_probability_transitions = 0

        self.available_actions = []
        self.actions_transitions = {}

        ''' 
         The following three fields are used for decomposing the graph into strongly connected components
        '''
        self.scc_index = -1
        self.low_link = -1
        self.scc = None

        """
        This field is used for DFS purposes
        """
        self.visited = False

        """
        This keeps the actions that the robot should avoid to do. 
        Doing any of them makes the robot to reach a dead end, a state from which no goal state is reachable.
        """
        self.avoid_actions = []

        self.a_goal_is_reachable = False

        """
        Set of transitions whose destination state is this
        """
        self.transitions_to = []

        """
        A Boolean vector assigned to the state as the weight of the state
        """
        self.weight_b_vector = []

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def add_transition(self, trans):
        if trans in self.transitions:
            return

        self.sum_probability_transitions += trans.probability

        self.transitions.append(trans)

    def add_transition_to(self, trans):
        if trans in self.transitions_to:
            return
        self.transitions_to.append(trans)

    def compute_available_actions(self):
        for tran in self.transitions:
            if tran.action not in self.available_actions:
                self.actions_transitions[tran.action] = []
                self.available_actions.append(tran.action)
            self.actions_transitions[tran.action].append(tran)

    def get_transition_by_dist_and_action(self, dst_state, action):
        for tran in self.transitions:
            if tran.action != action and tran.dst_state == dst_state:
                return tran
        return None


class MDPTransition:

    def __init__(self, src_state, dst_state, action, nature_action, probability):
        self.src_state = src_state
        self.dst_state = dst_state
        self.action = action
        self.nature_action = nature_action
        self.probability = probability
        self.eventPositive = False
        self.event_negative = False

    def __str__(self):
        return str(self.src_state) + "--" + str(self.action) + "--" + str(self.probability) + "-->" + str(
            self.dst_state)

    def __repr__(self):
        return str(self.src_state) + "--" + str(self.action) + "--" + str(self.probability) + "-->" + str(
            self.dst_state)


class MDPStrgConnComponent:
    def __init__(self):
        self.states = []
        self.name = ""
        self.scc_transitions = []

        self.leader = None

        """
        Used for DFS purposes
        """
        self.visited = False

    def add_state(self, state):
        self.states.append(state)

    def __str__(self):
        return self.name

    def __repr__(self):
        return self.name

    def get_full_name(self):
        result = "[name:" + self.name + ", states: {"
        i = 0
        for state in self.states:
            result += ("," if i > 0 else ",") + str(state)
            i += 1
        result += "} ]"
        return result

    def has_scc_transition_to(self, dst_scc):
        for scc in self.scc_transitions:
            if scc.dst_scc == dst_scc:
                return True
        return False

    def is_singular(self):
        return len(self.states) <= 1

    def is_a_singular_goal(self):
        return not (len(self.states) > 1 or not self.states[0].is_goal)


class SCCTransition:
    def __init__(self, src_scc, dst_scc):
        self.src_scc = src_scc
        self.dst_scc = dst_scc

    def __str__(self):
        result = "[" + self.src_scc.name + ", " + self.dst_scc.name + "]"
        return result

    def __repr__(self):
        return "[" + self.src_scc.name + ", " + self.dst_scc + "]"


class MDP:
    def __init__(self):
        self.states = []
        self.actions = []
        self.initial_state = MDPState()
        self.goal_states = []
        self.transitions = []
        self.transitions_dict = {}
        self.made_transition_dict = False
        self.has_evidence = False
        self.evidence_list = []
        self.observations = []

        """
        To access an entry of the observation function use observation_function[o][s][a], where
        'o' is an action, 's' is a state, and 'a' is an action.
        """
        self.observation_function = {}
        self.verbose = True
        self.beliefTree = None

        """
        These fields are used for decomposing the MDP into strongly connected components
        """
        self.strg_conn_compoments = []
        self.initial_scc = None
        self.scc_transitions = []
        self.scc_topological_order = []  # The strongly connected components are saved in a reverse topological ordering

        """
        This field is used for saving a topological ordering of the states 
        """
        self.topological_order = []

        self.states_dict_by_name = {}

        """
        The property evidence_list contains only the original evidence and not the results of whether the prediction was right or not.
        We make new list of evidences here.  
        """
        self.evidence_tuple_list = []

        """
        The observation that a goal state is reached
        """
        self.goal_reached_observation = "goal_reached"

        self.available_actions_computed = False

        self.base_num4_cost_linearization = 10

    def get_observation_string(self, o):
        if isinstance(o, tuple):
            return str(o[0]) + "_" + str(o[1])
        else:
            return str(o)

    """
    prediction_result: True, False (whether the predicted event happened). evidence: raised from the event model
    """

    def create_observation_function_dict(self):
        self.observation_function = {}
        for o in self.observations:
            self.observation_function[o] = {}
            for x in self.states:
                self.observation_function[o][x] = {}
                for e in self.actions:
                    self.observation_function[o][x][e] = 0

    def sum_transition_probs(self, dstState, action, beleifState):
        total = 0
        for srcState in self.states:
            total += beleifState[srcState.index] * self.conditional_probability(dstState.index, srcState.index, action)
        return total

    def probability_observation_given_action_and_belief(self, observation, action, belief_state):
        total = 0
        for x in self.states:
            total += self.observation_function[observation][x][action] * self.sum_transition_probs(x, action,
                                                                                                   belief_state)
        return total

    def create_belief(self, belief, action, observation):
        b = [0] * len(self.states)
        prob_obs_action = 0
        sum_trans_probs = {}

        for x in self.states:
            stp = self.sum_transition_probs(x, action, belief)
            sum_trans_probs[x] = stp
            prob_obs_action += self.observation_function[observation][x][action] * stp

        for x in self.states:
            # total = self.sumTransitionProbs(n, action, belief)
            total = sum_trans_probs[x]
            if total == 0 or self.observation_function[observation][x][action] == 0:
                b[x.index] = 0
            else:
                b[x.index] = (self.observation_function[observation][x][action] * total) / prob_obs_action
                """
            if self.observation_function[observation][n][action] ==0 and self.probabilityObservationGivenActionAndbeleif(observation, action, belief) == 0:
                print("O["+str(observation)+"]["+str(n)+"]["+str(action)+"]")
                b[n.index]
            else:
                if self.probabilityObservationGivenActionAndbeleif(observation, action, belief) == 0:
                    print("b:"+ str(belief))
                    print("O["+str(observation)+"]["+str(n)+"]["+str(action)+"]="+str(self.observation_function[observation][n][action]))
                   
                b[n.index]=(self.observation_function[observation][n][action]*total)/self.probabilityObservationGivenActionAndbeleif(observation, action, belief)
                """
        # print("b="+str(b))
        return b

    def get_goal_avg_value(self, belief):
        total = 0
        for x in self.goal_states:
            total += belief[x.index]
        return total

    def add_state(self, mdp_state):
        mdp_state.index = len(self.states)
        self.states.append(mdp_state)
        self.states_dict_by_name[mdp_state.name] = mdp_state

    def set_as_goal(self, mdp_state):
        mdp_state.is_goal = True
        if mdp_state not in self.goal_states:
            self.goal_states.append(mdp_state)

    def add_transition(self, mdp_transition):
        self.transitions.append(mdp_transition)
        mdp_transition.src_state.add_transition(mdp_transition)
        mdp_transition.dst_state.add_transition_to(mdp_transition)

    def compute_avoidable_actions(self):
        for state in self.states:
            state.a_goal_is_reachable = False

        queue = []
        for state in self.states:
            if state.is_goal:
                queue.append(state)
                state.a_goal_is_reachable = True

        while queue:
            state = queue.pop(0)
            for t in state.transitions_to:
                # if t.dst_state != state:
                #    continue
                all_reach_to_goals = True
                for t2 in t.src_state.actions_transitions[t.action]:
                    # if t2.action != t.action:
                    #    continue 
                    if not t2.dst_state.a_goal_is_reachable:
                        all_reach_to_goals = True
                if all_reach_to_goals and not t.src_state.a_goal_is_reachable:
                    t.src_state.a_goal_is_reachable = True
                    queue.append(t.src_state)
        for state in self.states:
            for action in self.actions:
                all_dst_reachable = True
                # if action not in state.actions_transitions.keys():
                #    continue
                if action not in state.actions_transitions.keys():
                    print("Action " + str(
                        action) + " has no key in dictionary state.actions_transitions for state " + state.name)
                else:
                    for trans in state.actions_transitions[action]:
                        if not trans.dst_state.a_goal_is_reachable:
                            all_dst_reachable = False
                            break
                if not all_dst_reachable:
                    state.avoid_actions.append(action)

        print("End computing avoidable actions ")

    def topological_order_helper(self, state):
        state.visited = True
        for tran in state.transitions:
            if not tran.dst_state.visited:
                self.topological_order_helper(tran.dst_state)
        self.topological_order.append(state)

    def topological_order_stack(self, state):
        state.visited = True
        for tran in state.transitions:
            if not tran.dst_state.visited:
                self.topological_order_helper(tran.dst_state)
        self.topological_order.append(state)

    def compute_topological_order(self):
        self.topological_order = []
        for state in self.states:
            state.visited = False
        for state in self.states:
            if not state.visited:
                self.topological_order_helper(state)

    def compute_topological_order_recursive(self):
        self.topological_order = []
        for state in self.states:
            state.visited = False
        stack = []
        stack.append(self.initial_state)
        state = self.initial_state
        state.visited = True

    def scc_topological_order_helper(self, scc):
        scc.visited = True
        for sccTrans in scc.scc_transitions:
            if not sccTrans.dst_scc.visited:
                self.scc_topological_order_helper(sccTrans.dst_scc)
        self.scc_topological_order.append(scc)

    def compute_scc_topological_order(self):
        self.scc_topological_order = []
        for scc in self.strg_conn_compoments:
            scc.visited = False
        for scc in self.strg_conn_compoments:
            if scc.visited == False:
                self.scc_topological_order_helper(scc)

    def make_scc_transitions(self):
        for scc in self.strg_conn_compoments:
            for state in scc.states:
                for trans in state.transitions:
                    if trans.dst_state.scc == scc:
                        continue
                    if state.scc.has_scc_transition_to(trans.dst_state.scc) == False:
                        sccTransition = SCCTransition(scc, trans.dst_state.scc)
                        self.scc_transitions.append(sccTransition)
                        state.scc.scc_transitions.append(sccTransition)

    def decompose_to_strng_conn_components(self):
        self.sccIndex = 0
        self.stack4scc = []
        for state in self.states:
            if state.scc_index == -1:
                self.strongconnect(state)

        for i in range(len(self.strg_conn_compoments)):
            self.strg_conn_compoments[i].name = "C" + str(len(self.strg_conn_compoments) - i - 1)
        self.make_scc_transitions()

    def strongconnect(self, state):
        state.scc_index = self.sccIndex
        state.low_link = self.sccIndex
        self.sccIndex = self.sccIndex + 1
        self.stack4scc.append(state)
        state.onStack = True
        for trans in state.transitions:
            state2 = trans.dst_state
            if state2.scc_index == -1:
                self.strongconnect(state2)
                state.low_link = min(state.low_link, state2.low_link)
            elif state2.onStack:
                state.low_link = min(state.low_link, state2.low_link)

        if state.scc_index == state.low_link:
            scc = MDPStrgConnComponent()
            scc.leader = state
            # scc.name = "C"+str(len(self.strg_conn_compoments))
            self.strg_conn_compoments.append(scc)
            if state == self.initial_state:
                self.initial_scc = scc
            hasPopedState = False
            while hasPopedState == False:
                state2 = self.stack4scc.pop()
                state2.onStack = False
                scc.add_state(state2)
                state2.scc = scc
                if state2 == state:
                    hasPopedState = True

    def print_strg_conn_components(self):
        print("------------------------- Start Printing Strongly Connected Components ----------------------")
        print("The MDP has " + str(len(self.strg_conn_compoments)) + " strongly connected components")
        for scc in self.strg_conn_compoments:
            print(scc.get_full_name())
        print("------------------------- End Printing Strongly Connected Components ----------------------")

    def print_graph_of_strg_conn_components(self):
        print(
            "------------------------- Start Printing The Graph of Strongly Connected Components ----------------------")
        print("The MDP has " + str(len(self.strg_conn_compoments)) + " strongly connected components")
        print("Initial SCC:" + str(self.initial_scc))
        for sccTrans in self.scc_transitions:
            print(str(sccTrans))
        print(
            "------------------------- End Printing The Graph of Strongly Connected Components ----------------------")

    def compute_states_available_actions(self):
        for state in self.states:
            state.compute_available_actions()
        self.available_actions_computed = True

    def reindex_states(self):
        i = 0
        for state in self.states:
            state.index = i
            i += 1

    def remove_un_reachable_states(self):
        self.recognize_reachable_states()
        if self.verbose:
            print("Unreachable state have been recognized")
        states_to_remove = []
        for state in self.states:
            if not state.reachable:
                states_to_remove.append(state)
        transitions_to_remove = []
        if self.verbose:
            print("Start checking transitions to remove")
            print("Number of transitions: " + str(len(self.transitions)))
        for trans in self.transitions:
            if not trans.src_state.reachable or not trans.dst_state.reachable:
                transitions_to_remove.append(trans)
        if self.verbose:
            print("End checking transitions to remove")

        if self.verbose:
            print("Start removing unreachable transitions")
        cnt_removed_transitions = 0
        for trans in transitions_to_remove:
            self.transitions.remove(trans)
            cnt_removed_transitions += 1
        if self.verbose:
            print("End removing unreachable transitions")

        if self.verbose:
            print("Start removing unreachable states")
        ctn_removed_states = 0
        for state in states_to_remove:
            self.states.remove(state)
            if state in self.goal_states:
                self.goal_states.remove(state)
            ctn_removed_states += 1
        if self.verbose:
            print("End removing unreachable states")

        if self.verbose:
            print("Start reindexing states")
        self.reindex_states()
        if self.verbose:
            print("End reindexing states")

        if self.verbose:
            print("Number of unreachable states removed = " + str(ctn_removed_states))
            print("Number of unreachable transitions removed = " + str(cnt_removed_transitions))

    def recognize_reachable_states(self):
        self.visited = [False] * len(self.states)
        for state in self.states:
            state.reachable = False
        self.dfs(self.initial_state)

    def dfs(self, state):
        # print("DFS "+str(state))
        self.visited[state.index] = True
        state.reachable = True
        for t in state.transitions:
            if self.visited[t.dst_state.index]:
                continue
            self.dfs(t.dst_state)

    def get_state_by_anchor(self, anchor):
        for s in self.states:
            if s.anchor == anchor:
                return s

        for s in self.states:
            eq = True
            for i in range(len(anchor)):
                if isinstance(anchor[i], set):
                    if len(anchor[i].difference(s.anchor[i])) != 0 or (len(s.anchor[i].difference(anchor[i])) != 0):
                        eq = False
                        break
                else:
                    if str(anchor[i]) != str(s.anchor[i]):
                        eq = False
                        break
            if eq:
                return s

        return None

    def get_next_state_for_event_positive(self, state, event):
        for t in self.transitions:
            if t.src_state == state and t.action == event and t.eventPositive:
                return t.dst_state
        return None

    def get_next_state_for_event_negative(self, state, event):
        for t in self.transitions:
            if t.src_state == state and t.action == event and t.event_negative:
                return t.dst_state
        return None

    def make_transitions_dict(self):
        n = len(self.states)
        self.transitions_dict = {}
        for a in self.actions:
            self.transitions_dict[a] = {}
            for i in range(n):
                self.transitions_dict[a][i] = {}
                for j in range(n):
                    self.transitions_dict[a][i][j] = 0
        for t in self.transitions:
            self.transitions_dict[t.action][t.src_state.index][t.dst_state.index] = t.probability
        self.made_transition_dict = True
        print("TransitionsDict has been made")

    def finite_horizon_optimal_policy(self, F, verbose):
        if verbose:
            print("------computing optimal policy for finite horizon--------------")
        n = len(self.states)
        G = [[0.0 for _ in range(n)] for _ in range(F + 1)]
        A = [["" for _ in range(n)] for _ in range(F + 1)]

        for j in range(n):
            if self.states[j].is_goal:
                G[F][j] = 0.0
            else:
                G[F][j] = 10000.0

        for i in range(F - 1, -1, -1):
            # print(i)
            for j in range(n):
                if self.states[j].is_goal:
                    A[i][j] = "STOP"
                    G[i][j] = 0.0
                    continue

                min_val = float_info.max
                opt_action = ""
                state = self.states[j]

                for action in self.actions:
                    val = 0.0
                    if not state.is_goal:
                        val += 1
                    for k in range(n):
                        term = G[i + 1][k] * self.conditional_probability(k, j, action)
                        val += term
                    if val < min_val:
                        min_val = val
                        opt_action = action
                G[i][j] = min_val
                A[i][j] = opt_action

        opt_policy = {}

        for j in range(n):
            opt_policy[self.states[j]] = A[0][j]
            if verbose:
                print("\pi(" + self.states[j].name + ")=" + A[0][j])

        if verbose:
            print("optimal policy for finite horizon has been computed")

        return opt_policy

    def conditional_probability(self, next_state_index, current_state_index, action):
        if not self.made_transition_dict:
            self.make_transitions_dict()
        return self.transitions_dict[action][current_state_index][next_state_index]

    def print_all(self):
        print("-------------------------------------------MDP--------------------------------------------")
        print("States:")
        for s in self.states:
            print(s)
        print("Transitions:")

        for t in self.transitions:
            print(t)

        print("Goal States")
        for g in self.goal_states:
            print(g)

        print("-------------------------------------------------------------------------------------------")

    def print_to_file(self, file_name):
        str_p = "-------------------------------------------MDP--------------------------------------------" + "\n"
        str_p += "Number of states=" + str(len(self.states)) + "\n"
        str_p += "States:" + "\n"
        for s in self.states:
            str_p += str(s) + "\n"
        str_p += "Transitions:" + "\n"

        for t in self.transitions:
            str_p += str(t) + "\n"

        str_p += "Goal States" + "\n"
        for g in self.goal_states:
            str_p += str(g) + "\n"

        str_p += "-------------------------------------------------------------------------------------------" + "\n"

        with open(file_name, "w") as f:
            f.write(str_p)

    def make_goal_states_absorbing(self):
        cnt = 0

        for t in self.transitions:
            if not t.src_state.is_goal:
                continue

            if t.src_state != t.dst_state:
                t.dst_state = t.src_state
                cnt = cnt + 1

        for s in self.states:
            if not s.is_goal:
                continue

            for t in s.transitions:
                if t.src_state != t.dst_state:
                    t.dst_state = t.src_state

        self.make_transitions_dict()

        print("Number of transitions become self-loops:" + str(cnt))

        cnt = 0
        for t in self.transitions:
            if t.src_state.is_goal:
                if t.src_state != t.dst_state:
                    cnt = cnt + 1

        print("Number of remained transitions from goal states:" + str(cnt))

    def check_transition_function(self):
        ok = True
        not_fixed = ""
        for s in self.states:
            for a in self.actions:
                if self.available_actions_computed:
                    if a not in s.available_actions:
                        continue
                sum = 0
                num_of_trans = 0
                recent_t = None
                trans = []
                for t in s.transitions:
                    if t.action == a:
                        sum += t.probability
                        num_of_trans += 1
                        trans.append(t)
                        if t.probability != 0:
                            recent_t = t
                if sum != 1:
                    ok = False
                    # print("Probability transition function does not summed_step_numbers to one: state="+s.name+", action="+ a+", prob="+str(summed_step_numbers)+", numOfTrans="+str(numOfTrans)+". "+str(trans))
                    # raise Exception("state="+s.name+", action="+ a+", prob="+str(summed_step_numbers)+", numOfTrans="+str(numOfTrans)+". "+str(trans))
                    if sum == 0.9999999999999999:
                        # print("Fix the error of mathematical operations")
                        recent_t.probability = 1 - (sum - recent_t.probability)
                    elif abs(sum - 1) < 0.000000001:
                        # print("Fix the error of mathematical operations")
                        recent_t.probability = 1 - (sum - recent_t.probability)
                    else:
                        print("Could not fix the error")
                        not_fixed += "Could not fix error for: state=" + s.name + ", is_goal=" + str(
                            s.is_goal) + ", action=" + str(a) + ", prob=" + str(sum) + ", numOfTrans=" + str(
                            num_of_trans) + ". " + str(trans) + "\n"

        if not_fixed != "":
            print(not_fixed)
            raise Exception(not_fixed)
        return ok

    """
    It checks whether
    \Sigma_{o \in observations} observation_function[o][s][a] = 1
    """

    def checkobservation_function(self):
        ok = True
        for a in self.actions:
            for s in self.states:
                sum_prob = 0
                for o in self.observations:
                    sum_prob += self.observation_function[o][s][a]
                if sum_prob != 1:
                    ok = False
                    print("The summed_step_numbers of observations for s=" + str(
                        s) + " and a=" + a + " does not summed_step_numbers to 1; it sums to " + str(sum_prob))

        return ok

    def make_belief_tree(self, H):

        if self.verbose:
            print("Start making the belief Tree with height " + str(H))
            print("The size of the belief tree will be in the order of (" + str(len(self.actions)) + "*" + str(
                len(self.observations)) + ")^" + str(H) + ", " + str(
                pow(len(self.actions) * len(self.observations), H)))

        beleif = [0] * len(self.states)
        for i in range(len(self.states)):
            beleif[i] = 0

        j = self.states.index(self.initial_state)
        beleif[j] = 1

        node = BeliefTreeNode(beleif)
        node.height = 0
        if self.initial_state in self.goal_states:
            node.goal_avg_value = 1

        bt = BeliefTree(node)

        queue = []

        queue.append(node)

        i = 0

        while len(queue) > 0:
            node = queue.pop(0)
            b = node.belief_state
            if node.height == H:
                continue
            for a in self.actions:
                for o in self.observations:
                    b2 = self.create_belief(b, a, o)
                    node2 = BeliefTreeNode(b2)
                    node2.goal_avg_value = self.get_goal_avg_value(b2)
                    node2.height = node.height + 1
                    prob = self.probability_observation_given_action_and_belief(o, a, b)
                    edge = BeliefTreeEdge(node, node2, a, o, prob)
                    node.add_edge(edge)
                    node2.probability_to = node.probability_to * prob
                    queue.append(node2)
            if self.verbose and i % 1000 == 0:
                print("Number of nodes added to the belief tree is " + str(i))
            i = i + 1

        self.beliefTree = bt

        if self.verbose:
            print("The belief tree was made ")
            # print("mdp.beliefTree="+str(self.beliefTree))

        return bt

    def get_observation(self, exMarkovChainState, eventOccured):
        for o in self.observations:
            if exMarkovChainState == o[1]:
                if eventOccured == o[0]:
                    return o
        return None

    def get_pomdpx_xml(self):

        #         transFunOk = self.checkTransitionFunction()
        #         print("Checking transition function")
        #         if transFunOk == False:
        #             print("Transition function is not a distribution")
        #         else:
        #             print("Transition function is fine")
        #
        #         obserFunOk = self.checkobservation_function()
        #         print("Checking observation function")
        #         if obserFunOk == False:
        #             print("Observation function is not a distribution")
        #         else:
        #             print("Observation function is fine")

        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            # if s.reachable == False:
            #        continue
            st += s.name + " "
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        st += "<ObsVar vname='prediction_result'>" + "\n"
        st += "<ValueEnum>correct wrong goal_reached</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        if self.has_evidence:
            st += "<ObsVar vname='evidence'>" + "\n"
            st += "<ValueEnum>"
            for ev in self.evidence_list:
                st = st + ev + " "
            st += "</ValueEnum>" + "\n"
            st += "</ObsVar>" + "\n"

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initial_state)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            # if t.dst_state.reachable == False:
            #    continue
            if t.src_state.is_goal == True:  ## Added for testing
                print("Not making a transition for the goal state " + t.src_state.name)
                continue
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.src_state.name + " " + t.dst_state.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        for a in self.actions:
            for s in self.states:
                if s.is_goal == False:
                    continue
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + s.name + " " + s.name + "</Instance>" + "\n"
                st += "<ProbTable>1</ProbTable>" + "\n"
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>prediction_result</Var>" + "\n"
        st += "<Parent>predicted_event state_1</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            for a in self.actions:
                # if self.states[i].reachable == False:
                #    continue
                if self.states[i].is_goal:

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                    st += "<ProbTable>1</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                    st += "<ProbTable>0</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                    st += "<ProbTable>0</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"
                else:
                    if a in self.states[i].anchor[1].events:
                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                        st += "<ProbTable>1</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"
                    else:
                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " correct" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " wrong" + "</Instance>" + "\n"
                        st += "<ProbTable>1</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

                        st += "<Entry>" + "\n"
                        st += "<Instance>" + a + " " + self.states[i].name + " goal_reached" + "</Instance>" + "\n"
                        st += "<ProbTable>0</ProbTable>" + "\n"
                        st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        # st += "</ObsFunction>" + "\n"

        if self.has_evidence:
            # st += "<ObsFunction>" + "\n"
            st += "<CondProb>" + "\n"
            st += "<Var>evidence</Var>" + "\n"
            st += "<Parent>state_1</Parent>" + "\n"
            st += "<Parameter type = 'TBL'>" + "\n"
            for s in self.states:
                # if s.reachable == False:
                #    continue
                for j in range(len(self.evidence_list)):
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + s.name + " " + self.evidence_list[j] + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(s.evidence_distribution[j]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

            st += "</Parameter>" + "\n"
            st += "</CondProb>" + "\n"
            # st += "</ObsFunction>" + "\n"

        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            # if self.states[i].reachable == False:
            #    continue
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].is_goal:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>-1</ValueTable>" + "\n"
                # if self.states[i].is_goal == False:
                #    st += "<ValueTable>-1</ValueTable>"+"\n"                    
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    """
    In this version, the observation function of the pomdpx comes directly from the field observation_function of this object
    """

    def get_pomdpx_xml2(self):

        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            # if s.reachable == False:
            #        continue
            st += s.name + " "
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        observationTitle = "predictionResult"
        if len(self.evidence_list) > 0:
            observationTitle += "Evidence"

        st += "<ObsVar vname='" + observationTitle + "'>" + "\n"
        st += "<ValueEnum>" + "\n"
        for o in self.observations:
            st += self.get_observation_string(o) + " "
        st += "</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initial_state)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            # if t.dst_state.reachable == False:
            #    continue
            if t.src_state.is_goal == True:  ## Added for testing
                continue
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.src_state.name + " " + t.dst_state.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        for a in self.actions:
            for s in self.states:
                if s.is_goal == False:
                    continue
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + s.name + " " + s.name + "</Instance>" + "\n"
                st += "<ProbTable>1</ProbTable>" + "\n"
                st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>" + observationTitle + "</Var>" + "\n"
        st += "<Parent>predicted_event state_1</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for s in self.states:
            for a in self.actions:
                # if self.states[i].reachable == False:
                #    continue
                for o in self.observations:
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + a + " " + s.name + " " + self.get_observation_string(
                        o) + " " + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(self.observation_function[o][s][a]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            # if self.states[i].reachable == False:
            #    continue
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].is_goal:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>-1</ValueTable>" + "\n"
                # if self.states[i].is_goal == False:
                #    st += "<ValueTable>-1</ValueTable>"+"\n"                    
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    def get_pomdpx_4_exmdp_xml(self):
        st = "<?xml version='1.0' encoding='ISO-8859-1'?>" + "\n"
        st += "<pomdpx version='0.1' id='autogenerated' xmlns:xsi='http://www.w3.org/2001/XMLSchema-instance' xsi:noNamespaceSchemaLocation='pomdpx.xsd'>" + "\n"
        st += "<Description>This is an auto-generated POMDPX file</Description>" + "\n"
        st += "<Discount>0.99999999999999</Discount>" + "\n"

        st += "<Variable>"

        st += "<StateVar vnamePrev='state_0' vnameCurr='state_1' fullyObs='false'>" + "\n"
        st += "<ValueEnum>"
        for s in self.states:
            st += s.name + " "
        # st += "trapping "    # adding a trapping state
        st += "</ValueEnum>" + "\n"
        st += "</StateVar>" + "\n"

        st += "<ObsVar vname='prediction_result_state'>" + "\n"
        st += "<ValueEnum>"
        for ob in self.observations:
            st += str(ob).replace(", ", "_").replace("(", "").replace(")", "") + " "
        st += "</ValueEnum>" + "\n"
        st += "</ObsVar>" + "\n"

        """
        if self.has_evidence:
            st += "<ObsVar vname='evidence'>"+"\n"
            st += "<ValueEnum>"
            for ev in self.evidence_list:
                st = st + ev + " " 
            st += "</ValueEnum>"+"\n"
            st += "</ObsVar>"+"\n"
        """

        st += "<ActionVar vname='predicted_event'>" + "\n"
        st += "<ValueEnum>"
        for a in self.actions:
            st += a + " "
        st += "</ValueEnum>" + "\n"
        st += "</ActionVar>" + "\n"

        st += "<RewardVar vname='cost'/>" + "\n"

        st += "</Variable>" + "\n"

        st += "<InitialStateBelief>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_0</Var>" + "\n"
        st += "<Parent>null</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        st += "<Entry>" + "\n"
        st += "<Instance>-</Instance>" + "\n"
        st += "<ProbTable>"
        indexOfInitialState = self.states.index(self.initial_state)
        for i in range(len(self.states)):
            if i == indexOfInitialState:
                st += "1 "
            else:
                st += "0 "
        # st += "0 "      # initial distribution entry for the trapping state
        st += "</ProbTable>" + "\n"
        st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</InitialStateBelief>" + "\n"

        st += "<StateTransitionFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>state_1</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for t in self.transitions:
            st += "<Entry>" + "\n"
            st += "<Instance>" + t.action + " " + t.src_state.name + " " + t.dst_state.name + "</Instance>" + "\n"
            st += "<ProbTable>" + str(t.probability) + "</ProbTable>" + "\n"
            st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"
        st += "</StateTransitionFunction>" + "\n"

        st += "<ObsFunction>" + "\n"
        st += "<CondProb>" + "\n"
        st += "<Var>prediction_result_state</Var>" + "\n"
        st += "<Parent>state_1 predicted_event</Parent>" + "\n"
        st += "<Parameter type = 'TBL'>" + "\n"
        for x in self.states:
            for e in self.actions:
                for o in self.observations:
                    st += "<Entry>" + "\n"
                    st += "<Instance>" + x.name + " " + e + " " + str(o).replace(", ", "_").replace("(", "").replace(
                        ")", "") + "</Instance>" + "\n"
                    st += "<ProbTable>" + str(self.observation_function[o][x][e]) + "</ProbTable>" + "\n"
                    st += "</Entry>" + "\n"

        st += "</Parameter>" + "\n"
        st += "</CondProb>" + "\n"

        st += "</ObsFunction>" + "\n"

        st += "<RewardFunction>" + "\n"
        st += "<Func>" + "\n"
        st += "<Var>cost</Var>" + "\n"
        st += "<Parent>predicted_event state_0</Parent>"
        st += "<Parameter type = 'TBL'>" + "\n"
        for i in range(len(self.states)):
            for a in self.actions:
                st += "<Entry>" + "\n"
                st += "<Instance>" + a + " " + self.states[i].name + "</Instance>" + "\n"
                if self.states[i].is_goal:
                    st += "<ValueTable>1</ValueTable>" + "\n"
                else:
                    st += "<ValueTable>0</ValueTable>" + "\n"
                st += "</Entry>" + "\n"
        st += "</Parameter>" + "\n"
        st += "</Func>" + "\n"
        st += "</RewardFunction>" + "\n"
        st += "</pomdpx>"
        return st

    def add_shadow_states_for_goal_states(self):
        for s in self.states:
            if not s.is_goal:
                continue
            s2 = MDPState(s.name + "_shadow", s)
            self.add_state(s2)
            for o in self.observations:
                self.observation_function[o][s2] = {}
                for a in self.actions:
                    self.observation_function[o][s2][a] = self.observation_function[o][s][a]

        for t in self.transitions:
            if not t.src_state.is_goal:
                continue
            if not t.dst_state.isGoal:
                continue
            src_state_shadow = self.get_state_by_anchor(t.src_state)
            dst_state_shadow = self.get_state_by_anchor(t.dst_state)
            t2 = MDPTransition(src_state_shadow, dst_state_shadow, t.action, t.nature_action, t.probability)
            self.add_transition(t2)

        for t in self.transitions:
            if not t.src_state.is_goal:
                continue
            dst_state_shadow = self.get_state_by_anchor(t.dst_state)
            t.dst_state = dst_state_shadow

    def get_preference_value_boolean_vector(self, booleanVector):
        m = len(booleanVector)
        val = 0.0
        for j in range(m):
            val += math.pow(2, m - j) if booleanVector[j] == True else 0.0
        return val

    def get_preference_value_vector(self, vect):
        m = len(vect)
        val = 0.0
        # print("vect="+str(vect))
        for j in range(m):
            val += math.pow(2, m - j) * vect[j]
        return val

    """
    Check if vect1 is dominated by vect2
    """

    def is_dominated(self, vect1, vect2):
        m = len(vect1)
        result = False
        # print("vect="+str(vect))
        for j in range(m):
            if vect2[j] > vect1[j]:
                result = True
                break
            elif vect2[j] < vect1[j]:
                result = False
                break
        return result

    def is_vects_equal(self, vect1, vect2):
        m = len(vect1)
        result = True
        for j in range(m):
            if vect2[j] != vect1[j]:
                result = False
                break
        return result

    def containts_val_tuple(self, valTupleList, valTuple):
        for v in valTupleList:
            if self.are_val_tuples_equal(v, valTuple):
                return True
        return False

    def are_val_tuples_equal(self, valueTuple1, valueTuple2):
        if valueTuple1[0] != valueTuple2[0]:
            return False
        for i in range(len(valueTuple1[1])):
            if valueTuple1[1][i] != valueTuple2[1][i]:
                return False
        return True

    def get_non_dominant_vectors(self, v_vectors):
        nonDominants = []
        for vect in v_vectors:
            dominated = False
            for vect2 in v_vectors:
                if vect2[0] < vect[0] and self.get_preference_value_vector(vect2[1]) > self.get_preference_value_vector(
                        vect[1]):
                    dominated = True
                    break

            if not dominated:
                nonDominants.append(vect)

        return nonDominants

    def get_non_dominant_actions_and_vectors(self, v_action_vectors):
        nonDominants = []
        # print("v_action_vectors: "+str(v_action_vectors))
        for a_vect in v_action_vectors:
            dominated = False
            for a_vect2 in v_action_vectors:
                if a_vect2[1][0] <= a_vect[1][0] and self.is_dominated(a_vect[1][1], a_vect2[1][1]):
                    dominated = True
                    break
                elif a_vect2[1][0] < a_vect[1][0] and (
                        self.is_dominated(a_vect[1][1], a_vect2[1][1]) or self.is_vects_equal(a_vect[1][1],
                                                                                              a_vect2[1][1])):
                    dominated = True
                    break
                elif a_vect2[1][0] == a_vect[1][0] and self.is_vects_equal(a_vect[1][1], a_vect2[1][1]) and a_vect2[0] < \
                        a_vect[0]:
                    dominated = True
                    break

            if not dominated:
                nonDominants.append(a_vect)

        return nonDominants

    def get_non_dominant_time_violation_cost_vectors(self, v_vectors):
        nonDominants = []
        # print("v_action_vectors: "+str(v_action_vectors))
        for a_vect in v_vectors:
            dominated = False
            for a_vect2 in v_vectors:
                if a_vect2[0] <= a_vect[0] and self.is_dominated(a_vect[1], a_vect2[1]):
                    dominated = True
                    break
                elif a_vect2[0] < a_vect[0] and (
                        self.is_dominated(a_vect[1], a_vect2[1]) or self.is_vects_equal(a_vect[1], a_vect2[1])):
                    dominated = True
                    break
                    # elif a_vect2[0] == a_vect[0] and self.isVectsEqual(a_vect[1], a_vect2[1]) and a_vect2[0] < a_vect[0]:
                #    dominated = True
                #    break

            if not dominated:
                nonDominants.append(a_vect)

        return nonDominants

    def round_value_vect(self, valueVect, digits):
        # print("valueVect: "+str(valueVect))
        # print("valueVect[0]: " +str(valueVect[0]))
        result = []
        lng = round(valueVect[0], digits)
        softs = []
        for num in valueVect[1]:
            softs.append(round(num, digits))
        tp = (lng, softs)
        # print("tp: "+str(tp))
        return tp

    def round_value_vects(self, valueVect, digits):
        result = []
        for t in valueVect:
            print("t: " + str(t))
            print("t[0]: " + str(t[0]))
            lng = round(t, digits)
            softs = []
            for num in t[1]:
                softs.append(round(num, digits))
            tp = (lng, softs)
            print("tp: " + str(tp))
            result.append(tp)
        return result

    def get_order4_computing_policy(self):
        queue = []
        for s in self.goal_states:
            for t in s.transitions_to:
                if t.src_state not in queue and (not t.src_state.is_goal):
                    queue.append(t.src_state)
        i = 0
        while i < len(queue):
            s = queue[i]
            for t in s.transitions_to:
                if t.src_state not in queue and (not t.src_state.is_goal):
                    queue.append(t.src_state)
            i += 1
        return queue

    def compute_numsteps_and_satis_prob_of_a_policy(self, states, policy, V, byStateNameOrIndex=False):
        """
        First compute the values of goal states
        """
        if byStateNameOrIndex:
            V = {}
        else:
            V = [() for i in range(len(self.states))]
        m = len(self.goal_states[0].weight_b_vector)
        for s in self.states:
            soft_const_prob_vect = [0] * m
            # vect = None
            if s.is_goal:
                for k in range(m):
                    soft_const_prob_vect[k] = 1.0 if s.weight_b_vector[k] else 0.0
                vect = (0, soft_const_prob_vect)
            else:
                vect = (0, soft_const_prob_vect)
            if byStateNameOrIndex:
                V[s.name] = vect
            else:
                V[s.index] = vect

        cnt = 1
        max_cnt = 500
        while cnt < max_cnt:
            for s in states:
                if s.is_goal:
                    continue
                # print(f"self.available_actions_computed: {self.available_actions_computed}")
                # if self.available_actions_computed and not s.a_goal_is_reachable:
                #    continue
                if byStateNameOrIndex:
                    a = policy[s.name]
                else:
                    a = policy[s.index]
                steps = 1
                soft_probs = [0] * m
                if a not in s.available_actions:
                    print(f"Error1: Action {a} is not in availableActions of state {s}")
                    print("Action: " + str(a))
                if a not in s.actions_transitions.keys():
                    print(f"Error2: Action {a} is not in availableActions of state {s}")
                    print("Action: " + str(a))
                    self.print_actions()
                    continue
                for tran in s.actions_transitions[a]:
                    s2 = tran.dst_state
                    if byStateNameOrIndex:
                        v_vect = V[s2.name]
                    else:
                        v_vect = V[s2.index]
                    steps += v_vect[0] * tran.probability
                    for j in range(m):
                        soft_probs[j] += v_vect[1][j] * tran.probability
                tple = (steps, soft_probs)
                if byStateNameOrIndex:
                    V[s.name] = tple
                else:
                    V[s.index] = tple
            cnt += 1
        print("Recalculated V: " + str(V))
        return V

    def print_actions(self):
        i = 1
        for a in self.actions:
            print(f"{i}. {a}")

    def are_policies_equal(self, policy1, policy2):
        for i in range(len(self.states)):
            if str(policy1[i]) != str(policy2[i]):
                return False
        return True

    def is_dominated_by_vector_num_steps_and_viol_cost(self, valuePolicy1, valuePolicy2):

        if valuePolicy2[0] <= valuePolicy1[0] and self.is_dominated(valuePolicy1[1], valuePolicy2[1]):
            return True
        elif valuePolicy2[0] < valuePolicy1[0] and (
                self.is_dominated(valuePolicy1[1], valuePolicy2[1]) or self.is_vects_equal(valuePolicy1[1],
                                                                                           valuePolicy2[1])):
            return True
        #                 elif a_vect2[1][0] == a_vect[1][0] and self.isVectsEqual(a_vect[1][1], a_vect2[1][1]) and a_vect2[0] < a_vect[0]:
        #                     dominated = True
        #                     break

        return False

    def is_policy_dominated(self, policy1, policy2):
        for i in range(len(self.states)):
            if self.is_dominated_by_vector_num_steps_and_viol_cost(policy1[i], policy2[i]):
                return True
        return False

    def select_a_random_policy(self, V_O, V_O_Actions):
        policy = [None] * len(self.states)
        for s in self.states:
            if s.is_goal:
                policy[s.index] = V_O_Actions[s.name]
            else:
                policy[s.index] = random.choice(V_O_Actions[s.name])
        policy_values = self.compute_numsteps_and_satis_prob_of_a_policy(self.states, list(policy), V_O)
        return policy, policy_values

    def recompute_pareto_policies_value_functions(self, V_O, V_O_Actions):
        paretoPolicy = [[] for _ in range(len(self.states))]
        for s in self.states:
            print("s.index: " + str(s.index))
            paretoPolicy[s.index] = list(set(V_O_Actions[s.name]))
        print("paretoPolicy: " + str(paretoPolicy))
        # print("num of policies within the pareto: "+str(len(itertools.product(*paretoPolicy))))
        num = 1
        policies = []
        policyValues = []
        print("")
        for policy in itertools.product(*paretoPolicy):
            print(f"{num}. policy: {list(policy)}")
            # print("policy: "+str(list(policy)))
            policies.append(list(policy))
            policy_value = self.compute_numsteps_and_satis_prob_of_a_policy(self.states, list(policy), V_O)
            policyValues.append(policy_value)
            num += 1

        num_eq_policies = 0
        num_of_dominated = 0
        for i in range(len(policies)):
            for j in range(len(policies)):
                if i == j:
                    continue
            if self.are_policies_equal(policies[i], policies[j]):
                num_eq_policies += 1
            if self.is_policy_dominated(policies[i], policies[j]):
                num_of_dominated += 1

        print("number of polices for which values are recomputed: " + str(num))
        print("Number of policies that are equal: " + str(num_eq_policies))
        print("Number of dominated policies is: " + str(num_of_dominated))

    def get_convex_hull(self, value_vectors, base_num4_cost_liniarization=10):
        value_vectors = self.comp_violation_costs_and_conct(value_vectors, base_num4_cost_liniarization)
        # hull = self.convexHull(value_vectors, len(value_vectors))
        hull = self.convex_hull(value_vectors)
        # print("hull="+str(hull))
        new_val_vectors = []
        for tple in hull:
            new_tple = (tple[0], tple[1])
            new_val_vectors.append(new_tple)
        return new_val_vectors

    def comp_violation_costs_and_conct(self, value_vectors, base_num4_cost_liniarization):
        for i in range(len(value_vectors)):
            violation_cost = 0.0
            m = len(self.goal_states[0].weight_b_vector)
            for j in range(m):
                violation_cost += (1 - value_vectors[i][1][j]) * math.pow(base_num4_cost_liniarization, m - j - 1)
            value_vectors[i] = value_vectors[i] + (violation_cost,)
        return value_vectors

    def sort_based_on_violation_cost(self, value_vectors):
        n = len(value_vectors)
        for i in range(n - 1):
            for j in range(0, n - i - 1):
                # if self.isDominated(value_vectors[j + 1], value_vectors[j]):
                # print("value_vectors[j]: " + str(value_vectors[j]))
                if self.is_dominated(value_vectors[j + 1][1], value_vectors[j][1]):
                    # if value_vectors[j] > value_vectors[j + 1] :
                    value_vectors[j], value_vectors[j + 1] = value_vectors[j + 1], value_vectors[j]

        for i in range(len(value_vectors)):
            value_vectors[i] = value_vectors[i] + (i,)

        # print("sorted items="+str(value_vectors))

        return value_vectors

    def convex_hull(self, points):
        """Computes the convex hull of a set of 2D points.

        Input: an iterable sequence of (n, y) pairs representing the points.
        Output: a list of vertices of the convex hull in counter-clockwise order,
          starting from the vertex with the lexicographically smallest coordinates.
        Implements Andrew's monotone chain algorithm. O(n log n) complexity.
        """

        # Sort the points lexicographically (tuples are compared lexicographically).
        # Remove duplicates to detect the case we have just one unique point.
        # points = sorted(set(points))

        points = sorted(points, key=lambda k: [k[0], k[2]])

        # Boring case: no points or a single point, possibly repeated multiple times.
        if len(points) <= 1:
            return points

        # 2D cross product of OA and OB vectors, i.e. z-component of their 3D cross product.
        # Returns a positive value, if OAB makes a counter-clockwise turn,
        # negative for clockwise turn, and zero if the points are collinear.
        def cross(o, a, b):
            return (a[0] - o[0]) * (b[2] - o[2]) - (a[2] - o[2]) * (b[0] - o[0])

        # Build lower hull 
        lower = []
        for p in points:
            while len(lower) >= 2 and cross(lower[-2], lower[-1], p) <= 0:
                lower.pop()
            lower.append(p)

        # Build upper hull
        upper = []
        for p in reversed(points):
            while len(upper) >= 2 and cross(upper[-2], upper[-1], p) <= 0:
                upper.pop()
            upper.append(p)

        # Concatenation of the lower and upper hulls gives the convex hull.
        # Last point of each list is omitted because it is repeated at the beginning of the other list. 
        return lower[:-1] + upper[:-1]

    def merge_two_convex_hulls_two_next_states(self, tran1, tran2, hull1, hull2, apply_limited_precision, precision):
        m = len(self.goal_states[0].weight_b_vector)
        Q_L = []
        for q1 in hull1:
            for q2 in hull2:
                steps = 1
                soft_probs = [0] * m

                steps += q1[0] * tran1.probability
                for k in range(m):
                    soft_probs[k] += q1[1][k] * tran1.probability

                steps += q2[0] * tran2.probability
                for k in range(m):
                    soft_probs[k] += q2[1][k] * tran2.probability

                tple = (steps, soft_probs)

                if apply_limited_precision:
                    tple = self.round_value_vect(tple, precision)
                    if not self.containts_val_tuple(Q_L, tple):
                        Q_L.append(tple)
                else:
                    Q_L.append(tple)

        Q_L = self.get_convex_hull(Q_L)
        return Q_L

    def merge_sum_convex_hulland_next_state_convex_hull(self, tran2, hull1, hull2, apply_limited_precision, precision):
        m = len(self.goal_states[0].weight_b_vector)
        Q_L = []
        # print(f"-------mergeSumConvexHullandNextStateConvexHull--------------")
        # print(f"tran2: {tran2}")
        # print(f"hull1: {hull1}")
        # print(f"hull2: {hull2}")
        for q1 in hull1:
            for q2 in hull2:
                steps = q1[0]
                soft_probs = deepcopy(q1[1])
                # print(f"q1: {q1}")
                # print(f"q2: {q2}")
                # print(f"soft_probs: {soft_probs}")

                steps += q2[0] * tran2.probability
                for k in range(m):
                    # print(f"before soft_probs[{k}]: {soft_probs[k]}")
                    soft_probs[k] += q2[1][k] * tran2.probability
                    # print(f"soft_probs[{k}]: {soft_probs[k]}")
                    # print(f"q2[1][{k}]: {q2[1][k]}")
                    # print(f"tran2: {tran2}")
                    if soft_probs[k] >= 1:
                        if soft_probs[k] < 1.001:
                            soft_probs[k] = 1
                            print(f" Fixed soft_probs[{k}] from {soft_probs[k]} to 1")
                        else:
                            raise Exception(f"Sum of soft_probs[{k}] is greate than 1 and it is {soft_probs[k]}")

                tple = (steps, soft_probs)

                if apply_limited_precision:
                    tple = self.round_value_vect(tple, precision)
                    if not self.containts_val_tuple(Q_L, tple):
                        Q_L.append(tple)
                else:
                    Q_L.append(tple)

        Q_L = self.get_convex_hull(Q_L)
        # print("Q_L: {Q_L}")
        return Q_L

    def qd_values_differ_less(self, Q_D1, Q_D2, epsilonExpcNumSteps, epsilonSoftConstSatis):
        m = len(self.goal_states[0].weight_b_vector)

        Q_1 = []
        for vec in Q_D1:
            violationCost = 0.0
            for j in range(m):
                violationCost += (1 - vec[1][j]) * math.pow(self.base_num4_cost_linearization, m - j - 1)
            vec2 = (vec[0], vec[1], violationCost)
            Q_1.append(vec2)
        Q_1 = sorted(Q_1, key=lambda k: [k[0], k[1]])

        Q_2 = []
        for vec in Q_D2:
            violationCost = 0.0
            for j in range(m):
                violationCost += (1 - vec[1][j]) * math.pow(self.base_num4_cost_linearization, m - j - 1)
            vec2 = (vec[0], vec[1], violationCost)
            Q_2.append(vec2)
        Q_2 = sorted(Q_2, key=lambda k: [k[0], k[2]])

        for i in range(len(Q_1)):
            vec1 = Q_1[i]
            vec2 = Q_2[i]
            if abs(vec1[0] - vec2[0]) > epsilonExpcNumSteps:
                return False
            for j in range(m):
                if abs(vec1[1][j] - vec2[1][j]) > epsilonSoftConstSatis:
                    return False
        return True

    def choose_best_polices_base_on_weights(self, Q_D, weights, num_of_digits):
        Q = {}
        m = len(self.goal_states[0].weight_b_vector)
        policies = []
        for weight_vect in weights:
            # if len(weight_vect) == 2 and weight_vect[0] == 0 and weight_vect[1] == 1:
            #    weight_vect[0] == 0.00000001
            #    weight_vect[1] == 0.99999999
            # print("success")
            # raise Exception("")
            for s in self.states:
                if s.is_goal:
                    continue
                Q[s.name] = {}
                for a in s.available_actions:
                    if self.available_actions_computed and (a in s.avoid_actions):
                        continue
                    min_cost = float_info.max
                    for i in range(len(Q_D[s.name][a])):
                        val_vect = Q_D[s.name][a][i]
                        violation_cost = 0
                        for j in range(m):
                            violation_cost += (1 - val_vect[1][j]) * math.pow(10, m - j - 1)
                        cost = weight_vect[0] * val_vect[0] + weight_vect[1] * violation_cost * 3
                        # violation_cost = violation_cost*10
                        if cost < min_cost:
                            min_cost = cost
                    Q[s.name][a] = min_cost

            P = {}
            C = {}
            for s in self.states:
                if s.is_goal:
                    P[s.name] = "STOP"
                    C[s.name] = 0
                    continue
                min_cost = float_info.max
                for a in s.available_actions:
                    if self.available_actions_computed and (a in s.avoid_actions):
                        continue
                    if Q[s.name][a] < min_cost:
                        min_cost = Q[s.name][a]
                        P[s.name] = a
                        C[s.name] = min_cost
                    elif Q[s.name][a] == min_cost and a[1] != self.special_noEvent and P[s.name][
                        1] == self.special_noEvent:
                        P[s.name] = a
                        C[s.name] = min_cost

            policies.append(P)
            print(f"C: {C}")
        return policies

    def get_observation_of_tuple(self, prediction_result, evidence):
        for o in self.observations:
            if o[0] != prediction_result:
                continue
            if o[1] != evidence:
                continue
            return o
        return None

    def get_boolean_observation(self, boolean_value: bool):
        for o in self.observations:
            if o == boolean_value:
                return o
        return None

    def make_observation_function(self):
        self.observations = []
        if len(self.evidence_list) > 0:
            for ev in self.evidence_list:
                o1 = (True, ev)
                o2 = (False, ev)
                self.observations.append(o1)
                self.observations.append(o2)
        else:
            self.observations.append(True)
            self.observations.append(False)
        self.observations.append(self.goal_reached_observation)

        self.create_observation_function_dict()

        if len(self.evidence_list) > 0:
            for x in self.states:
                s = x.anchor[1]  # The state of event model
                for a in self.actions:  # Event
                    for i in range(len(self.evidence_list)):
                        y = self.evidence_list[i]
                        o1 = self.get_observation_of_tuple(True, y)
                        o2 = self.get_observation_of_tuple(False, y)
                        o3 = self.goal_reached_observation
                        if x.is_goal:
                            self.observation_function[o1][x][a] = 0
                            self.observation_function[o2][x][a] = 0
                            self.observation_function[o3][x][a] = 1
                        else:
                            self.observation_function[o3][x][a] = 0
                            if a in s.events:
                                self.observation_function[o1][x][a] = s.evidenceDistribution[i]
                                self.observation_function[o2][x][a] = 0
                            else:
                                self.observation_function[o2][x][a] = s.evidenceDistribution[i]
                                self.observation_function[o1][x][a] = 0
        else:
            for x in self.states:
                s = x.anchor[1]
                for a in self.actions:
                    o1 = self.get_boolean_observation(True)
                    o2 = self.get_boolean_observation(False)
                    o3 = self.goal_reached_observation
                    if x.is_goal:
                        self.observation_function[o1][x][a] = 0
                        self.observation_function[o2][x][a] = 0
                        self.observation_function[o3][x][a] = 1
                    else:
                        self.observation_function[o3][x][a] = 0
                        if a in s.events:
                            self.observation_function[o1][x][a] = 1
                            self.observation_function[o2][x][a] = 0
                        else:
                            self.observation_function[o2][x][a] = 1
                            self.observation_function[o1][x][a] = 0
