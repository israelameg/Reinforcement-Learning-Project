"""
Name: Israela Megira
ID: 209015817
"""
import hashlib
import os
from pddlsim import planner
import my_valid_actions
import random
import sys
import json
from Graph import Graph

# global
save_prev_state = 0
visited = []
goals = []
goals_parts = []
count = 0
save_steps = []
save_action = []
save_goal = []
real_state = []
real_goal = []
goals_size = 0
goals_state = []
lalala = 0
i = 0


class best_agent(object):
    def __init__(self):
        super(best_agent, self).__init__()

    def initialize(self, services):
        global goals_size
        self.services = services
        is_deterministic = 1
        for action in self.services.parser.task.actions:
            if action.effects_probs is not None:
                is_deterministic = 0
        if is_deterministic == 1:
            # for planning
            self.steps = planner.make_plan(sys.argv[2], sys.argv[3])
        else:
            # for learning
            for goal in self.services.goal_tracking.uncompleted_goals[0].parts:
                goals.append(goal.args)
                goals_size = goals_size + 1
            for goal in self.services.goal_tracking.uncompleted_goals:
                for part in goal.parts:
                    goals_parts.append(part.args)

    def next_action(self):
        global goals_state

        # check if the domain is deterministic or not
        is_deterministic = 1
        for action in self.services.parser.task.actions:
            if action.effects_probs is not None:
                is_deterministic = 0

        if is_deterministic == 1:
            # Planning
            if len(self.steps) > 0:
                return self.steps.pop(0).lower()
            return None
        else:
            # Learn
            if sys.argv[1] == "-L":
                if self.services.goal_tracking.reached_all_goals():
                    if os.path.exists('PrevState.json'):
                        os.remove('PrevState.json')
                    if os.path.isfile('Qtable' + sys.argv[3] + '.json') and os.access('Qtable' + sys.argv[3] + '.json', os.R_OK):
                        with open('Qtable' + sys.argv[3] + '.json', 'r') as fp:
                            Q = json.load(fp)
                            Q["numRun"] = Q["numRun"] + 1
                    if os.path.isfile('QGraph' + sys.argv[3]  +'.json') and os.access('QGraph' + sys.argv[3]  +'.json', os.R_OK):
                        with open('QGraph' + sys.argv[3]  +'.json', 'r') as fp:

                            q_dict = json.load(fp)
                    if os.path.isfile('Qaction' + sys.argv[3] + '.json') and os.access('Qaction' + sys.argv[3] + '.json', os.R_OK):
                        with open('Qaction' + sys.argv[3] + '.json', 'r') as fp:
                            dict_action = json.load(fp)

                    # update backward
                    i = 0
                    next_max = 0
                    future_state = 0
                    while save_steps:
                        my_state = save_steps.pop()
                        my_action = save_action.pop()
                        my_goals = save_goal.pop()
                        my_real_state = real_state.pop()
                        if i == 0:
                            print my_state
                            print my_goals
                            print my_action
                            print Q[str(my_state)][my_goals][str(my_action)]

                            Q[str(my_state)][my_goals][str(my_action)] += self.reward(my_real_state, my_action,
                                                                                      save_steps[-1])
                            next_max = Q[str(my_state)][my_goals][my_action]
                            future_state = my_state
                        else:
                            print my_state
                            print my_goals
                            print my_action
                            if save_steps:
                                Q[str(my_state)][my_goals][str(my_action)] += self.reward(my_real_state, my_action,
                                                                                          save_steps[
                                                                                              -1]) + 0.9 * next_max
                            else:
                                Q[str(my_state)][my_goals][str(my_action)] += self.reward(my_real_state, my_action,
                                                                                          0) + 0.9 * next_max
                            next_max = Q[str(my_state)][my_goals][str(my_action)]
                            exist = 0
                            for search in q_dict:
                                if str(my_state) == search:
                                    q_dict[str(my_state)][str(future_state)] = Q[str(my_state)][my_goals][
                                        str(my_action)]
                                    dict_action[str(my_state)][str(future_state)] = my_action
                                    exist = 1

                                if exist == 0:
                                    q_dict[str(my_state)] = {}
                                    dict_action[str(my_state)] = {}
                                    q_dict[str(my_state)][str(future_state)] = Q[str(my_state)][my_goals][my_action]
                                    dict_action[str(my_state)][str(future_state)] = my_action

                        future_state = my_state
                        i = i + 1

                    # save data in json file
                    with open('Qtable' + sys.argv[3] + '.json', 'w+') as fp:
                        json.dump(Q, fp)
                    with open('QGraph' + sys.argv[3] + '.json', 'w+') as fp:
                        json.dump(q_dict, fp)
                    with open('Qaction' + sys.argv[3] + '.json', 'w+') as fp:
                        json.dump(dict_action, fp)
                    with open('goals_state' + sys.argv[3] + '.json', 'w+') as fp:
                        json.dump(goals_state, fp)
                    policy = {}
                    policy = self.make_policy(Q)
                    # save policy in json file
                    with open("POLICY" + sys.argv[3], 'w+') as fp:
                        json.dump(policy, fp)
                    return None
                else:
                    """ create a states list and a Q table"""
                    if os.path.isfile('Qtable' + sys.argv[3] +'.json') and os.access('Qtable' + sys.argv[3] +'.json', os.R_OK):
                        with open('Qtable' + sys.argv[3] +'.json', 'r') as fp:
                            Q = json.load(fp)
                            if os.path.isfile('QGraph' + sys.argv[3]  + '.json') and os.access('QGraph' + sys.argv[3]  + '.json', os.R_OK):
                                with open('QGraph' + sys.argv[3]  + '.json', 'r') as fp:
                                    q_dict = json.load(fp)
                            if os.path.isfile('Qaction' + sys.argv[3]  +'.json') and os.access('Qaction' + sys.argv[3]  +'.json', os.R_OK):
                                with open('Qaction' + sys.argv[3]  +'.json', 'r') as fp:
                                    dict_action = json.load(fp)
                            if os.path.isfile('goals_state' + sys.argv[3]  + '.json') and os.access('goals_state' + sys.argv[3]  + '.json', os.R_OK):
                                with open('goals_state' + sys.argv[3]  + '.json', 'r') as fp:
                                    goals_state = json.load(fp)
                    # states = self.create_states()
                    else:
                        Q = {}
                        q_dict = {}
                        dict_action = {}
                        Q["numRun"] = 1
                    return self.learn(Q, q_dict, dict_action)
        # Execute
        if sys.argv[1] == "-E":
            if self.services.goal_tracking.reached_all_goals():
                return None
            else:
                return self.execute()

    def learn(self, Q, q_dict, dict_action):
        global visited
        global real_state
        global real_goal
        global save_steps
        global save_action
        global save_goal
        global save_prev_state
        global goals
        valid_actions = my_valid_actions.ValidActions(self.services.parser, self.services.pddl,
                                                      self.services.perception)

        exploration_rate = 0.4
        learning_rate = 0.4
        discount_factor = 0.9

        # change the exploration_rate and learning_rate

        if Q["numRun"] < 4:
            exploration_rate = 1
            # learning_rate = 0.2
        if Q["numRun"] > 3:
            exploration_rate = 0.5
            # learning_rate = 0.2
        if Q["numRun"] > 6:
            exploration_rate = 0.4
            # learning_rate = 0.2
        if Q["numRun"] > 8:
            exploration_rate = 0.3
            # learning_rate = 0.2

        """ if os.path.isfile('PrevState.json') and os.access('PrevState.json', os.R_OK):
            with open('PrevState.json', 'r') as fp:
                save_prev_state = json.load(fp) """

        # make states dictionary to a hash string
        real_current_state = self.services.perception.get_state()
        dict = self.services.perception.get_state()
        for key in dict:
            val = dict[key]
            if isinstance(val, (set)):
                dict[key] = list(val)
                dict[key].sort()
        json_represntion = json.dumps(dict, sort_keys=True)
        current_state = hashlib.sha1(json_represntion).hexdigest()
        # current_state = hash(str(dict))

        # make goals list to a string
        goals_hash = ','.join(map(str, goals))

        # current_state = self.get_current_state(real_current_state)

        """ If we dont have enough knowledge - it is more possible that we choose
        a random action. But if we already have enough, we would prefer to choose
        an action based on our knowledge """

        if random.uniform(0, 1) < exploration_rate:
            """ chose a random action """
            # action = random.choice(self.services.valid_actions.get())
            action = random.choice(valid_actions.get(real_current_state))
            action_exist = 0
            state_exist = 0
            goals_exist = 0

            for state in Q:
                if current_state == state:
                    state_exist = 1
                    for search_goals in Q[current_state]:
                        if search_goals == goals_hash:
                            goals_exist = 1
            if state_exist == 1:
                if goals_exist == 1:
                    for option in Q[current_state][goals_hash]:
                        if action == option:
                            action_exist = 1
                    if action_exist == 0:
                        Q[current_state][goals_hash][action] = 0

                else:
                    Q[current_state][goals_hash] = {}
                    Q[current_state][goals_hash][action] = 0
            else:
                # json_represntion = json.dumps(real_current_state, sort_keys=True)
                # current_state = hashlib.sha1(json_represntion).hexdigest()

                Q[current_state] = {}
                Q[current_state][goals_hash] = {}
                Q[current_state][goals_hash][action] = 0
        else:
            """ take the best action from the Q table """
            maxi = 0
            i = 0
            state_exist = 0
            for state in Q:
                if current_state == state:
                    state_exist = 1
            if state_exist == 0:
                # json_represntion = json.dumps(real_current_state, sort_keys=True)
                # current_state = hashlib.sha1(json_represntion).hexdigest()
                Q[current_state] = {}
                Q[current_state][goals_hash] = {}
            # Q[frozenset(real_current_state)] = {}
            # If there is not an action from this state yet
            if Q[current_state][goals_hash] == {}:
                action = random.choice(valid_actions.get(real_current_state))
                Q[current_state][goals_hash][action] = 0
            else:
                for option in Q[current_state][goals_hash]:
                    if i == 0:
                        action = option
                        maxi = Q[current_state][goals_hash][option]
                    if Q[current_state][goals_hash][option] > maxi:
                        maxi = Q[current_state][goals_hash][option]
                        action = option
                        # action = option
                    i = i + 1

        """ save the current state before we move to the next state """
        real_prev_state = real_current_state
        prev_state = current_state
        # real_prev_state = current_state
        # prev_state = self.get_current_state(real_prev_state)

        """ next state - apply the chosen action and save the new state"""
        self.services.parser.apply_action_to_state(action, real_current_state, check_preconditions=False)
        # next_state = self.get_current_state(real_current_state)
        dict = real_current_state
        for key in dict:
            val = dict[key]
            if isinstance(val, (set)):
                dict[key] = list(val)
                dict[key].sort()
        json_represntion = json.dumps(dict, sort_keys=True)
        next_state = hashlib.sha1(json_represntion).hexdigest()

        # next_state = hash(str(dict))

        """ save the old val from the Q table - not necessary"""
        old_val = Q[current_state][goals_hash][action]

        """ gets reward"""
        reward = self.reward(real_current_state, action, save_prev_state)
        goals_hash_new = ','.join(map(str, goals))

        """ gets the best value of the next state """
        next_max = 0
        next_action = 0
        j = 0
        state_exist = 0
        goals_exist = 0
        for state in Q:
            if next_state == state:
                state_exist = 1
                for search_goals in Q[next_state]:
                    if search_goals == goals_hash_new:
                        goals_exist = 1
        if state_exist == 0 or goals_exist == 0:
            next_max = 0
        else:
            for bla in Q[next_state][goals_hash_new]:
                if j == 0:
                    next_max = Q[next_state][goals_hash_new][bla]
                    next_action = bla
                if Q[next_state][goals_hash_new][bla] > next_max:
                    next_max = Q[next_state][goals_hash_new][bla]
                    next_action = bla
                j = j + 1

        # gets action probability
        prob_next_action = 0
        if next_action != 0:
            for option in self.services.parser.task.actions:
                if option.name in next_action:
                    if option.effects_probs is not None:
                        prob_next_action = option.effects_probs[0]
                    else:
                        prob_next_action = 1
        # new_val = (1 - learning_rate) * Q[current_state][action] + learning_rate * (
        # self.reward(real_prev_state, real_current_state, action, save_prev_state) + discount_factor * next_max)

        new_val = (1 - learning_rate) * Q[current_state][goals_hash][action] + learning_rate * (
                reward + discount_factor * next_max)

        """ update the Q table """
        Q[current_state][goals_hash][action] = new_val
        q_dict = self.add_edge(q_dict, current_state, next_state, Q[current_state][goals_hash][action])
        dict_action = self.add_edge1(dict_action, current_state, next_state, action)


        if real_prev_state != real_current_state:
            save_prev_state = real_prev_state
        # save Q table in json file
        """ with open('PrevState.json', 'w+') as fp:
            json.dump(save_prev_state, fp) """

        visited.append(real_current_state)
        save_steps.append(current_state)
        save_action.append(action)
        save_goal.append(goals_hash)
        real_state.append(real_current_state)
        real_goal.append(list(goals))

        # save Q table in json file
        with open('Qtable' + sys.argv[3] + '.json', 'w+') as fp:
            json.dump(Q, fp)
        with open('QGraph' + sys.argv[3] + '.json', 'w+') as fp:
            json.dump(q_dict, fp)
        with open('Qaction' + sys.argv[3] + '.json', 'w+') as fp:
            json.dump(dict_action, fp)

        return action

    def reward(self, curr_state, action, prev_state):
        global goals_state
        """ Reward function """
        re = -1
        """ if prev_state == curr_state:
            re = -0.6 """
        for visit in visited:
            if curr_state == visit:
                re = -5
        if prev_state == curr_state:
            re = -10
        # if get a goal
        new_state_from_option = self.services.parser.copy_state(curr_state)

        """ for goal in self.services.goal_tracking.uncompleted_goals:
            for part in goal.parts:
                if part.args in goals_parts:
                    boolRes = part.test(self.services.perception.get_state())
                    if boolRes:
                        goals_parts.remove(part.args)
                        re = 1 """
        if self.services.goal_tracking.reached_all_goals():
            # backward upate
            if len(real_goal) > 0:
                before = len(real_goal.pop())
                if len(real_goal) > 0:
                    after = len(real_goal.pop())
                    if before > after:
                        re = 100
                else:
                    re = 100

        else:
            # regular update
            for goal in self.services.goal_tracking.uncompleted_goals[0].parts:
                if self.services.parser.test_condition(goal, new_state_from_option):
                    if goal.args in goals:
                        for key in new_state_from_option:
                            val = new_state_from_option[key]
                            if isinstance(val, (set)):
                                new_state_from_option[key] = list(val)
                                new_state_from_option[key].sort()

                        json_represntion = json.dumps(new_state_from_option, sort_keys=True)
                        my_goal_state = hashlib.sha1(json_represntion).hexdigest()
                        # my_goal_state = hash(str(curr_state))
                        goals_state.append(my_goal_state)
                        goals.remove(goal.args)
                        re = 100

        return re

    def create_states(self):
        """ Create list of states """
        dict = self.services.perception.get_state()
        states = []
        for key in dict:
            for list in dict[key]:
                if key == "empty":
                    states.append(list[0])
        return states

    def get_current_state(self, state):
        """ Return the current state (only the tile) """
        for key in state:
            for list in state[key]:
                # find where is the person
                if key == "at":
                    current_person_temp = list[0]
                    current_tile_temp = list[1]
        return current_tile_temp

    def make_policy(self, Q):
        """ Make policy - take the best action from the Q table for each state """
        policy = {}
        # q_graph = Graph()
        for state in Q:
            policy[state] = {}
            best_action = None
            best_value = 0
            second_action = None
            second_value = 0
            i = 0
            if state != "numRun":
                for search_goals in Q[state]:
                    policy[state][search_goals] = {}
                    for action in Q[state][search_goals]:
                        if i == 0:
                            best_action = action
                            best_value = Q[state][search_goals][action]
                            second_action = action
                            second_value = Q[state][search_goals][action]
                        else:
                            if Q[state][search_goals][action] > best_value:
                                best_action = action
                                best_value = Q[state][search_goals][action]
                        i = i + 1
                    policy[state][search_goals] = best_action
        return policy

    def execute(self):
        """ Execute by using the policy """

        global save_prev_state
        global count
        global i
        global lalala

        """ load policy file """
        if os.path.isfile("POLICY" + sys.argv[3]) and os.access("POLICY" + sys.argv[3], os.R_OK):
            with open("POLICY" + sys.argv[3], 'r') as fp:
                policy = json.load(fp)

        """policy_graph = Graph([])
        for state in policy:
            for goals_left in policy[state]:
                policy_graph.add_edge(policy_graph[],)"""

        """ gets the current state"""
        dict = self.services.perception.get_state()
        for key in dict:
            val = dict[key]
            if isinstance(val, (set)):
                dict[key] = list(val)
                dict[key].sort()
        json_represntion = json.dumps(dict, sort_keys=True)
        pleaseWork = hashlib.sha1(json_represntion).hexdigest()
        # pleaseWork = hash(str(dict))

        # load q dict
        if os.path.isfile('Qaction' + sys.argv[3] + '.json') and os.access('Qaction' + sys.argv[3] + '.json', os.R_OK):
            with open('Qaction' + sys.argv[3] + '.json', 'r') as fp:
                dict_action = json.load(fp)
        if os.path.isfile('QGraph' + sys.argv[3] + '.json') and os.access('QGraph' + sys.argv[3] + '.json', os.R_OK):
            with open('QGraph' + sys.argv[3] + '.json', 'r') as fp:
                q_dict = json.load(fp)
        if os.path.isfile('goals_state' + sys.argv[3] + '.json') and os.access('goals_state' + sys.argv[3] + '.json', os.R_OK):
            with open('goals_state' + sys.argv[3] + '.json', 'r') as fp:
                goals_state = json.load(fp)
                # find the goal
                # for goal in self.services.goal_tracking.uncompleted_goals[0].parts:
                # # dijkstra from Q table
                graphi = Graph([])
                for start in q_dict:
                    for end in q_dict[start]:
                        graphi.add_edge(start, end, q_dict[start][end])
                for goal in goals_state:
                    # print goal
                    for state in policy:
                        if state == goal:
                            path = graphi.dijkstra(pleaseWork, goal)
                            length = len(path)
                            # print length
                            if length != 0:
                                if path[0] in dict_action:
                                    if path[1] in dict_action[path[0]]:
                                        my_action = dict_action[path[0]][path[1]]
                                        return my_action

        goals_string = ','.join(map(str, goals))
        state_exist = 0
        action_exist = 0
        for search_state in policy:
            if pleaseWork == search_state:
                state_exist = 1
                for search_goals in policy[search_state]:
                    if goals_string == search_goals:
                        action_exist = 1
        """ check if state and goal state exist """
        if state_exist == 1 and action_exist == 1:
            returnAction = policy[pleaseWork][goals_string]
            new_state_from_option = self.services.parser.copy_state(dict)
            self.services.parser.apply_action_to_state(returnAction, new_state_from_option, check_preconditions=False)

            # if reach goal - so remove the goal from the goals list
            for goal in self.services.goal_tracking.uncompleted_goals[0].parts:
                if self.services.parser.test_condition(goal, new_state_from_option):
                    if goal.args in goals:
                        goals.remove(goal.args)

            for key in new_state_from_option:
                val = new_state_from_option[key]
                if isinstance(val, (set)):
                    new_state_from_option[key] = list(val)
                    new_state_from_option[key].sort()
            json_represntion = json.dumps(new_state_from_option, sort_keys=True)
            new_state_from_option_hash = hashlib.sha1(json_represntion).hexdigest()
            # new_state_from_option_hash = hash(str(new_state_from_option))

            if save_prev_state != 0:
                for key in save_prev_state:
                    val = save_prev_state[key]
                    if isinstance(val, (set)):
                        save_prev_state[key] = list(val)
                        save_prev_state[key].sort()
                json_represntion = json.dumps(save_prev_state, sort_keys=True)
                save_prev_state_hash = hashlib.sha1(json_represntion).hexdigest()
                # save_prev_state_hash = hash(str(save_prev_state))

            if save_prev_state != 0:
                if new_state_from_option_hash == save_prev_state_hash or pleaseWork == new_state_from_option_hash:
                    count = count + 1
                else:
                    save_prev_state = dict
                    count = 0
                # infinity loop
                if count > 5:
                    save_prev_state = dict
                    return random.choice(self.services.valid_actions.get())
                else:
                    # do the policy action
                    save_prev_state = dict
                    return returnAction
            else:
                # do the policy action
                save_prev_state = dict
                return returnAction
        else:
            # if state does not exist in the policy - so do random action
            return random.choice(self.services.valid_actions.get())

    def add_edge(self, my_dict, state, next_state, cost):
        exist = 0
        for search_state in my_dict:
            if state == search_state:
                my_dict[state][next_state] = cost
                exist = 1
        if exist == 0:
            my_dict[state] = {}
            my_dict[state][next_state] = cost

        return my_dict

    def add_edge1(self, my_dict, state, next_state, action):
        exist = 0
        for search_state in my_dict:
            if state == search_state:
                my_dict[state][next_state] = action
                exist = 1
        if exist == 0:
            my_dict[state] = {}
            my_dict[state][next_state] = action

        return my_dict


