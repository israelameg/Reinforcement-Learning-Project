"""
Name: Israela Megira
ID: 209015817
"""
import ast
import hashlib
import os

import numpy as np
from Graph import Graph
import pickle

import dijkstra
import string

from pddlsim import planner
from pddlsim.services import valid_actions

import my_valid_actions
import random
import sys
import json
from collections import defaultdict
from RMaxAgent import RMaxAgent

from Graph import Graph

reward = {}
count_r_s_a = 0


class Rmax(object):

    def __init__(self):
        super(Rmax, self).__init__()
        self.t_s_a_counts = defaultdict(lambda: defaultdict(int))  # S --> A --> #ts
        self.r_s_a_counts = defaultdict(lambda: defaultdict(int))  # S --> A --> #rs
        self.transitions = defaultdict(lambda: defaultdict(lambda: defaultdict(int)))  # S --> A --> S' --> counts
        self.rewards = defaultdict(lambda: defaultdict(list))  # S --> A --> reward
        custom_q_init = None
        self.name = "RMax"
        # Agent.__init__(self, name=name, actions=actions, gamma=gamma)
        self.rmax = 1
        self.s_a_threshold = 2
        self.custom_q_init = custom_q_init
        self.reset()
        self.custom_q_init = custom_q_init
        self.gamma = 0.95
        self.epsilon_one = 0.99
        self.episode_number = 0
        self.prev_state = None
        self.prev_action = None
        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.rmax))

    def initialize(self, services):
        self.services = services
        self.actions = list(self.services.valid_actions.get())

    def next_action(self):
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
                self.actions = self.services.valid_actions.get()

                if self.services.goal_tracking.reached_all_goals():
                    return None
                else:
                    current_state = self.services.perception.get_state()
                    reward = 0
                    #bla = RMaxAgent(self.services.valid_actions.get())
                    # load jsons

                    """if os.path.isfile('QRmax.json') and os.access('QRmax.json', os.R_OK):
                        with open('QRmax.json') as f:
                            data = f.read()
                        data = ast.literal_eval(data[data.index('{'):data.rindex('}') + 1])
                        self.q_func = defaultdict(list, data)"""
                    """if os.path.isfile('TRmax.json') and os.access('TRmax.json', os.R_OK):
                        with open('TRmax.json') as f:
                            data = f.read()
                        data = ast.literal_eval(data[data.index('{'):data.rindex('}') + 1])
                        self.transitions = defaultdict(list, data)
                    if os.path.isfile('RRmax.json') and os.access('RRmax.json', os.R_OK):
                        with open('RRmax.json') as f:
                            data = f.read()
                        data = ast.literal_eval(data[data.index('{'):data.rindex('}') + 1])
                        self.rewards = defaultdict(list, data)"""

                    return self.act(current_state, reward)
                # init
                self.init()

                # update
                self.update()
            else:
                return self.execute()

    def reset(self):
        '''
        Summary:
            Resets the agent back to its tabula rasa config.
        '''
        self.prev_state = None
        self.prev_action = None

        if self.custom_q_init:
            self.q_func = self.custom_q_init
        else:
            self.q_func = defaultdict(lambda: defaultdict(lambda: self.rmax))

    def get_num_known_sa(self):
        return sum([self.is_known(s, a) for s, a in self.r_s_a_counts.keys()])

    def is_known(self, s, a):
        return self.r_s_a_counts[s][a] >= self.s_a_threshold and self.t_s_a_counts[s][a] >= self.s_a_threshold

    def act(self, state, reward):
        # Update given s, a, r, s' : self.prev_state, self.prev_action, reward, state
        self.update(self.prev_state, self.prev_action, reward, state)

        # Compute best action by argmaxing over Q values of all possible s,a pairs
        dict = state
        for key in dict:
            val = dict[key]
            if isinstance(val, (set)):
                dict[key] = list(val)
                dict[key].sort()
        json_represntion = json.dumps(dict, sort_keys=True)
        hash_state = hashlib.sha1(json_represntion).hexdigest()
        #hash_state = hash(frozenset(state))
        action = self.get_max_q_action(hash_state)

        # Update pointers.
        self.prev_action = action
        self.prev_state = state

        # save jsons
        if self.q_func != {}:
            with open('QRmax.json', 'w+') as fp:
                json.dumps(self.q_func,fp)
        if self.transitions != {}:
            with open('TRmax.json', 'w+') as fp:
                json.dump(self.transitions, fp)
            with open('RRmax.json', 'w+') as fp:
                json.dump(self.rewards, fp)

        return action

    def update(self, state, action, reward, next_state):
        if state != None and action != None:
            dict = state
            for key in dict:
                val = dict[key]
                if isinstance(val, (set)):
                    dict[key] = list(val)
                    dict[key].sort()
            json_represntion = json.dumps(dict, sort_keys=True)
            hash_state = hashlib.sha1(json_represntion).hexdigest()
            #hash_state = hash(str(state))
            if self.r_s_a_counts[hash_state][action] <= self.s_a_threshold or self.t_s_a_counts[hash_state][
                action] <= self.s_a_threshold:
                dict = next_state
                for key in dict:
                    val = dict[key]
                    if isinstance(val, (set)):
                        dict[key] = list(val)
                        dict[key].sort()
                json_represntion = json.dumps(dict, sort_keys=True)
                hash_next_state = hashlib.sha1(json_represntion).hexdigest()
                # Add new data points if we haven't seen this s-a enough.
                self.rewards[hash_state][action] += [reward]
                self.r_s_a_counts[hash_state][action] += 1
                self.transitions[hash_state][action][hash_next_state] += 1
                self.t_s_a_counts[hash_state][action] += 1

                if self.r_s_a_counts[hash_state][action] == self.s_a_threshold:
                    # Start updating Q values for subsequent states
                    lim = int(np.log(1 / (self.epsilon_one * (1 - self.gamma))) / (1 - self.gamma))
                    for i in range(1, lim):
                        for curr_state in self.rewards.keys():
                            for curr_action in self.actions:
                                if self.r_s_a_counts[curr_state][curr_action] >= self.s_a_threshold:
                                    self.q_func[curr_state][curr_action] = self._get_reward(curr_state, curr_action) + (
                                                self.gamma * self.get_transition_q_value(curr_state, curr_action))

    def get_transition_q_value(self, state, action):
        return sum(
            [(self._get_transition(state, action, next_state) * self.get_max_q_value(next_state)) for next_state in
             self.q_func.keys()])

    def get_value(self, state):
        return self.get_max_q_value(state)

    def _compute_max_qval_action_pair(self, state):
        # Grab random initial action in case all equal
        best_action = random.choice(self.actions)
        max_q_val = self.get_q_value(state, best_action)

        # Find best action (action w/ current max predicted Q value)
        for action in self.actions:
            q_s_a = self.get_q_value(state, action)
            if q_s_a > max_q_val:
                max_q_val = q_s_a
                best_action = action
        return max_q_val, best_action

    def get_max_q_action(self, state):
        return self._compute_max_qval_action_pair(state)[1]

    def get_max_q_value(self, state):
        return self._compute_max_qval_action_pair(state)[0]

    def get_q_value(self, state, action):
        return self.q_func[state][action]

    def _get_reward(self, state, action):
        if self.r_s_a_counts[state][action] >= self.s_a_threshold:
            # Compute MLE if we've seen this s,a pair enough.
            rewards_s_a = self.rewards[state][action]
            return float(sum(rewards_s_a)) / len(rewards_s_a)
        else:
            # Otherwise return rmax.
            return self.rmax

    def _get_transition(self, state, action, next_state):
        return self.transitions[state][action][next_state] / self.t_s_a_counts[state][action]

    def rmax_policy(self, state, R, T):
        expected_rewards = [np.sum()]
        return np.argmax(expected_rewards)

    def execute(self):
        if os.path.isfile('TRmax.json') and os.access('TRmax.json', os.R_OK):
            with open('TRmax.json') as f:
                data = f.read()
            data = ast.literal_eval(data[data.index('{'):data.rindex('}') + 1])
            self.transitions = defaultdict(list, data)
        current_state = self.services.perception.get_state()

        dict = current_state
        for key in dict:
            val = dict[key]
            if isinstance(val, (set)):
                dict[key] = list(val)
                dict[key].sort()
        json_represntion = json.dumps(dict, sort_keys=True)
        current_state_hash = hashlib.sha1(json_represntion).hexdigest()
        best_ai = 0
        best_ai_value = 0
        i = 0
        for bla in self.transitions:
            if current_state_hash == bla:
                for action in self.transitions[bla]:
                    for next_state in self.transitions[bla][action]:
                        if i == 0:
                            best_ai = action
                            best_ai_value = self.transitions[bla][best_ai][next_state]
                        else:
                            if best_ai_value < self.transitions[bla][action][next_state] or best_ai_value == self.transitions[bla][action][next_state]:
                                best_ai = action
                                best_ai_value = self.transitions[bla][action][next_state]
                        i = i + 1
        return best_ai

