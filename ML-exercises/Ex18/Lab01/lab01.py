# -*- coding: utf-8 -*-
"""Lab 01 solution module (same content as Lab02/lab01.py).

Exports:
    - environment: GridWorld simulator
    - MAB_agent:   greedy Multi-Armed-Bandit agent (no exploration)
    - MABe_agent:  epsilon-greedy MAB agent
"""

import numpy as np


class environment:
    """GridWorld simulator.

    Actions: 0=UP, 1=DOWN, 2=LEFT, 3=RIGHT.
    Special cells: when you step into a 'start' cell you are teleported to the
    corresponding 'end' cell and receive the corresponding reward.
    """

    def __init__(self, grid_height, grid_width):
        self.height = grid_height
        self.width = grid_width
        self.start = []
        self.end = []
        self.reward = []
        self.map = np.array([i for i in range(grid_height * grid_width)])
        self.action_space = [0, 1, 2, 3]

    def get_Map(self):
        print(self.map.reshape([self.width, self.height]))

    def get_NumState(self):
        return self.height * self.width

    def map_Designate(self, start_cell, end_cell, reward):
        self.start.append(start_cell)
        self.end.append(end_cell)
        self.reward.append(reward)

    def get_Observation(self, location, action):
        if action == -1:
            return None, self.action_space, None

        if location in self.start:
            idx = self.start.index(location)
            return self.end[idx], self.action_space, self.reward[idx]

        reward = 0
        new_location = location

        if action == 0:    # UP
            if location - self.width >= 0:
                new_location = location - self.width
        elif action == 1:  # DOWN
            if location + self.width <= self.height * self.width - 1:
                new_location = location + self.width
        elif action == 2:  # LEFT
            if location % self.width != 0:
                new_location = location - 1
        elif action == 3:  # RIGHT
            if (location + 1) % self.width != 0:
                new_location = location + 1

        return new_location, self.action_space, reward


class MAB_agent:
    def __init__(self, envir, init_location):
        self.envir = envir
        self.reward_trace = []
        self.location_now = init_location

        num_states = envir.get_NumState()
        num_actions = len(envir.action_space)
        self.Q_table = np.zeros((num_states, num_actions), dtype=float)
        self.count = np.zeros((num_states, num_actions), dtype=int)

        self.prev_state = None
        self.prev_action = None

    def get_TotalReward(self):
        return np.sum(self.reward_trace)

    def _update(self, state, action, reward):
        self.count[state][action] += 1
        n = self.count[state][action]
        self.Q_table[state][action] += (reward - self.Q_table[state][action]) / n

    def _greedy(self, state, action_space):
        q = self.Q_table[state]
        max_q = q.max()
        best = [a for a in action_space if q[a] == max_q]
        return int(np.random.choice(best))

    def getAction(self, observation):
        location_now, action_space, pre_reward = observation

        if location_now is not None:
            self.location_now = location_now

        if pre_reward is not None:
            self.reward_trace.append(pre_reward)
            if self.prev_state is not None and self.prev_action is not None:
                self._update(self.prev_state, self.prev_action, pre_reward)

        action = self._greedy(self.location_now, action_space)
        self.prev_state = self.location_now
        self.prev_action = action

        assert action in action_space, "INVALID action taken"
        return action


class MABe_agent(MAB_agent):
    def __init__(self, envir, init_location, epsilon=0.1):
        super().__init__(envir, init_location)
        self.epsilon = epsilon

    def _epsilon_greedy(self, state, action_space):
        if np.random.rand() <= self.epsilon:
            return int(np.random.choice(action_space))
        return self._greedy(state, action_space)

    def getAction(self, observation):
        location_now, action_space, pre_reward = observation

        if location_now is not None:
            self.location_now = location_now

        if pre_reward is not None:
            self.reward_trace.append(pre_reward)
            if self.prev_state is not None and self.prev_action is not None:
                self._update(self.prev_state, self.prev_action, pre_reward)

        action = self._epsilon_greedy(self.location_now, action_space)
        self.prev_state = self.location_now
        self.prev_action = action

        assert action in action_space, "INVALID action taken"
        return action
