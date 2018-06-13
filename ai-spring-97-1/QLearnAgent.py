from util import *
from node import *
import copy
import sys
import numpy as np
import random


class QLearnAgent():
    def __init__(self):
        self.R = None
        self.Q = None

        self.num_rows = None
        self.goal_state = None
        self.num_states = None
        self.next_states = {}
        self.trained = False

    def is_trained(self):
        return self.trained

    def initialize(self, maze):
        if maze is None:
            return
        goal = maze.goal
        if goal is None:
            return
        self.num_states = len(maze.cells) * len(maze.cells[0])
        self.num_rows = len(maze.cells[0])

        self.R = np.full((self.num_states, self.num_states), -1, dtype=np.float64)

        for y in range(len(maze.cells)):
            for x in range(len(maze.cells[0])):
                current_cell = (y, x)
                neighbors = maze.get_neighbors(current_cell)

                current_state = self.__maze_dims_to_state(current_cell)
                neighbor_states = []
                for i in range(len(neighbors)):
                    neighbor_states.append(self.__maze_dims_to_state(neighbors[i]))

                if current_state not in self.next_states:
                    self.next_states[current_state] = []

                for i in range(len(neighbors)):
                    self.next_states[current_state].append(neighbor_states[i])
                    self.R[current_state][neighbor_states[i]] = 0

        self.goal_state = self.__maze_dims_to_state(goal)
        for i in range(self.num_states):
            if self.R[i][self.goal_state] != -1 or i == self.goal_state:
                self.R[i][self.goal_state] = 1.0

        self.Q = np.full((self.num_states, self.num_states), 0.0)
        self.trained = False

    def train(self, learning_rate, gamma, min_change_per_epoch):
        print('Training...')
        epoch_iteration = 0
        while True:
            previous_q = np.copy(self.Q)

            for i in range(10):
                current_state = random.randint(0, self.num_states - 1)

                while current_state != self.goal_state:

                    possible_next_states = self.next_states[current_state]
                    next_state = random.choice(possible_next_states)

                    max_q_next_state = -1
                    next_next_states = self.next_states[next_state]
                    for next_next_state in next_next_states:
                        max_q_next_state = max([max_q_next_state, self.Q[next_state][next_next_state]])

                    old_q = self.Q[current_state][next_state]
                    self.Q[current_state][next_state] = old_q + learning_rate * (
                                self.R[current_state][next_state] + (gamma * max_q_next_state) - old_q)

                    current_state = next_state

            self.Q = self.Q / np.max(self.Q)

            diff = np.sum(np.abs(self.Q - previous_q))
            print('In epoch {0}, difference is {1}'.format(epoch_iteration, diff))
            if diff < min_change_per_epoch:
                break

            epoch_iteration += 1

        self.trained = True

    def solve(self, start):
        if not self.trained:
            return []

        path = [start]
        current_state = self.__maze_dims_to_state(start)

        while current_state != self.goal_state and len(path) < self.num_states:

            possible_next_states = self.next_states[current_state]
            best_next_state = possible_next_states[0]
            best_next_state_reward = self.Q[current_state][best_next_state]

            for i in range(1, len(possible_next_states)):
                potential_next_state = possible_next_states[i]
                if self.Q[current_state][potential_next_state] > best_next_state_reward:
                    best_next_state = potential_next_state
                    best_next_state_reward = self.Q[current_state][potential_next_state]

            current_state = best_next_state
            path.append(self.__state_to_maze_dims(current_state))

        return path

    def __maze_dims_to_state(self, cell):
        return cell[0] * self.num_rows + cell[1]

    def __state_to_maze_dims(self, state):
        y = int(state // self.num_rows)
        x = state % self.num_rows
        return y, x
