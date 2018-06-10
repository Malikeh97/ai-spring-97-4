from util import *
from node import *
import copy
import sys
import numpy as np
import random


class QLearnAgent():
    def __init__(self):
        # Reward matrix
        # R(A(s, s')) in the blog post
        # s indexes the rows, s' the columns
        # So R(A(s,s')) = R[s][s']
        self.R = None

        # State matrix
        # Q(A(s,s')) in the blog post
        # s indexes the rows, s' the columns
        # So Q(A(s,s')) = Q[s][s']
        self.Q = None

        self.num_rows = None
        self.goal_state = None
        self.num_states = None
        self.next_states = {}
        self.trained = False

    # Public Members
    #
    def is_trained(self):
        return self.trained

    # Initializes the learning matricies
    # Q matrix -> zeros
    # R matrix -> 0 if connected, 1 if goal, -1 otherwise
    #
    def initialize(self, maze):
        if maze is None:
            return
        goal = maze.goal
        if goal is None:
            return
        self.num_states = len(maze.cells) * len(maze.cells[0])
        self.num_rows = len(maze.cells[0])

        # Begin by initializing the reward matrix to zero connectivity
        #
        self.R = np.full((self.num_states, self.num_states), -1, dtype=np.float64)

        # For each point in the maze, connect it to its parent
        #
        for y in range(len(maze.cells)):
            for x in range(len(maze.cells[0])):
                current_cell = (y, x)
                neighbors = maze.get_neighbors(current_cell)

                # States may not point to themselves
                #
                current_state = self.__maze_dims_to_state(current_cell)
                neighbor_states = []
                for i in range(len(neighbors)):
                    neighbor_states.append(self.__maze_dims_to_state(neighbors[i]))

                # Create a hashtable for neighboring states to avoid continuous iteration over matrix
                #
                if current_state not in self.next_states:
                    self.next_states[current_state] = []

                for i in range(len(neighbors)):
                    self.next_states[current_state].append(neighbor_states[i])
                    self.R[current_state][neighbor_states[i]] = 0

        # Set terminal state to point to self, as well as all neighbors to point to it
        #
        self.goal_state = self.__maze_dims_to_state(goal)
        for i in range(self.num_states):
            if self.R[i][self.goal_state] != -1 or i == self.goal_state:
                self.R[i][self.goal_state] = 1.0

        # Initialize Q matrix to zeros
        #
        self.Q = np.full((self.num_states, self.num_states), 0.0)
        self.trained = False

    # Trains the agent
    # initialize() should have been called before this function is called
    #
    def train(self, learning_rate, gamma, min_change_per_epoch):
        print('Training...')
        epoch_iteration = 0
        while True:
            previous_q = np.copy(self.Q)

            # Consider multiple states per epoch.
            # Early termination can happen if same state is picked twice
            #
            for i in range(10):
                # Pick a random starting position
                #
                current_state = random.randint(0, self.num_states - 1)

                # Keep iterating until goal is reached
                #
                while current_state != self.goal_state:

                    # Pick a random next state
                    #
                    possible_next_states = self.next_states[current_state]
                    next_state = random.choice(possible_next_states)

                    # Get the outgoing states from next state.
                    # Compute the max Q values of those outgoing states
                    #
                    max_q_next_state = -1
                    next_next_states = self.next_states[next_state]
                    for next_next_state in next_next_states:
                        max_q_next_state = max([max_q_next_state, self.Q[next_state][next_next_state]])

                    # Set Q value for transition from current->next state via bellman equation
                    #
                    old_q = self.Q[current_state][next_state]
                    self.Q[current_state][next_state] = old_q + learning_rate * (self.R[current_state][next_state] + (gamma * max_q_next_state) - old_q)

                    # Move to next state
                    #
                    current_state = next_state

            # Normalize the Q matrix to avoid overflow
            #
            self.Q = self.Q / np.max(self.Q)

            # Check stopping criteria
            diff = np.sum(np.abs(self.Q - previous_q))
            print('In epoch {0}, difference is {1}'.format(epoch_iteration, diff))
            if diff < min_change_per_epoch:
                break

            epoch_iteration += 1

        # Agent is trained!
        #
        self.trained = True

    # Given a starting state, predict the optimal path to the ending state
    # This should be called only on a trained agent
    def solve(self, start):
        if not self.trained:
            return []

        # The first point in the path is the starting state
        # Translate from (y,x) coordinates into state index
        #
        path = [start]
        current_state = self.__maze_dims_to_state(start)

        # Keep going until we reach the goal
        # (or we've visited every spot - safety check to ensure that we don't get stuck in infinite loop)
        #
        while current_state != self.goal_state and len(path) < self.num_states:

            # For all of the next states, determine the state with the highest Q value
            #
            possible_next_states = self.next_states[current_state]
            best_next_state = possible_next_states[0]
            best_next_state_reward = self.Q[current_state][best_next_state]

            for i in range(1, len(possible_next_states)):
                potential_next_state = possible_next_states[i]
                if self.Q[current_state][potential_next_state] > best_next_state_reward:
                    best_next_state = potential_next_state
                    best_next_state_reward = self.Q[current_state][potential_next_state]

            # Move to that state, and add to path
            #
            current_state = best_next_state
            path.append(self.__state_to_maze_dims(current_state))

        return path

    # Private Members
    #

    # Converts (y,x) coordinates to a numerical state
    #
    def __maze_dims_to_state(self, cell):
        return cell[0] * self.num_rows + cell[1]

    # Converts a numerical state to (y,x) coordinates
    #
    def __state_to_maze_dims(self, state):
        y = int(state // self.num_rows)
        x = state % self.num_rows
        return (y, x)
