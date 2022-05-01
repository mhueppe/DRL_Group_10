import numpy as np
from enum import Enum

class Action(Enum):
    UP: Enum = [0,1]
    DOWN: Enum = [0,-1]
    LEFT: Enum = [-1, 0]
    RIGHT: Enum = [1, 0]

class Agent:
    def __init__(self, shape, actions=None):
        if actions is None:
            actions = [Action.UP, Action.DOWN, Action.LEFT, Action.RIGHT]
        self.position = [0, 0]
        self.actions = actions
        self.q_table = np.zeros((len(self.actions), shape[0], shape[1]))
        self.actionValues = {0: Action.UP, 1: Action.DOWN, 2: Action.LEFT, 3: Action.RIGHT}


    def select_action(self):

        eps = 0.25
        p = np.random.random()
        # random
        if p <= eps:
            ind = np.random.randint(0, 4)
            return ind, self.actions[ind], self.q_table[
                ind, self.position[0], self.position[1]]

        # choose according to policy/max Q-value action
        else:
            q_vals = self.q_table[:, self.position[0], self.position[1]]
            print("q vals", q_vals)

            #is_equal = np.all(q_vals == q_vals[0])
            max_q_vals_ind = np.argmax(q_vals)

            print("max index", max_q_vals_ind)

        action = self.actions[max_q_vals_ind]
        # idx, action, q_val = self.e_greedy()
        # indx q_table,
        return max_q_vals_ind, action, self.q_table[
            max_q_vals_ind, self.position[0], self.position[1]]
        # return idx, actions[action], q_val
