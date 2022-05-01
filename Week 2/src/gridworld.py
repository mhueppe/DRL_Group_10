import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

class Gridworld:
    def __init__(self, shape: tuple, num_pun: int, blocks=10,
                 terminal_reward=10, num_blocks=5):
        """

        Args:
            num_pun (int): number of negative rewards that are randomly placed on the board

        """
        self.shape = shape
        self.grid = np.zeros(shape)

        # agent position
        self.agent_position = [0, 0]
        self.fig = plt.figure()
        # set negative rewards
        for i in range(num_pun):
            self.grid[rand.randint(shape[0]), rand.randint(shape[1])] = -1

        # set terminal state
        x, y = np.where(self.grid != -1)
        i = rand.randint(len(list(x)))
        self.grid[x[i], y[i]] = 10

        plt.ion()

        # self.q_table = np.zeros(shape)

        # set blocks. Marked with 5
        x, y = np.where(self.grid == 0)
        for i in range(num_blocks):
            idx = rand.randint(len(list(x)))
            self.grid[x[idx], y[idx]] = -5

        plt.ion()

        #set all to -1

        # x,y = np.where(self.grid == 0)
        # for i in range(len(x)):
        #     self.grid[x[i], y[i]]=-1


        self.fig, self.plots = plt.subplots(ncols=5)

        # this example doesn't work because array only contains zeroes
        # alternatively this process can be automated from the data
        self.world = self.plots[0].imshow(self.grid,  vmin=-5, vmax=15, cmap="jet")
        movement: dict = {"up": 0, "down": 1, "left": 2, "right": 3}
        self.actionPlots = []
        for plot, m in zip(self.plots[1:], movement.keys()):
            a = plot.imshow(np.ones(self.shape), vmin=-5, vmax=5)
            plot.set_title(m)
            self.actionPlots.append(a)


    def reset(self):
        """should reset the state"""

        self.grid[:, :] = 0

    def visualize(self, agent):
        #print("in visualize")
        # prin

        vis = self.grid.copy()
        vis[agent.position[0], agent.position[1]] = 15
        movement: dict = {"up": 0, "down": 1, "left": 2, "right": 3}
        for plot, m in zip(self.actionPlots, movement.keys()):
            plot.set_data(agent.q_table[movement[m]])
            # print(f"{m} : {agent.q_table[movement[m]]}")


        self.world.set_data(vis)
        self.fig.canvas.flush_events()


    def step(self, action, agent):

        """
        Args:
            action: transition values


        """
        action = action.value
        end = False

        # update agents' position/state
        # rint("action", action)
        # new_pos = [self.agent_position[0] + action[0], self.agent_position[1] + action[1]]
        new_pos = [agent.position[0] + action[0], agent.position[1] + action[1]]

        # if out of range
        if (new_pos[0] < self.shape[0]) and (new_pos[0] >= 0) and (
                new_pos[1] < self.shape[1]) and (new_pos[1] >= 0):

            if self.grid[new_pos[0], new_pos[1]] == -5:
                pass

            else:

                agent.position = new_pos

                if self.grid[agent.position[0], agent.position[1]] == 10:
                    print("in terminal")
                    end = True

        # reward
        reward = self.grid[agent.position[0], agent.position[1]]

        # print
        self.visualize(agent)

        return (reward, end)
