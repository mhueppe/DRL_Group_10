import numpy as np
import numpy.random as rand
import matplotlib.pyplot as plt

class Gridworld:
    def __init__(self, shape: tuple, num_pun: int, blocks=10,
                 terminal_reward=10, num_blocks=3):
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
            self.grid[x[idx], y[idx]] = 5

        plt.ion()

        self.fig, ax = plt.subplots()

        # this example doesn't work because array only contains zeroes
        # alternatively this process can be automated from the data
        self.world = ax.imshow(self.grid,  vmin=-5, vmax=10, cmap="jet")


    def reset(self):
        """should reset the state"""

        self.grid[:, :] = 0

    def visualize(self, agent):
        print("in visualize")
        # prin
        # print("agent position",agent.position)

        vis = self.grid.copy()
        vis[agent.position[0], agent.position[1]] = -5

        self.world.set_data(vis)
        self.fig.canvas.flush_events()


    #         ax = self.fig.add_subplot(111)
    #         shw = ax.imshow(vis)

    #         # make bar
    #         #bar = plt.colorbar(shw)

    #         plt.colorbar(ax)
    #         i = input()
    #         plt.show()
    #         self.fig.canvas.draw()
    #         self.fig.canvas.flush_events()

    def step(self, action, agent):

        """
        Args:
            action: transition values


        """
        print("Action", action)
        action = action.value
        end = False

        print("Agent position", agent.position)
        print(type(agent.position))
        print("action", action)
        print(type(action))
        # update agents' position/state
        # rint("action", action)
        # new_pos = [self.agent_position[0] + action[0], self.agent_position[1] + action[1]]
        new_pos = [agent.position[0] + action[0], agent.position[1] + action[1]]

        # if out of range
        # print("in Step executrion")
        # print("aaction", action)

        # print("agent old position", agent.position)
        # print("New pos y", new_pos[1])
        # print("grid value", self.grid[new_pos[0], new_pos[1]])
        if (new_pos[0] < self.shape[0]) and (new_pos[0] >= 0) and (
                new_pos[1] < self.shape[1]) and (new_pos[1] >= 0):

            # print("No wall")

            # if obstacle

            if self.grid[new_pos[0], new_pos[1]] == 5:
                pass

            else:
                print("Should update")
                print(new_pos)
                # self.agent_position = new_pos
                agent.position = new_pos
                print(agent.position)

                if self.agent_position == 10:
                    end = True

        print("Agent new position", agent.position)
        # reward
        reward = self.grid[agent.position[0], agent.position[1]]

        # print
        self.visualize(agent)

        return (reward, end)
