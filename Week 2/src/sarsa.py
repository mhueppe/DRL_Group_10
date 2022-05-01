from .gridworld import Gridworld
from .agent import Agent
class SARSA:
    def __init__(self, n, grid_shape, num_pun=4, learning_rate=0.6, gamma=0.3):
        self.n = n
        self.grid = Gridworld(grid_shape, num_pun)
        self.agent = Agent(grid_shape)
        self.learning_rate = learning_rate
        self.gamma = gamma

    # Noch n reinbringen
    def episode(self):
        print(
            "One episode################################################################################")
        end = False
        # print("Im in")

        while end == False:

            print("Neu while")

            old_state = self.agent.position

            # select action
            action_ind, action, q_val_old = self.agent.select_action()
            print("action typeeee", action)

            # execute action in state class
            # return new agent position
            # return rewar
            im_rewards = []

            # n-Steps/Sampling
            for i in range(self.n):

                if end == True:
                    break

                else:

                    print("im in")

                    # execute action
                    reward, end = self.grid.step(action, self.agent)
                    print("STOPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")

                    im_rewards.append(reward)

                    # Choose r# from s'
                    _, _, q_val_new = self.agent.select_action()

            # Update q-Table
            print("im rew", im_rewards)

            sum_dir_rew = [self.gamma * (i + 1) * im_rewards[i] for i in
                           range(self.n)]

            self.agent.q_table[action_ind, old_state[0], old_state[
                1]] = q_val_old + self.learning_rate * (
                        sum_dir_rew + q_val_new - q_val_old)

    def go(self, it=10):

        for i in range(it):
            print("hallo")
            self.episode()
