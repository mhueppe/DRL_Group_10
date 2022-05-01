import numpy as np

from .gridworld import Gridworld
from .agent import Agent
class SARSA:
    def __init__(self, n, grid_shape, num_pun=4, learning_rate=0.6, gamma=0.8):
        self.n = n
        self.grid = Gridworld(grid_shape, num_pun)
        self.agent = Agent(grid_shape)
        self.learning_rate = learning_rate
        self.gamma = gamma

    # Noch n reinbringen
    def episode_old(self):
        print(
            "One episode################################################################################")
        end = False
        # print("Im in")

        while end == False:

            print("New step")

            old_state = self.agent.position

            # select action
            action_ind, action, q_val_old = self.agent.select_action()
           # print("action typeeee", action)

            # execute action in state class
            # return new agent position
            # return rewar
            im_rewards = []

            # n-Steps/Sampling
            for i in range(self.n):
                if end == True:
                    break
                else:
                    #print("im in")
                    # execute action
                    reward, end = self.grid.step(action, self.agent)
                    #print("STOPPPPPPPPPPPPPPPPPPPPPPPPPPPPP")
                    im_rewards.append(reward)
                    # Choose r# from s'
                    _, _, q_val_new = self.agent.select_action()

            # Update q-Table
            #print("im rew", im_rewards)

            sum_dir_rew = [self.gamma * (i + 1) * im_rewards[i] for i in
                           range(self.n)]

            self.agent.q_table[action_ind, old_state[0], old_state[
                1]] = q_val_old + self.learning_rate * (
                        sum_dir_rew + q_val_new - q_val_old)


    def episode(self, n=3):
        end = False
        total_t = 10000




        action_i = None

        state_memory = []
        q_values_memory = []
        action_index_memory = []

        #First action
        action_ind, action, q_val_old = self.agent.select_action()
        action_to_execute = action
        #S_0 position
        state_memory.append(self.agent.position)
        action_index_memory.append(action_ind)
        q_values_memory.append(q_val_old)

        im_rewards = [0]

        for t in range(total_t):



            if t < total_t:
                print("t", t)

                # execute action
                reward, end = self.grid.step(action_to_execute, self.agent)
                im_rewards.append(reward)
                #new position S1
                state_memory.append(self.agent.position)

                if end:
                    total_t = t+1
                    print("im end")

                else:
                    #print("in Else")
                    print("halllllllllllllllllllllllllllllllllllllllllllllllllll")
                    action_ind, action, q_val_old = self.agent.select_action()

                    action_to_execute = action
                    #action_i = action_ind
                    #von S1
                    q_values_memory.append(q_val_old)
                    action_index_memory.append(action_ind)
                    #reward, end = self.grid.step(action, self.agent)
                    #im_rewards.append(reward)


            tau = t - n + 1 #time whose estimate is being updated
            print("tau", tau)

            #only executed after nth iteration
            if tau >= 0:
                print("in update")
               # agent_pos = self.agent.position
               # q_val_old = self.agent.q_table[]
                #im_rewards = []
                returns = None
                #sum
                # for i in range(tau+n):
                #     returns = sum([np.power(self.gamma, i) * r for r in im_rewards])
                discounted = []


                for p, r in enumerate(im_rewards[tau+1: min(tau+n, total_t)]):
                    discounted.append(np.power(self.gamma, p) * r)

                print("Im here")

                returns = sum(discounted)

                #not end
                if tau + n < total_t:
                    print("im in")
                    #execute action
                    #self.grid.step(action_to_execute, self.agent)
                    print("im rewards", im_rewards)
                    q_value_next_state = self.agent.q_table[action_index_memory[tau+n], state_memory[tau+n][0], state_memory[tau+n][1]]

                    returns = returns + np.power(self.gamma, n) * q_value_next_state

                    print("action index",action_index_memory[tau])

                    act_ind = action_index_memory[tau]

                    #returns = returns + np.power(self.gamma, n) * self.agent.q_table[action_i, self.agent.position[0], self.agent.position[1]]
                    self.agent.q_table[act_ind, state_memory[tau][0], state_memory[tau][1]] += self.learning_rate*(returns - self.agent.q_table[action_index_memory[tau], state_memory[tau][0], state_memory[tau][1]])
                    print(self.agent.q_table)

    def go(self, it=10):

        for i in range(it):
            print(self.grid)
            print("new episode")
            #print("hallo")
            self.episode()
