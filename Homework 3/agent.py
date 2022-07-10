import numpy as np
import buffer
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model

def build_dqn(lr, n_actions, input_dims, fc1_dims, fc2_dims):
    # model = keras.Sequential([
    #     keras.layers.Dense(fc1_dims, activation='relu'),
    #     keras.layers.Dense(fc2_dims, activation='relu'),
    #     keras.layers.Dense(n_actions, activation=None)])
    # model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    inputLayer = keras.layers.Input(input_dims)
    hiddenLayer = keras.layers.Dense(fc1_dims, activation='relu')(inputLayer)
    hiddenLayer1 = keras.layers.Dense(fc2_dims, activation='relu')(hiddenLayer)
    outputLayer = keras.layers.Dense(n_actions, activation=None)(hiddenLayer1)
    model = keras.Model(inputs=inputLayer, outputs=outputLayer)
    model.compile(optimizer=Adam(learning_rate=lr), loss='mean_squared_error')
    return model

class Agent():
    def __init__(self, lr, gamma, n_actions, epsilon, batch_size,
                input_dims, epsilon_dec=1e-3, epsilon_end=0.01,
                mem_size=1000000, fname='dqn_model.h5'):
        self.action_space = [i for i in range(n_actions)]
        self.gamma = gamma
        self.epsilon = epsilon
        self.eps_dec = epsilon_dec
        self.eps_min = epsilon_end
        self.batch_size = batch_size
        self.model_file = fname
        self.memory = buffer.ReplayBuffer(mem_size, input_dims)
        self.q_eval = build_dqn(lr, n_actions, input_dims, 256, 256)

    def store_transition(self, state, action, reward, new_state, done):
        self.memory.store_transition(state, action, reward, new_state, done)

    def choose_action(self, observation):
        if np.random.random() < self.epsilon:
            action = np.random.choice(self.action_space)
        else:
            state = np.array([observation])
            actions = self.q_eval.predict(state)

            action = np.argmax(actions)

        return action

    def learn(self):
        if self.memory.mem_cntr < self.batch_size:
            return

        states, actions, rewards, states_, dones = \
                self.memory.sample_buffer(self.batch_size)

        q_eval = self.q_eval.predict(states)
        q_next = self.q_eval.predict(states_)


        q_target = np.copy(q_eval)
        batch_index = np.arange(self.batch_size, dtype=np.int32)

        q_target[batch_index, actions] = rewards + \
                        self.gamma * np.max(q_next, axis=1)*dones


        self.q_eval.train_on_batch(states, q_target)

        self.epsilon = self.epsilon - self.eps_dec if self.epsilon > \
                self.eps_min else self.eps_min

    def save_model(self):
        self.q_eval.save(self.model_file)


    def load_model(self):
        self.q_eval = load_model(self.model_file)
