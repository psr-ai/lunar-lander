from brain import Brain
from memory import Memory
from hyperparameters import MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, \
                            EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY, BRAIN_LEARNING_RATE,NEURAL_NETWORK_LAYERS

import numpy as np


class Agent:
    steps = 0
    epsilon = EPSILON_MAX

    def __init__(self, number_of_states, number_of_actions, type_of_agent='FullDQN'):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions

        self.brain = Brain(number_of_states, number_of_actions)
        self.memory = Memory(MAX_MEMORY_LENGTH)
        self.type_of_agent = type_of_agent

    def act(self, s):
        if np.random.rand() <= self.epsilon:
            return np.random.choice(self.number_of_actions)
        else:
            return np.argmax(self.brain.predict(s))

    def observe(self, instance):
        self.memory.add(instance)

    def update_epsilon(self):
        if self.epsilon > EPSILON_MIN:
            self.epsilon *= EPSILON_DECAY

    def replay(self):
        minibatch = self.memory.sample(BATCH_SIZE)
        minibatch = np.array(minibatch)
        not_done_indices = np.where(minibatch[:, 4] == False)
        y = np.copy(minibatch[:, 2])

        # If minibatch contains any non-terminal states, use separate update rule for those states
        if len(not_done_indices[0]) > 0:
            predict_sprime = self.brain.predict(np.vstack(minibatch[:, 3]))
            predict_sprime_target = self.brain.predict(np.vstack(minibatch[:, 3]), True)

            # Non-terminal update rule
            if self.type_of_agent == 'DDQN':
                y[not_done_indices] += np.multiply(GAMMA, \
                                                   predict_sprime_target[not_done_indices, \
                                                                         np.argmax(predict_sprime[not_done_indices, :][0],
                                                                                   axis=1)][0])
            elif self.type_of_agent == 'FullDQN':
                y[not_done_indices] += np.multiply(GAMMA, np.max(predict_sprime_target[not_done_indices, :][0], axis=1))
        actions = np.array(minibatch[:, 1], dtype=int)
        y_target = self.brain.predict(np.vstack(minibatch[:, 0]))
        y_target[range(BATCH_SIZE), actions] = y
        self.brain.train(np.vstack(minibatch[:, 0]), y_target)
