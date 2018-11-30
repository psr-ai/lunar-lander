import random
import gym
import numpy as np
from collections import deque
from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam
import os # for creating directories

env = gym.make('LunarLander-v2') # initialise environment
state_size = env.observation_space.shape[0]
print(state_size)

action_size = env.action_space.n
print(action_size)

# define batch size for gradient descent (hyperparameter that we can vary with power of 2)
batch_size = 32

# number of episodes (number of episodes we want our agent to play, going to give us data for training)
# randomly remember a number of episodes to train our deep reinforcement learning agent

n_episodes = 1001

# in order to store our model output
output_dir = 'model_output/cartpole'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

# Define Agent

class DQNAgent:
    def __init__(self, state_size, action_size):
        self.state_size = state_size
        self.action_size = action_size

        # define our memory, in order to remember some of the events that happened in each episode, some of the
        # state and action pairs and rewards, use those memories to replay
        # advantages: 1. inefficient to remember every single event that happened in every episode, where we can
        #                generalise (build on the fact that events closer to each other in time are highly correlated)
        #             2. also get a greater diversity of training data

        # enables us to add and remove elements from either end, as we keep adding,
        # we can remove oldest elements
        self.memory = deque(maxlen=2000)

        # hyperparameters, how much we discount the future reward
        # we would want to weight the upcoming actions more, as compared to actions in distant future
        self.gamma = 0.95

        # exploration rate, two modes:
        # 1. Exploitation: using existing knowledge to take actions
        # 2. Exploration: since environments are complex, and if we are stuck with existing "best practices", we might
        #                 not discover something new that is helpful

        # at the beginning, we are just going to explore since we assume our agent doesn't know what best steps
        # to take to exploit the best information out of the environment
        self.epsilon = 1.0

        # with ongoing training, we slowly shift our agent to start exploiting the information it has learnt
        self.epsilon_decay = 0.995

        self.epsilon_min = 0.01

        # step size for stochastic gradient descent optimiser
        self.learning_rate = 0.001

        self.model = self.build_model()

    def build_model(self):
        # neural net to approximate Q-value function:
        model = Sequential()
        model.add(Dense(24, input_dim=self.state_size, activation='relu'))  # 1st hidden layer; states as input
        model.add(Dense(24, activation='relu'))  # 2nd hidden layer
        model.add(Dense(self.action_size, activation='linear'))  # 2 actions, so 2 output neurons: 0 and 1 (L/R)
        model.compile(loss='mse',
                      optimizer=Adam(lr=self.learning_rate))
        return model

    # takes in state at current timestep, takes in action at the current timestep, takes in reward at
    # current timestep. Helps to model that given a state and action, what is going to happen in next state and
    # what kind of reward we expect to receive
    def remember(self, state, action, reward, next_state, done):
        self.memory.append(
            (state, action, reward, next_state, done))  # list of previous experiences, enabling re-training later

    # figure out what action to take given the state
    def act(self, state):
        # incorporate random action based on epsilon value
        if np.random.rand() <= self.epsilon:
            return random.randrange(self.action_size)
        act_values = self.model.predict(state)
        return np.argmax(act_values[0])

    def replay(self, batch_size):
        # we are going to randomly sample some memories
        minibatch = random.sample(self.memory, batch_size)

        for state, action, reward, next_state, done in minibatch:
            target = reward
            if not done:
                target = (reward + self.gamma * np.amax(self.model.predict(next_state)[0]))
            target_f = self.model.predict(state)

            target_f[0][action] = target

            self.model.fit(state, target_f, epochs=1, verbose=0)

        if self.epsilon > self.epsilon_min:
            self.epsilon *= self.epsilon_decay

    def load(self, name):
        self.model.load_weights(name)

    def save(self, name):
        self.model.save_weights(name)


agent = DQNAgent(state_size, action_size)

# Interact with the environment
# done = False
# for e in range(n_episodes):  # iterate over new episodes of the game
#     state = env.reset()  # reset state at start of each new episode of the game
#     state = np.reshape(state, [1, state_size])
#
#     for time in range(
#             5000):  # time represents a frame of the game; goal is to keep pole upright as long as possible up to range, e.g., 500 or 5000 timesteps
#         #         env.render()
#         action = agent.act(state)  # action is either 0 or 1 (move cart left or right); decide on one or other here
#         next_state, reward, done, _ = env.step(
#             action)  # agent interacts with env, gets feedback; 4 state data points, e.g., pole angle, cart position
#         reward = reward if not done else -10  # reward +1 for each additional frame with pole upright
#         next_state = np.reshape(next_state, [1, state_size])
#         agent.remember(state, action, reward, next_state,
#                        done)  # remember the previous timestep's state, actions, reward, etc.
#         state = next_state  # set "current state" for upcoming iteration to the current next state
#         if done:  # episode ends if agent drops pole or we reach timestep 5000
#             print("episode: {}/{}, score: {}, e: {:.2}"  # print the episode's score and agent's epsilon
#                   .format(e, n_episodes, time, agent.epsilon))
#             break  # exit loop
#     if len(agent.memory) > batch_size:
#         agent.replay(batch_size)  # train the agent by replaying the experiences of the episode
#     if e % 50 == 0:
#         agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")


for e in range(n_episodes):
    state = env.reset()
    state = np.reshape(state, [1, state_size])
    is_end = False
    total_reward = 0
    while not is_end:
        action = agent.act(state)
        next_state, reward, is_end, _ = env.step(action)
        total_reward += reward
        next_state = np.reshape(next_state, [1, state_size])
        agent.remember(state, action, reward, next_state, is_end)
        state = next_state
        if is_end:
            print("episode: {}/{}, reward: {}, e: {:.2}"
                  .format(e, n_episodes, total_reward, agent.epsilon))
            break  # exit loop
    if len(agent.memory) > batch_size:
        agent.replay(batch_size)

    if e % 50 == 0:
        agent.save(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5")
