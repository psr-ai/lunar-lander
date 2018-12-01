import numpy as np
from collections import deque
from environment import Environment
from agent import Agent

viewOnly = False

PROBLEM = 'LunarLander-v2'
env = Environment(PROBLEM)

np.set_printoptions(precision=2)

agent = Agent(env.number_of_states(), env.number_of_actions())

# Set to true to use saved model

if viewOnly:
    agent.model.load_weights('./weights/trained_agent.h5')
    episodes = 100
    agent.epsilon = 0
else:
    episodes = 10000

# Cumulative reward
reward_avg = deque(maxlen=100)

for e in range(episodes):
    episode_reward, number_of_frames = env.run_episode(agent)
    reward_avg.append(episode_reward)
    print('episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(
        reward_avg), ' frames: ', number_of_frames, ' epsilon: ', '%.2f' % agent.epsilon)

env.close()
