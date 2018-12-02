import numpy as np
import os
from collections import deque
from environment import Environment
from agent import Agent

viewOnly = False

PROBLEM = 'LunarLander-v2'
env = Environment(PROBLEM)

np.set_printoptions(precision=2)

agent = Agent(env.number_of_states(), env.number_of_actions())
output_dir = 'model_output/'
if not os.path.exists(output_dir):
    os.makedirs(output_dir)


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
    average_score = np.average(reward_avg)
    training_stats = 'episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % average_score, ' frames: ', number_of_frames, ' epsilon: ', '%.2f' % agent.epsilon
    print(training_stats)
    with open(output_dir + 'trained_agent.txt', 'a') as f:
        f.write(str([e, episode_reward, average_score, number_of_frames, agent.epsilon]) + '\n')
    if e % 50 == 0:
        agent.brain.save_weights(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5", True)

env.close()
