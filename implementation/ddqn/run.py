import numpy as np
import os
from collections import deque
from environment import Environment
from agent import Agent


# Define the parameters here
should_learn = True
should_render = False
output_dir = 'model_output_full_dqn_1/'
# initial_weights = './ddqn/model_output_ddqn/weights_2000.hdf5'
initial_weights = ''
episodes = 10000
type_of_agent = 'FullDQN'
# End parameters

PROBLEM = 'LunarLander-v2'
env = Environment(PROBLEM, should_learn=should_learn, should_render=should_render)

np.set_printoptions(precision=2)

agent = Agent(env.number_of_states(), env.number_of_actions(), type_of_agent=type_of_agent, initial_weights=initial_weights, only_exploitation=not should_learn)

if output_dir and not os.path.exists(output_dir):
    os.makedirs(output_dir)


# Cumulative reward
reward_avg = deque(maxlen=100)

for e in range(episodes):
    episode_reward, number_of_frames = env.run_episode(agent)
    reward_avg.append(episode_reward)
    average_score = np.average(reward_avg)
    training_stats = 'episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % average_score, ' frames: ', number_of_frames, ' epsilon: ', '%.2f' % agent.epsilon
    print(training_stats)
    if output_dir:
        with open(output_dir + 'trained_agent.txt', 'a') as f:
            f.write(str([e, episode_reward, average_score, number_of_frames, agent.epsilon]) + '\n')
        if e % 50 == 0:
            agent.brain.save_weights(output_dir + "weights_" + '{:04d}'.format(e) + ".hdf5", True)

env.close()
