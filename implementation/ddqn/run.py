import numpy as np
from collections import deque
from environment import Environment
from agent import Agent
import tensorflow as tf

import logging

import os
from hyperparameters import MAX_MEMORY_LENGTH, BATCH_SIZE, GAMMA, \
                            EPSILON_MAX, EPSILON_MIN, EPSILON_DECAY, BRAIN_LEARNING_RATE, MIN_MEMORY_LENGTH

logging.basicConfig(level=logging.INFO)

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

def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # relative path of the main directory
EXPERIMENTS_DIR = os.path.join(MAIN_DIR, "experiments/Dueling_v4")  # relative path of experiments dir

session = tf.Session()

summary_writer = tf.summary.FileWriter(EXPERIMENTS_DIR, session.graph)


# Setup experiment dir and logfile
if not os.path.exists(EXPERIMENTS_DIR):
    os.makedirs(EXPERIMENTS_DIR)
file_handler = logging.FileHandler(os.path.join(EXPERIMENTS_DIR, "log.txt"))
logging.getLogger().addHandler(file_handler)

logging.info("Beginning Episodes...")
logging.info("----------HyperParameters Used----------")
logging.info("----------Batch Size  :%d" % BATCH_SIZE)
logging.info("----------LearningRate:%f" % BRAIN_LEARNING_RATE)
logging.info("----------Discount(Gamma)    :%f" % GAMMA)
logging.info("----------Epsilon Min :%f" % EPSILON_MIN)
logging.info("----------Epsilon Max :%f" % EPSILON_MAX)
logging.info("----------Epsilon Decau :%f" % EPSILON_DECAY)
logging.info("----------Min Memory Length    :%d" % MIN_MEMORY_LENGTH)
logging.info("----------Max Memory Length :%d" % MAX_MEMORY_LENGTH)

scores_window = deque(maxlen=100)
statistics = {"mean": [], "std": []}
scores = []
for e in range(episodes):
    episode_reward, number_of_frames = env.run_episode(agent)
    reward_avg.append(episode_reward)
    # print('episode: ', e, ' score: ', '%.2f' % episode_reward, ' avg_score: ', '%.2f' % np.average(
    #     reward_avg), ' frames: ', number_of_frames, ' epsilon: ', '%.2f' % agent.epsilon)
    logging.info('episode: %s, score: %s, average_score : %s, number of frames: %s, epsilon: %s'% (str(e), (str(episode_reward)), str(np.average(reward_avg)), str(number_of_frames),str(agent.epsilon)))

    write_summary(np.average(reward_avg),"Average Reward",summary_writer, e)
    write_summary(episode_reward,"Episode Reward",summary_writer, e)
    write_summary(number_of_frames,"Number of Frames",summary_writer, e)

    scores.append(episode_reward)
    scores_window.append(episode_reward)
    statistics["mean"].append(np.mean(scores_window))
    statistics["std"].append(np.std(scores_window))

    # if np.mean(scores_window) >= 200.00:
    #     print("\nLunarLander-v2 Environment solved in {:d} episodes!".format(e - 100))
    #     break

env.close()


