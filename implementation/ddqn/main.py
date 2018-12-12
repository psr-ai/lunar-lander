
"""This file contains the entrypoint to the rest of the code"""

from __future__ import absolute_import
from __future__ import division

import numpy as np
from collections import deque
from environment import Environment
from agent import Agent
import os
import io
import json
import sys
import logging

import tensorflow as tf

logging.basicConfig(level=logging.INFO)

# High-level options
tf.app.flags.DEFINE_integer("gpu", 0, "Which GPU to use, if you have multiple.")
tf.app.flags.DEFINE_string("experiment_name", "NewExperiment","Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment")
tf.app.flags.DEFINE_integer("num_episodes", 600, "Number of episodes to train. 0 means train indefinitely")
tf.app.flags.DEFINE_boolean("should_learn", True, "Available modes: learning / eval (no exploration). ")
tf.app.flags.DEFINE_boolean("should_render", False, "render the Lander while running.")
tf.app.flags.DEFINE_string("agent", "DDQN", "type of DQN Network to be performed")

# Hyperparameters
NEURAL_NETWORK_LAYERS = [{ 'activation': 'relu', 'number_of_nodes': 128}, { 'activation': 'relu', 'number_of_nodes': 64}]
tf.app.flags.DEFINE_float("BRAIN_LEARNING_RATE", 0.001, "Learning rate.") #original learning rate 0.001
tf.app.flags.DEFINE_integer("MAX_MEMORY_LENGTH", 2**16, "Max memory length.")
tf.app.flags.DEFINE_integer("BATCH_SIZE", 64, "Batch size to use")
tf.app.flags.DEFINE_integer("MIN_MEMORY_LENGTH", 2**6, "Max memory length.")
tf.app.flags.DEFINE_float("GAMMA", 0.99, "Max memory length.")
tf.app.flags.DEFINE_float("EPSILON_MAX", 1, "Max memory length.")
tf.app.flags.DEFINE_float("EPSILON_MIN", 0.01, "Max memory length.")
tf.app.flags.DEFINE_float("EPSILON_DECAY", 0.995, "Max memory length.")

# new added flags
tf.app.flags.DEFINE_string("initial_weights", "weights_0450.hdf5","Take the model parameters for evaluation")

FLAGS = tf.app.flags.FLAGS

os.environ["CUDA_VISIBLE_DEVICES"] = str(FLAGS.gpu)

MAIN_DIR = os.path.relpath(os.path.dirname(os.path.dirname(os.path.abspath(__file__))))  # relative path of the main directory

EXPERIMENTS_DIR = os.path.join(os.path.join(MAIN_DIR, "experiments", FLAGS.experiment_name)) # relative path of experiments dir

if FLAGS.should_learn:
    initial_weights = ""
else:
    initial_weights = os.path.join(EXPERIMENTS_DIR, "weights_0450.hdf5")  # relative path of experiments dir


def write_summary(value, tag, summary_writer, global_step):
    """Write a single summary value to tensorboard"""
    summary = tf.Summary()
    summary.value.add(tag=tag, simple_value=value)
    summary_writer.add_summary(summary, global_step)


def main(unused_argv):

    PROBLEM = 'LunarLander-v2'

    env = Environment(PROBLEM, should_learn=FLAGS.should_learn, should_render=FLAGS.should_render)

    np.set_printoptions(precision=2)

    agent = Agent(env.number_of_states(), env.number_of_actions(), type_of_agent=FLAGS.agent,
                  only_exploitation=not FLAGS.should_learn, initial_weights=initial_weights)

    if FLAGS.should_learn:
        session = tf.Session()
        summary_writer = tf.summary.FileWriter(EXPERIMENTS_DIR, session.graph)

    # Cumulative reward
    reward_avg = deque(maxlen=100)

    # Setup experiment dir and logfile
    if not os.path.exists(EXPERIMENTS_DIR):
        os.makedirs(EXPERIMENTS_DIR)

    if FLAGS.should_learn :
        file_handler = logging.FileHandler(os.path.join(EXPERIMENTS_DIR, "log.txt"))
    else:
        FLAGS.num_episodes = 100
        file_handler = logging.FileHandler(os.path.join(EXPERIMENTS_DIR, "eval_results.txt"))

    logging.getLogger().addHandler(file_handler)

    logging.info("Beginning Training Episodes...")
    logging.info("----------HyperParameters Used----------")
    logging.info("----------Batch Size  :%d" % FLAGS.BATCH_SIZE)
    logging.info("----------LearningRate:%f" % FLAGS.BRAIN_LEARNING_RATE)
    logging.info("----------Discount(Gamma)    :%f" % FLAGS.GAMMA)
    logging.info("----------Epsilon Min :%f" % FLAGS.EPSILON_MIN)
    logging.info("----------Epsilon Max :%f" % FLAGS.EPSILON_MAX)
    logging.info("----------Epsilon Decau :%f" % FLAGS.EPSILON_DECAY)
    logging.info("----------Min Memory Length    :%d" % FLAGS.MIN_MEMORY_LENGTH)
    logging.info("----------Max Memory Length :%d" % FLAGS.MAX_MEMORY_LENGTH)

    scores_window = deque(maxlen=100)

    statistics = {"mean": [], "std": []}
    scores = []

    for e in range(FLAGS.num_episodes): # defauts to 600
        episode_reward, number_of_frames = env.run_episode(agent)
        reward_avg.append(episode_reward)

        logging.info('episode: %s, score: %s, average_score : %s, number of frames: %s, epsilon: %s' % (
        str(e), (str(episode_reward)), str(np.average(reward_avg)), str(number_of_frames), str(agent.epsilon)))

        scores.append(episode_reward)
        scores_window.append(episode_reward)
        statistics["mean"].append(np.mean(scores_window))
        statistics["std"].append(np.std(scores_window))

        if FLAGS.should_learn and e % 50 == 0:
                agent.brain.save_weights(EXPERIMENTS_DIR + "/" +"weights_" + '{:04d}'.format(e) + ".hdf5", True)

    env.close()

    # Print an error message if you've entered flags incorrectly
    if len(unused_argv) != 1:
        raise Exception("There is a problem with how you entered flags: %s" % unused_argv)

    # Check for Python 2
    if sys.version_info[0] != 3:
        raise Exception("ERROR: You must use Python 3 but you are running Python %i" % sys.version_info[0])

    # Some GPU settings
    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True

    if FLAGS.should_learn :
        write_summary(np.average(reward_avg),"Average Reward",summary_writer, e)
        write_summary(episode_reward,"Episode Reward",summary_writer, e)
        write_summary(number_of_frames,"Number of Frames",summary_writer, e)

if __name__ == "__main__":
    tf.app.run()
