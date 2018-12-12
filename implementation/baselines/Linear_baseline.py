from __future__ import print_function

import gym
from gym import wrappers, logger
import numpy as np
import os

class LinearPolicy(object):

    def __init__(self,theta):
        self.w = theta[:-1]
        self.b = theta[-1]

    def get_action(self,observations):
        y = observations.dot(self.w) + self.b
        a = int(y<0)
        return a

def cem(noisy_evaluation, th_mean, batch_size, n_iter, elite_frac, initial_std=1.0):
    """
    Generic implementation of the cross-entropy method for maximizing a black-box function
    f: a function mapping from vector -> scalar
    th_mean: initial mean over input distribution
    batch_size: number of samples of theta to evaluate per batch
    n_iter: number of batches
    elite_frac: each batch, select this fraction of the top-performing samples
    initial_std: initial standard deviation over parameter vectors
    """
    n_elite = int(np.round(batch_size*elite_frac))
    th_std = np.ones_like(th_mean) * initial_std

    for _ in range(n_iter):
        ths = np.array([th_mean + dth for dth in  th_std[None,:]*np.random.randn(batch_size, th_mean.size)])
        ys = np.array([noisy_evaluation(th) for th in ths])
        elite_inds = ys.argsort()[::-1][:n_elite]
        elite_ths = ths[elite_inds]
        th_mean = elite_ths.mean(axis=0)
        th_std = elite_ths.std(axis=0)
        return {'ys' : ys, 'theta_mean' : th_mean, 'y_mean' : ys.mean()}

def do_rollout(agent, env, num_steps, render=False):
    total_rew = 0
    ob = env.reset()
    for t in range(num_steps):
        a = agent.get_action(ob)
        (ob, reward, done, _info) = env.step(a)
        total_rew += reward
        if render and t%3==0: env.render()
        if done: break
    return total_rew, t+1

if __name__ == '__main__':
    logger.set_level(logger.INFO)

    env = gym.make('LunarLander-v2')
    env.seed(0)
    np.random.seed(0)
    params = dict(n_iter=200, batch_size=25, elite_frac = 0.2)
    num_steps = 2000

    # You provide the directory to write to (can be an existing
    # directory, but can't contain previous monitor results. You can
    # also dump to a tempdir if you'd like: tempfile.mkdtemp().

    # Where we save our monitoring files
    experiment_dir = os.path.abspath("./experiments/{}".format(env.spec.id))

    outdir = os.path.join(experiment_dir,'LinearPolicy')

    env = wrappers.Monitor(env, outdir, force=True)


    def noisy_evaluation(theta):
        agent = LinearPolicy(theta)
        rew, T = do_rollout(agent, env, num_steps)
        return rew

    for i in range(20):
        results = cem(noisy_evaluation, np.zeros(env.observation_space.shape[0]+1), **params)
        agent = LinearPolicy(results['theta_mean'])
        do_rollout(agent, env, 200, render=True)

    env.close()