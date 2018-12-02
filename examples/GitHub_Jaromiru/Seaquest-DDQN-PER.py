# OpenGym Seaquest-v0
# -------------------
#
# This code demonstrates a Double DQN network with Priority Experience Replay
# in an OpenGym Seaquest-v0 environment.
#
# Made as part of blog series Let's make a DQN, available at: 
# https://jaromiru.com/2016/11/07/lets-make-a-dqn-double-learning-and-prioritized-experience-replay/
# 
# author: Jaromir Janisch, 2016

import random, numpy, math, gym, scipy
from SumTree import SumTree
from collections import deque

LEARNING_RATE = 0.0001

#-------------------- UTILITIES -----------------------
#-------------------- BRAIN ---------------------------
from keras.models import Sequential
from keras.layers import *
from keras.optimizers import *

class Brain:
    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.model = self._createModel()
        self.model_ = self._createModel()  # target network

    def _createModel(self):
        model = Sequential()
        #print(self.stateCnt)
        model.add(Dense(128, input_dim=(self.stateCnt), activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.actionCnt, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=LEARNING_RATE))
        return model


    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=32, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        #print("PREDICT", s)
        if target:
            return self.model_.predict(s)
        else:
            return self.model.predict(s)

    def predictOne(self, s, target=False):
        #return self.predict(s.reshape(1, IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT), target).flatten()
        return self.predict(s.reshape(1, self.stateCnt), target=target).flatten()
		
    def updateTargetModel(self):
        self.model_.set_weights(self.model.get_weights())

#-------------------- MEMORY --------------------------
class Memory:   # stored as ( s, a, r, s_ ) in SumTree
    e = 0.01
    a = 0.6

    def __init__(self, capacity):
        self.tree = SumTree(capacity)

    def _getPriority(self, error):
        return (error + self.e) ** self.a

    def add(self, error, sample):
        p = self._getPriority(error)
        self.tree.add(p, sample) 

    def sample(self, n):
        batch = []
        segment = self.tree.total() / n

        for i in range(n):
            a = segment * i
            b = segment * (i + 1)

            s = random.uniform(a, b)
            (idx, p, data) = self.tree.get(s)
            batch.append( (idx, data) )

        return batch

    def update(self, idx, error):
        p = self._getPriority(error)
        self.tree.update(idx, p)

#-------------------- AGENT ---------------------------
MEMORY_CAPACITY = 2**16

BATCH_SIZE = 32

GAMMA = 0.99

MAX_EPSILON = 1.0
MIN_EPSILON = 0.00

EXPLORATION_STOP = 500000   # at this step epsilon will be 0.01
LAMBDA = - math.log(0.01) / EXPLORATION_STOP  # speed of decay

UPDATE_TARGET_FREQUENCY = 10000

class Agent:
    steps = 0
    epsilon = MAX_EPSILON

    def __init__(self, stateCnt, actionCnt):
        self.stateCnt = stateCnt
        self.actionCnt = actionCnt

        self.brain = Brain(stateCnt, actionCnt)
        # self.memory = Memory(MEMORY_CAPACITY)
        
    def act(self, s):
        if random.random() < self.epsilon:
            return random.randint(0, self.actionCnt-1)
        else:
            return numpy.argmax(self.brain.predictOne(s))

    def observe(self, sample):  # in (s, a, r, s_) format
        x, y, errors = self._getTargets([(0, sample)])
        self.memory.add(errors[0], sample)

        if self.steps % UPDATE_TARGET_FREQUENCY == 0:
            self.brain.updateTargetModel()

        # slowly decrease Epsilon based on our eperience
        self.steps += 1
        self.epsilon = MIN_EPSILON + (MAX_EPSILON - MIN_EPSILON) * math.exp(-LAMBDA * self.steps)
        

    def _getTargets(self, batch):
        no_state = numpy.zeros(self.stateCnt)

        states = numpy.array([ o[1][0] for o in batch ])
        states_ = numpy.array([ (no_state if o[1][3] is None else o[1][3]) for o in batch ])
        #states = numpy.array([ o[0] for o in batch ])
        #states_ = numpy.array([ (no_state if o[3] is None else o[3]) for o in batch ])
        
        #print("AMEY", states, states_)
        p = agent.brain.predict(states)

        p_ = agent.brain.predict(states_, target=False)
        pTarget_ = agent.brain.predict(states_, target=True)

        x = numpy.zeros((len(batch), self.stateCnt))
        y = numpy.zeros((len(batch), self.actionCnt))
        errors = numpy.zeros(len(batch))
        
        for i in range(len(batch)):
            o = batch[i][1]
            s = o[0]; a = o[1]; r = o[2]; s_ = o[3]
            
            t = p[i]
            oldVal = t[a]
            if s_ is None:
                t[a] = r
            else:
                t[a] = r + GAMMA * pTarget_[i][ numpy.argmax(p_[i]) ]  # double DQN

            x[i] = s
            y[i] = t
            errors[i] = abs(oldVal - t[a])

        return (x, y, errors)

    def replay(self):    
        batch = self.memory.sample(BATCH_SIZE)
        x, y, errors = self._getTargets(batch)

        #update errors
        for i in range(len(batch)):
            idx = batch[i][0]
            self.memory.update(idx, errors[i])

        self.brain.train(x, y)

class RandomAgent:
    memory = Memory(MEMORY_CAPACITY)
    exp = 0
    epsilon = 0

    def __init__(self, actionCnt):
        self.actionCnt = actionCnt

    def act(self, s):
        return random.randint(0, self.actionCnt-1)

    def observe(self, sample):  # in (s, a, r, s_) format
        error = abs(sample[2])  # reward
        self.memory.add(error, sample)
        self.exp += 1

    def replay(self):
        pass

#-------------------- ENVIRONMENT ---------------------
class Environment:
    def __init__(self, problem):
        self.problem = problem
        self.env = gym.make(problem)
        self.running_rewards    = deque([], maxlen=100)
        self.episodes = 0

    def mean_reward(self):
        return np.mean(self.running_rewards)
        	
    def run(self, agent):                
        s = self.env.reset()

        #s = np.reshape(s, [1, stateCnt])

        R = 0
        while True:         
            # self.env.render()
            a = agent.act(s) #1

            r = 0
            s_, r, done, info = self.env.step(a)

            if done: # terminal state
                s_ = None

            agent.observe( (s, a, r, s_) ) #2
            agent.replay()            #3
            #print("Epsilon - {}".format(agent.epsilon))

            s = s_
            R += r

            if done:
                break
        
        self.episodes+=1
        self.running_rewards.append(R)
        mean_reward = self.mean_reward()
        
        print('#{}, Reward: {:.2f}, RunningAvgReward {:.2f} Epsilon {}'.format(self.episodes, R, mean_reward,agent.epsilon))


#-------------------- MAIN ----------------------------
PROBLEM = 'LunarLander-v2'
EPISODE_COUNT = 2000
ec = 0 
env = Environment(PROBLEM)

stateCnt  = env.env.observation_space.shape[0]#(IMAGE_STACK, IMAGE_WIDTH, IMAGE_HEIGHT)
actionCnt = env.env.action_space.n

agent = Agent(stateCnt, actionCnt)
randomAgent = RandomAgent(actionCnt)

try:
    print("Initialization with random agent...")
    while randomAgent.exp < MEMORY_CAPACITY:
        env.run(randomAgent)
        print(randomAgent.exp, "/", MEMORY_CAPACITY)

    agent.memory = randomAgent.memory

    randomAgent = None

    print("Starting learning")
    while ec != EPISODE_COUNT :
        env.run(agent)
        ec += 1 # increment episode counter
finally:
    agent.brain.model.save("LL-DQN-PER.h5")
