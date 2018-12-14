import random
from collections import deque


class Memory:   # stored as ( s, a, r, s_ )

    def __init__(self, max_length=1000):
        self.memory = deque(maxlen=max_length)

    def add(self, sample):
        self.memory.append(sample)

    def length(self):
        return len(self.memory)

    def sample(self, batch_size):
        return random.sample(self.memory, batch_size)
