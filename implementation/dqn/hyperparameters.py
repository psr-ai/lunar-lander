# number of instances in episode to keep in memory
MAX_MEMORY_LENGTH = 2**16
BATCH_SIZE = 32 #2 ** 5
# minimum memory populated before sampling
MIN_MEMORY_LENGTH = 2 ** 6
GAMMA = 0.99
EPSILON_MAX = 1  #1
EPSILON_MIN = .01  #0
EPSILON_DECAY = 0.995 #0.998
BRAIN_LEARNING_RATE = 0.0001
NEURAL_NETWORK_LAYERS = [{ 'activation': 'relu', 'number_of_nodes': 128}, { 'activation': 'relu', 'number_of_nodes': 32}]
