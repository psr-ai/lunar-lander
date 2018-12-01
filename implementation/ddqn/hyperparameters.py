# number of instances in episode to keep in memory
MAX_MEMORY_LENGTH = 2**16
BATCH_SIZE = 2 ** 5
GAMMA = 0.99
EPSILON_MAX = 1
EPSILON_MIN = 0
EPSILON_DECAY = 0.998
BRAIN_LEARNING_RATE = 0.0001

NEURAL_NETWORK_LAYERS = [{ 'activation': 'relu', 'number_of_nodes': 128}, { 'activation': 'relu', 'number_of_nodes': 64}]
