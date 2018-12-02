from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam

from hyperparameters import BRAIN_LEARNING_RATE, BATCH_SIZE, NEURAL_NETWORK_LAYERS


class Brain:

    def __init__(self, number_of_states, number_of_actions):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.model = self.create_model()
        self.model_ = self.create_model()

    def create_model(self):
        model = Sequential()

        # Add 2 hidden layers with 64 nodes each
        for index, layer in enumerate(NEURAL_NETWORK_LAYERS):
            if index == 0:
                model.add(Dense(layer['number_of_nodes'], input_dim=self.number_of_states, activation=layer['activation']))
            else:
                model.add(Dense(layer['number_of_nodes'], activation=layer['activation']))
        model.add(Dense(self.number_of_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=BRAIN_LEARNING_RATE))
        return model

    def target_model_update(self):
        self.model_.set_weights(self.model.get_weights())

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=BATCH_SIZE, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        return self.model_.predict(s) if target else self.model.predict(s)

    def save_weights(self, file_path, target=False):
        self.model_.save_weights(file_path) if target else self.model.save_weights(file_path)

    def load_weights(self, file_path, target=False):
        self.model_.load_weights(file_path) if target else self.model.load_weights(file_path)

