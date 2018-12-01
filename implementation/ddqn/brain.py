from keras.models import Sequential
from keras.layers import Dense
from keras.optimizers import Adam


class Brain:

    def __init__(self, number_of_states, number_of_actions, learning_rate=0.0001):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.learning_rate = learning_rate
        self.model = self.create_model()
        self.model_ = self.create_model()

    def create_model(self):
        model = Sequential()

        # Add 2 hidden layers with 64 nodes each
        model.add(Dense(64, input_dim=self.number_of_states, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.number_of_actions, activation='linear'))
        model.compile(loss='mse', optimizer=Adam(lr=self.learning_rate))
        return model

    def target_model_update(self):
        self.model_.set_weights(self.model.get_weights())

    def train(self, x, y, epochs=1, verbose=0):
        self.model.fit(x, y, batch_size=64, epochs=epochs, verbose=verbose)

    def predict(self, s, target=False):
        return self.model_.predict(s) if target else self.model.predict(s)

