from keras.models import Sequential, Model
from keras.layers import Dense, Lambda, Input, concatenate,Add, Subtract
from keras.optimizers import Adam
import keras.backend as K
from hyperparameters import BRAIN_LEARNING_RATE, BATCH_SIZE, NEURAL_NETWORK_LAYERS

class Brain:

    def __init__(self, number_of_states, number_of_actions, type_of_agent = ''):
        self.number_of_states = number_of_states
        self.number_of_actions = number_of_actions
        self.type_of_agent = type_of_agent
        self.model = self.create_model()
        self.model_ = self.create_model()

    def create_model(self):
        if self.type_of_agent == 'Dueling':
            inp = Input(shape=(self.number_of_states,))
            layer_shared1 = Dense(NEURAL_NETWORK_LAYERS[0]['number_of_nodes'], activation=NEURAL_NETWORK_LAYERS[0]['activation'], kernel_initializer='he_uniform', use_bias=True)(inp)
            layer_shared2 = Dense(NEURAL_NETWORK_LAYERS[1]['number_of_nodes'], activation=NEURAL_NETWORK_LAYERS[0]['activation'], kernel_initializer='he_uniform', use_bias=True)(layer_shared1)
            print("Shared layers initialized....")

            layer_v2 = Dense(1, activation='linear', kernel_initializer='he_uniform', use_bias=True)(layer_shared2)
            layer_a2 = Dense(self.number_of_actions, activation='linear', kernel_initializer='he_uniform', use_bias=True)(
                layer_shared2)
            print("Value and Advantage Layers initialised....")
            #Compute average of advantage function
            layer_mean = Lambda(lambda x: K.mean(x, axis=-1, keepdims=True))(layer_a2)
            temp = layer_v2
            temp2 = layer_mean

            #concatenate value function to advantage function
            for i in range(self.number_of_actions - 1):
                layer_v2 = concatenate([layer_v2, temp], axis=-1)
                layer_mean = concatenate([layer_mean, temp2], axis=-1)

            layer_q = Subtract()([layer_a2, layer_mean])
            layer_q = Add()([layer_q, layer_v2])

            print("Q-function layer initialized.... :)\n")

            model = Model(inp, layer_q)
            model.summary()
        elif self.type_of_agent == 'Linear':
            model = Sequential()

            model.add(Dense(8, input_dim=self.number_of_states,activation='linear'))
            model.add(Dense(self.number_of_actions, activation='linear'))
            # model.add(Dense(self.number_of_actions,kernel_initializer='normal'))
            model.summary()

        else:
            model = Sequential()
            print('number of states', self.number_of_states)
            # Add 2 hidden layers with 64 nodes each
            for index, layer in enumerate(NEURAL_NETWORK_LAYERS):
                if index == 0:
                    model.add(Dense(layer['number_of_nodes'], input_dim=self.number_of_states,
                                    activation=layer['activation']))
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
