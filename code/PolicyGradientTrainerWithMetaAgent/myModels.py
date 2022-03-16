import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, MaxPooling2D


class MyModel(keras.Model):
    def __init__(self, input_shape, conv_layers, dense_layers, num_actions):
        super(MyModel, self).__init__()

        # build model layer
        # input layer
        self.input_layer = tf.keras.layers.InputLayer(input_shape=(*input_shape,))

        # hidden layers
        self.hidden_layers = []
        for dim in conv_layers:
            self.hidden_layers.append(keras.layers.Conv2D(
                kernel_size=(3, 3), filters=dim, strides=(2, 2), activation='relu', kernel_initializer='RandomNormal'))

        self.hidden_layers.append(keras.layers.Flatten(input_shape=input_shape))
        for units in dense_layers:
            self.hidden_layers.append(keras.layers.Dense(
                units, activation='relu', kernel_initializer='RandomNormal'))

        # output layer
        self.output_layer = keras.layers.Dense(
            num_actions, activation='linear', kernel_initializer='RandomNormal')

        self.build((None, *input_shape))

    @tf.function
    def call(self, inputs):
        z = self.input_layer(inputs)
        for layer in self.hidden_layers:
            z = layer(z)
        output = self.output_layer(z)
        return output


def create_q_models(input_shape, n_actions, loss, lr):
    # Network defined by the Deepmind paper
    inputs = Input(shape=input_shape)

    # Convolutions on the frames on the screen
    layer1 = Conv2D(4, 3,  activation="relu", input_shape=input_shape)(inputs)
    layer2 = Conv2D(4, 3,  activation="relu")(layer1)
    mp = MaxPooling2D((2, 2))(layer2)

    layer4 = Flatten()(mp)

    layer5 = Dense(16, activation="relu")(layer4)
    action = Dense(n_actions, activation="linear")(layer5)

    opt = keras.optimizers.Adam(learning_rate=lr)

    # compile q net
    q_net = keras.Model(inputs=inputs, outputs=action)
    q_net.compile(optimizer=opt, loss=loss)

    # compile q net target
    q_net_target = keras.Model(inputs=inputs, outputs=action)
    q_net_target.compile(optimizer=opt, loss=loss)
    return q_net, q_net_target


def create_q_models_sequential(input_shape, n_actions, loss, lr):
    # Network defined by the Deepmind paper
    q_net = keras.models.Sequential()
    q_net.add(Input(shape=input_shape))

    q_net.add(Conv2D(4, 3,  activation="relu", input_shape=input_shape))
    q_net.add(MaxPooling2D((2, 2)))
    q_net.add(Flatten())
    q_net.add(Dense(24, activation="relu"))
    q_net.add(Dense(n_actions, activation="linear"))

    opt = keras.optimizers.Adam(learning_rate=lr)

    # compile q net
    q_net.compile(optimizer=opt, loss=loss)

    # compile q net target
    q_net_target = keras.models.clone_model(q_net)
    q_net_target.compile(optimizer=opt, loss=loss)

    return q_net, q_net_target
