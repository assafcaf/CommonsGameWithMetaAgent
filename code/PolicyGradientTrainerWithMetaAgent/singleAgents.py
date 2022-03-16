import os
import numpy as np
import tensorflow as tf
import tensorflow.keras as keras
from tensorflow.keras import initializers
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, BatchNormalization, GlobalMaxPool2D, MaxPool2D,\
    TimeDistributed, GRU, Dropout
import sys

# local imports
sys.path.append(os.path.join(os.getcwd()))


class PGAgent:
    def __init__(self, input_shape, n_actions, save_directory, gamma=0.95, lr=0.001):
        """
        :param input_shape: tuple, dimensions of observation in the environment
        :param n_actions: int, amount of possible actions
        :param save_directory: str, path to where to save neural-network to
        :param gamma: float, discount factor parameter (usually in range of 0.95-1)
        :param lr: float, learning rate of neural-network
        """
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.save_directory = save_directory

        self.policy = self.__build_conv_model(lr)

        # create directory to save models
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        self.states, self.actions, self.rewards = self.clean_memory()

    @staticmethod
    def clean_memory():
        """
        :return: return 3 empty lists
        """
        return [], [], []

    def get_action(self, obs):
        """
        chose action according to epsilon greedy
        :param obs: np.array, observation
        :return: int, indicate which action to take according to agent prediction
        """
        action = None
        if obs is not None:
            probs = self.policy.call(inputs=tf.convert_to_tensor([obs])).numpy().flatten()
            action = np.random.choice(self.n_actions, p=probs)
        return action

    def store(self, state, action, reward):
        """
        store all the relevant information to memory
        :param state:  np.array, current observation
        :param action: int, action agent performed
        :param reward: float, immediate reward corresponding to the action
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def pop(self):
        """
        retrieve all data in memory and empy content
        :return tuple, (states, actions, discounted_rewards):
        """
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = self.discounted_reward(np.array(self.rewards))

        self.states = []
        self.actions = []
        self.rewards = []

        return states, actions, rewards

    def discounted_reward(self, rewards):
        """
        take 1D float array of rewards and compute discounted reward
        :param rewards: list, indicate each time index a reward has given
        :return np.array, discounted rewards
        """
        r = np.array(rewards)
        discounted_r = np.zeros_like(r)
        running_add = 0
        cnt = 0
        # run all rewards from
        for t in reversed(range(len(r))):
            if cnt > 25:
                running_add = 0
                cnt = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            cnt += 1
        if sum(discounted_r):
            return (discounted_r - discounted_r.mean()) / discounted_r.std()
        else:
            return discounted_r

    def trainable_variables(self):
        """
        :return: keras model variables
        """
        return self.policy.trainable_variables

    def fit(self):
        """
        implementation of standard Gradient Policy
        loss given by ---> cross_entropy * discounted_reward
        @return: float, the loss value current states actions and discounted_rewards
        """
        # get data from agent memory
        states, actions, discounted_rewards = self.pop()
        fit_result = self.policy.fit(states, actions, verbose=0, sample_weight=discounted_rewards)
        return fit_result.history["loss"][0]

    def save(self):
        """
            save the current agent neural-networks to files
        """
        self.policy.save(os.path.join(self.save_directory, "policy_net.h5"))

    def load(self, path):
        """
            load agent neural-network from files
        @param path: str, path to where to find the h5 file to load
        """
        self.policy = load_model(os.path.join(path, "policy_net.h5"))

    def __build_dense_model(self, lr):
        """
                  build keras.Sequential model,
        @param loss: str/ tk.keras.losses.<loss> , specified the loss function of the neural-network ("mse", "mae", "tf.keras.losses.Huber()")
        @param lr: float, learning rate
        @return: keras.Sequential model for predictions, keras.Sequential target model
        """
        q_net = keras.models.Sequential()
        q_net.add(Input(shape=self.input_shape))

        q_net.add(Flatten())

        q_net.add(Dense(256, activation="relu"))
        q_net.add(Dense(64, activation="relu"))
        q_net.add(Dense(32, activation="relu"))
        q_net.add(Dense(self.n_actions, activation="softmax"))

        # compile q net
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name='sparse_categorical_crossentropy'
        )
        q_net.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

        return q_net

    def __build_conv_model(self, lr):
        """
        build keras.Sequential conv model
        @param lr: float, learning rate
        @return: keras.Sequential model
        """
        policy = keras.models.Sequential([
            # conv layers
            Conv2D(filters=128, kernel_size=(5, 5), strides=(1, 1), activation='relu', input_shape=self.input_shape,
                   kernel_initializer=initializers.RandomNormal(stddev=0.01), padding='same', use_bias=True),
            Conv2D(filters=32, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same",
                   kernel_initializer=initializers.RandomNormal(stddev=0.01),  use_bias=True),
            Conv2D(filters=16, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same",
                   kernel_initializer=initializers.RandomNormal(stddev=0.01),  use_bias=True),
            MaxPool2D(pool_size=(2, 2), strides=(2, 2)),

            # flatten and dense
            Flatten(),
            Dense(128, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), use_bias=True),
            Dense(64, activation='relu', kernel_initializer=initializers.RandomNormal(stddev=0.01), use_bias=True),
            Dense(self.n_actions, activation='softmax', kernel_initializer=initializers.RandomNormal(stddev=0.01),
                  use_bias=True)
        ])

        # compile q net
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name='sparse_categorical_crossentropy'
        )
        policy.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        return policy


class PGMetaAgent:
    def __init__(self, input_shape, n_actions, save_directory, gamma=0.95, lr=0.001):
        """
        @param input_shape: tuple, dimensions of states in the environment
        @param n_actions: int, amount of possible actions
        @param save_directory: str, path to where to save neural-network to
        @param gamma: float, Policy Gradient parameter (discount factor)
        @param lr: float, learning rate of neural-network
        """
        self.gamma = gamma
        self.n_actions = n_actions
        self.input_shape = input_shape
        self.save_directory = save_directory
        self.action_space = [.25, .5, 1, 2, 4]
        self.policy = self.__build_conv_video_model(lr)

        # create directory to save models
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        self.states, self.actions, self.rewards = [], [], []

    def act(self, obs):
        """
        chose action influencing the respawn rate of the apples
        :param obs: np.array, current observation
        :return: action index: int, indicate which action to take according to agent prediction
        :return: action: float, the real of action to perform

        """

        probs = self.policy.call(inputs=tf.convert_to_tensor([obs])).numpy().flatten()

        action = np.random.choice(self.n_actions, p=probs)
        return action, self.action_space[action]

    def store(self, state, action, reward):
        """
        store all the relevant information to memory
        @param state:  np.array, current state that came from environment
        @param action: int, current action
        @param reward: float, current reward
        """
        self.states.append(state)
        self.actions.append(action)
        self.rewards.append(reward)

    def pop(self):
        """
        retrieve all data in memory and empy content
        :return tuple, (states, actions, discounted_rewards):
        """
        states = np.array(self.states)
        actions = np.array(self.actions)
        rewards = self.discounted_reward(np.array(self.rewards))

        self.states = []
        self.actions = []
        self.rewards = []

        return states, actions, rewards

    def discounted_reward(self, rewards):
        """
        take 1D float array of rewards and compute discounted reward
        :param rewards: list, indicate each time index a reward has given
        :return np.array, discounted rewards
        """
        r = np.array(rewards)
        discounted_r = np.zeros_like(r)
        running_add = 0
        cnt = 0
        for t in reversed(range(len(r))):
            if cnt > 25:
                running_add = 0
                cnt = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            cnt += 1
        if sum(discounted_r):
            return (discounted_r - discounted_r.mean()) / discounted_r.std()
        else:
            return discounted_r

    def trainable_variables(self):
        """
        :return: keras model variables
        """
        return self.policy.trainable_variables

    def fit(self):
        """
        implementation of standard Gradient Policy
        loss given by ---> cross_entropy * discounted_reward
        @return: float, the loss value current states actions and discounted_rewards
        """
        # get data from agent memory
        # get data from agent memory
        states, actions, rewards = self.pop()
        fit_result = self.policy.fit(states, actions, verbose=0, sample_weight=rewards)
        return fit_result.history["loss"][0]

    def save(self):
        """
        save the current agent neural-networks to files
        """
        self.policy.save(os.path.join(self.save_directory, "policy_net.h5"))

    def load(self, path):
        """
        load agent neural-network from files
        :param path: str, path to where to find the h5 file to load
        """
        self.policy = load_model(os.path.join(path, "policy_net.h5"))

    def __build_dense_model(self, lr):
        """
        build keras.Sequential model,
        :param loss: str/ tk.keras.losses.<loss> , specified the loss function of the neural-network ("mse", "mae", "tf.keras.losses.Huber()")
        :param lr: float, learning rate
        :return: keras.Sequential model for predictions, keras.Sequential target model
        """
        q_net = keras.models.Sequential()
        q_net.add(Input(shape=self.input_shape))

        q_net.add(Flatten())

        q_net.add(Dense(256, activation="relu"))
        q_net.add(Dense(64, activation="relu"))
        q_net.add(Dense(32, activation="relu"))
        q_net.add(Dense(self.n_actions, activation="softmax"))

        # compile q net
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name='sparse_categorical_crossentropy'
        )
        q_net.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)

        return q_net

    def __build_conv_model(self, lr):
        """
                  build keras.Sequential model,
        @param loss: str/ tk.keras.losses.<loss> , specified the loss function of the neural-network ("mse", "mae", "tf.keras.losses.Huber()")
        @param lr: float, learning rate
        @return: keras.Sequential model for predictions, keras.Sequential target model
        """
        policy = keras.models.Sequential([
            keras.layers.Conv2D(filters=16, kernel_size=(5, 5), strides=(1, 1), activation='relu',
                                input_shape=self.input_shape, padding='same'),
            keras.layers.BatchNormalization(),
            keras.layers.Conv2D(filters=24, kernel_size=(3, 3), strides=(1, 1), activation='relu', padding="same"),
            keras.layers.BatchNormalization(),
            keras.layers.Flatten(),
            keras.layers.Dense(32, activation='relu'),
            keras.layers.Dense(16, activation='relu'),
            keras.layers.Dense(self.n_actions, activation='softmax')
        ])

        # compile q net
        loss = keras.losses.SparseCategoricalCrossentropy(
            from_logits=False, name='sparse_categorical_crossentropy'
        )
        policy.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        policy.summary()
        return policy

    def __build_conv_video_model(self, lr):
        """
        wrapper for conv network to handle a sequence of images (implementation from:
        https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f)
        :param lr: float, learning rate
        :return: keras.model, policy model
        """
        # Create our convnet with (112, 112, 3) input shape
        build_conv_net = self.build_conv_net()

        # then create our final model
        model = keras.Sequential()

        # add the convnet with (5, 112, 112, 3) shape
        model.add(TimeDistributed(build_conv_net, input_shape=self.input_shape))
        # here, you can also use GRU or LSTM
        model.add(GRU(64))
        # and finally, we make a decision network
        model.add(Dense(128, activation='relu'))
        model.add(Dense(64, activation='relu'))
        model.add(Dense(self.n_actions, activation='softmax'))

        # compile model
        loss = keras.losses.SparseCategoricalCrossentropy(from_logits=False, name='sparse_categorical_crossentropy')
        model.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss=loss)
        model.summary()
        return model

    def build_conv_net(self):
        """
        build conv network
        implementation from:
        https://medium.com/smileinnovation/training-neural-network-with-image-sequence-an-example-with-video-as-input-c3407f7a0b0f
        :param lr: float, learning rate
        :return: keras.model, policy model
        """
        momentum = .9
        model = keras.Sequential()
        model.add(Conv2D(64, (3, 3), input_shape=self.input_shape[1:], padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(128, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(256, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        # flatten...
        model.add(GlobalMaxPool2D())
        return model




