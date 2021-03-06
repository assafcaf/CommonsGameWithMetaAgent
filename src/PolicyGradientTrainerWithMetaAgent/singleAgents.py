import os
import random
import numpy as np
import tensorflow as tf
from utils import rgb2gray
import tensorflow.keras as keras
from .myMemory import QAgentBuffer
from tensorflow.keras import initializers
from utils import clean_memory
import inspect
from tensorflow.keras.models import load_model
from tensorflow.keras.layers import Input, Flatten, Dense, Conv2D, GlobalMaxPool2D, MaxPool2D, TimeDistributed,\
     GRU, Lambda


class PGAgent:
    def __init__(self, input_shape, n_actions, save_directory, gamma=0.95, lr=0.001, normalize=False):
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
        self.normalize = normalize
        self.policy = self.__build_conv_model(lr)

        # create directory to save models
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        self.states, self.actions, self.rewards = clean_memory()

    def get_action(self, obs):
        """
        chose action according to epsilon greedy
        :param obs: np.array, observation
        :return: int, indicate which action to take according to agent prediction
        """
        obs_ = rgb2gray(obs.copy())
        action = None
        if obs_ is not None:
            probs = self.policy.call(inputs=tf.convert_to_tensor([obs_])).numpy().flatten()
            action = np.random.choice(self.n_actions, p=probs)
        return action

    def store(self, state, action, reward):
        """
        store all the relevant information to memory
        :param state:  np.array, current observation
        :param action: int, action agent performed
        :param reward: float, immediate reward corresponding to the action
        """
        if state.sum() != 0:
            self.states.append(rgb2gray(state.copy()))
            self.actions.append(action)
            self.rewards.append(reward)

    def pop(self):
        """
        retrieve all data in memory and empy content
        :return tuple, (states, actions, discounted_rewards):
        """
        states = np.array(self.states)
        actions = np.array(self.actions)
        discounted_rewards = self.discounted_reward(np.array(self.rewards))

        self.states = []
        self.actions = []
        self.rewards = []

        return states, actions, discounted_rewards

    def discounted_reward(self, rewards):
        """
        take 1D float array of rewards and compute discounted reward
        :param rewards: list, indicate each time index a reward has given
        :return np.array, discounted rewards
        """
        r = np.array(rewards)
        discounted_r = np.zeros_like(r, dtype=np.float)
        running_add = 0
        for t in reversed(range(len(r))):
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
        if sum(discounted_r) and self.normalize is True:
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
        # fit_result = self.policy.fit(states, actions, verbose=0, sample_weight=discounted_rewards, batch_size=100)
        loss = self.policy.train_on_batch(states, actions, sample_weight=discounted_rewards)
        return loss

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
    def __init__(self, input_shape, save_directory, act_every, gamma=0.95, lr=0.001, normalize=False):
        """
        @param input_shape: tuple, dimensions of states in the environment
        @param n_actions: int, amount of possible actions
        @param save_directory: str, path to where to save neural-network to
        @param gamma: float, Policy Gradient parameter (discount factor)
        @param lr: float, learning rate of neural-network
        """
        self.gamma = gamma
        self.input_shape = input_shape
        self.save_directory = save_directory
        self.action_space = [.95, .98, .99, 1.0, 1.01, 1.02, 1.05]
        self.n_actions = len(self.action_space)
        self.policy = self.__build_conv_video_model(lr)
        self.normalize = normalize
        self.act_every = act_every

        # create directory to save models
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        self.states, self.actions, self.rewards = clean_memory()

    def act(self, obs):
        """
        chose action influencing the respawn rate of the apples
        :param obs: np.array, current observation
        :return: action index: int, indicate which action to take according to agent prediction
        :return: action: float, the real of action to perform

        """
        obs_ = rgb2gray(obs.copy())
        probs = self.policy.call(inputs=tf.convert_to_tensor([obs_])).numpy().flatten()

        action = np.random.choice(self.n_actions, p=probs)
        return action, self.action_space[action]

    def store(self, state, action, reward):
        """
        store all the relevant information to memory
        @param state:  np.array, current state that came from environment
        @param action: int, current action
        @param reward: float, current reward
        """
        self.states.append(rgb2gray(state.copy()))
        self.actions.append(action)
        self.rewards.append(reward)

    def pop(self):
        """
        retrieve all data in memory and empy content
        :return tuple, (states, actions, discounted_rewards):
        """
        states = np.array(self.states)
        actions = np.array(self.actions)
        discounted_rewards = self.discounted_reward(np.array(self.rewards))

        self.states = []
        self.actions = []
        self.rewards = []

        return states, actions, discounted_rewards

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
            if cnt > self.act_every:
                running_add = 0
                cnt = 0
            running_add = running_add * self.gamma + r[t]
            discounted_r[t] = running_add
            cnt += 1
        if sum(discounted_r) and self.normalize is True:
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
        states, actions, discounted_rewards = self.pop()
        loss = self.policy.train_on_batch(states, actions, sample_weight=discounted_rewards)
        return loss

    def save(self):
        """
        save the current agent neural-networks to files
        """
        self.policy.save(os.path.join(self.save_directory, "meta.h5"))

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

        q_net.add(Dense(128, activation="relu"))
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
        model.add(Conv2D(16, (3, 3), input_shape=self.input_shape[1:], padding='same', activation='relu'))
        model.add(Conv2D(16, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(32, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(Conv2D(64, (3, 3), padding='same', activation='relu'))
        model.add(MaxPool2D())

        # flatten...
        model.add(GlobalMaxPool2D())
        return model


class DQNAgent:
    instances = 0

    def __init__(self, input_shape, n_actions, save_directory, gamma=0.995, lr=0.001, min_epsilon_value=0.15,
                 batch_size=512, buffer_size=int(1e6), epsilon_decay_rate=0.985, update_every=50, min_to_learn=50):
        """
        @param input_shape: tuple, dimensions of states in the environment
        @param n_actions: int, amount of possible actions
        @param save_directory: str, path to where to save neural-network to
        @param gamma: float, DQN parameter (usually in range of 0.95-1)
        @param lr: float, learning rate of neural-network
        @param min_epsilon_value: float, min value of epsilon
        @param batch_size: int, batch size of model.fit
        @param buffer_size: int, size of replay buffer
        """

        self.n_actions = n_actions
        self.gamma = gamma
        self.input_shape = input_shape
        self.save_directory = save_directory
        self.epsilon = 1
        self.epsilon_decay_rate = epsilon_decay_rate
        self.min_epsilon_value = min_epsilon_value
        self.min_to_learn = min_to_learn
        self.batch_siz = batch_size
        self.buffer_size = buffer_size
        self.update_every = update_every

        self.optimizer = keras.optimizers.Adam(learning_rate=lr, epsilon=1e-6)

        self.q_predict, self.q_target = self.__build_dense_model(lr)

        # create directory to save models
        if not os.path.isdir(save_directory):
            os.mkdir(save_directory)

        self.memory = QAgentBuffer(buffer_size, input_shape, n_actions, discrete=False)

    def get_action(self, obs):
        """
            chose action according to epsilon greedy
        @param obs: np.array, state that came from environment
        @return: int, indicate which action to take according to agent prediction
        """
        if obs.sum() == 0:
            return np.random.choice(a=self.n_actions)

        q_values = self.q_predict.call(tf.convert_to_tensor(np.array([obs]))).numpy().flatten()
        q_norm = (q_values - q_values.min())

        p = q_norm/q_norm.sum()
        return np.random.choice(a=self.n_actions, p=p)

    def get_action_epsilon(self, obs):
        """
            chose action according to epsilon greedy
        @param obs: np.array, state that came from environment
        @return: int, indicate which action to take according to agent prediction
        """

        if obs.sum() == 0:
            return np.random.choice(a=self.n_actions)

        q_values = self.q_predict.call(tf.convert_to_tensor(np.array([obs]))).numpy().flatten()
        q_norm = (q_values - q_values.min())

        if random.random() < self.epsilon:
            p = q_norm/q_norm.sum()
            return np.random.choice(self.n_actions, p=p)

        return q_values.argmax()

    def store(self, state, next_state, action, reward, done):
        """
            store all the relevant information to memory
        @param state:  np.array, current state that came from environment
        @param next_state:  np.array, next state according to agent action the chosen
        @param action: int, the action that the agent chose from the current state that lead to next_State
        @param reward: float, the immediate reward that came from the environment according to current state and action the agent took
        @param done: bool, indicate if the episode is finish
        """
        self.memory.store(state, next_state, action, reward, done)

    def fit(self, ep):
        """
            implementation of DQN fit method
        @param ep: int, current episode in learning phase
        @return: float, the loss value according to keras model.fit
        """
        # get data from agent memory

        update_target = (ep % self.update_every == 0)

        states, next_states, actions, rewards, done = self.memory.sample_buffer(self.batch_siz)

        q_eval = self.q_predict.call(tf.convert_to_tensor(states)).numpy()
        q_next = self.q_target.call(tf.convert_to_tensor(next_states)).numpy()

        q_target = q_eval.copy()

        batch_index = np.arange(self.batch_siz, dtype=np.int8)

        future_reward = np.max(q_next, axis=1)
        q_target[batch_index, actions] = rewards + self.gamma*future_reward

        history = self.q_predict.fit(states, q_target, verbose=0, batch_size=self.batch_siz)
        if update_target:
            self.q_target.set_weights(self.q_predict.get_weights())

        max_predicted_reward = q_eval.max(axis=1).mean()
        self.epsilon = self.epsilon*self.epsilon_decay_rate if self.epsilon >= self.min_epsilon_value else self.epsilon
        return history.history['loss'][0], max_predicted_reward

    def save(self):
        """
            save the current agent neural-networks to files
        """
        q_file_name = os.path.join(self.save_directory, "q_net.h5")
        target_file_name = os.path.join(self.save_directory, "q_net_target.h5")
        self.q_predict.save(q_file_name)
        self.q_target.save(target_file_name)

    def load(self, path):
        """
            load agent neural-network from files
        @param path: str, path to where to find the h5 file to load
        """
        # file names
        q_net_file = os.path.join(path, "q_net.h5")
        q_net_target_file = os.path.join(path, "q_net_target.h5")

        # load models
        self.q_predict = load_model(q_net_file)
        self.q_target = load_model(q_net_target_file)

    def get_architecture_code(self):
        return inspect.getsource(self.__build_dense_model)

    def __build_conv_model(self, lr):
        """
        build keras.Sequential conv model
        @param lr: float, learning rate
        @return: keras.Sequential model
        """
        _input = Input(shape=self.input_shape)
        normalize = Lambda(lambda x: x/255)(_input)
        conv1 = Conv2D(filters=128, kernel_size=5, strides=1, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.01), padding='same', use_bias=True)(normalize)
        conv2 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding="same",
                       kernel_initializer=initializers.RandomNormal(stddev=0.01),  use_bias=True)(conv1)
        conv3 = Conv2D(filters=32, kernel_size=3, strides=1, activation='relu', padding="same",
                       kernel_initializer=initializers.RandomNormal(stddev=0.01),  use_bias=True)(conv1)
        mp = MaxPool2D(pool_size=(2, 2), strides=(2, 2))(conv3)
        flatten = Flatten()(mp)
        dense1 = Dense(128, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.01), use_bias=True)(flatten)
        dense2 = Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.01), use_bias=True)(dense1)
        q = Dense(self.n_actions, activation='linear',
                  kernel_initializer=initializers.RandomNormal(stddev=0.01), use_bias=True)(dense2)

        q_predict = keras.Model(inputs=_input, outputs=q)
        q_predict.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')
        q_target = keras.models.clone_model(q_predict)

        q_predict.trainable = True
        q_target.trainable = True
        return q_predict, q_target

    def __build_dense_model(self, lr):
        """
        build keras.Sequential conv model
        @param lr: float, learning rate
        @return: keras.Sequential model
        """
        _input = Input(shape=self.input_shape)
        normalize = Lambda(lambda x: x/255)(_input)

        flatten = Flatten()(normalize)
        dense1 = Dense(32, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.02), use_bias=True)(flatten)
        dense2 = Dense(64, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.02), use_bias=True)(dense1)
        dense3 = Dense(32, activation='relu',
                       kernel_initializer=initializers.RandomNormal(stddev=0.02), use_bias=True)(dense2)
        q = Dense(self.n_actions, activation='linear',
                  kernel_initializer=initializers.RandomNormal(stddev=0.02), use_bias=True)(dense3)

        # predict network
        q_predict = keras.Model(inputs=_input, outputs=q)
        q_predict.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')

        # target network
        q_target = keras.models.clone_model(q_predict)
        q_target.compile(optimizer=keras.optimizers.Adam(learning_rate=lr), loss='mse')

        q_predict.trainable = True
        q_target.trainable = True
        return q_predict, q_target
