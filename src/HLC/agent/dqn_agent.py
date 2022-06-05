import os
import random
import numpy as np
import tensorflow as tf
from HLC.utils.memory import ReplayMemory
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.models import load_model
from HLC.networks.dqn_network import DQNNetwork


class Agent:
    """
    Class for DQN model architecture.
    """
    def __init__(self, input_shape, num_actions, minibatch_size=32, agent_history_length=4, capacity=10000, lr=1e-4,
                 replay_start_size=10000, agent_directory="", discount_factor=.99):

        self.discount_factor = discount_factor
        self.minibatch_size = minibatch_size
        self.discount_factor = discount_factor
        self.agent_history_length = agent_history_length
        self.num_actions = num_actions
        self.agent_directory = agent_directory

        # memory
        self.memory = ReplayMemory(capacity=capacity, minibatch_size=self.minibatch_size, verbose=False)

        # agent networks
        self.main_network = DQNNetwork(input_shape, num_actions=num_actions,
                                       agent_history_length=self.agent_history_length)
        self.target_network = DQNNetwork(input_shape, num_actions=num_actions,
                                         agent_history_length=self.agent_history_length)

        self.update_target_network()
        self.optimizer = Adam(learning_rate=lr, epsilon=1e-6)

        self.replay_start_size = replay_start_size
        self.loss = tf.keras.losses.Huber()

        self.loss_metric = tf.keras.metrics.Mean(name="loss")
        self.q_metric = tf.keras.metrics.Mean(name="Q_value")

        # create directory to save agent weights
        if not os.path.isdir(self.agent_directory):
            os.mkdir(self.agent_directory)

    def get_action(self, state, exploration_rate):
        """Get action by ε-greedy method.

        Args:
            state (np.uint8): recent self.agent_history_length frames. (Default: (84, 84, 4))
            exploration_rate (int): Exploration rate for deciding random or optimal action.

        Returns:
            action (tf.int32): Action index
        """
        if random.random() < exploration_rate:
            action = np.random.choice(self.num_actions)
        else:
            recent_state = tf.expand_dims(state, axis=0)
            q_value = self.main_network(tf.cast(recent_state, tf.float32)).numpy()
            action = q_value.argmax()
        return action

    # @tf.function
    def update_main_q_network(self):
        """Update main q network by experience replay method.
        Returns:
            loss (tf.float32): Huber loss of temporal difference.
        """
        indices = self.memory.get_minibatch_indices()
        states, actions, rewards, next_states, terminal = self.memory.generate_minibatch_samples(indices)
        with tf.GradientTape() as tape:
            next_state_q = self.target_network(next_states)
            next_state_max_q = tf.math.reduce_max(next_state_q, axis=1)
            expected_q = rewards + self.discount_factor * next_state_max_q * (1.0 - tf.cast(terminal, tf.float32))
            main_q = tf.reduce_sum(self.main_network(states) * tf.one_hot(actions, self.num_actions, 1.0, 0.0), axis=1)
            loss = self.loss(tf.stop_gradient(expected_q), main_q)

        gradients = tape.gradient(loss, self.main_network.trainable_variables)
        clipped_gradients = [tf.clip_by_norm(grad, 10) for grad in gradients]
        self.optimizer.apply_gradients(zip(clipped_gradients, self.main_network.trainable_variables))

        self.loss_metric.update_state(loss)
        self.q_metric.update_state(main_q)

        # TODO: adding certainty, next_state_q_max param as an output,
        q_norm = tf.transpose(tf.transpose(next_state_q) - tf.math.reduce_min(next_state_q, axis=1))
        certainty = tf.reduce_max(q_norm,  axis=1) / tf.clip_by_value(tf.reduce_sum(q_norm, axis=1), 1e-8, 1e8)
        return loss, tf.math.reduce_mean(main_q), tf.reduce_mean(certainty)

    def update_target_network(self):
        """Synchronize weights of target network by those of main network."""
        
        main_vars = self.main_network.trainable_variables
        target_vars = self.target_network.trainable_variables
        for main_var, target_var in zip(main_vars, target_vars):
            target_var.assign(main_var)

    def remember(self, observation, action, reward, observation_next, done):
        self.memory.push(observation, action, reward, observation_next, done)

    def save_weights(self, ep):
        self.main_network.save_weights(os.path.join(self.agent_directory, f"episode_{ep}", ""))

    def load_weights(self, path):
        self.main_network = load_model(path)
        self.main_network = tf.keras.models.clone_model(self.main_network)

