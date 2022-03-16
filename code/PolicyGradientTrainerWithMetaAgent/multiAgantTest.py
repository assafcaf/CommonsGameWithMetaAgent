import os
import random
import numpy as np
import tensorflow as tf
from tensorflow.keras.models import load_model
import sys
import time

# local imports
sys.path.append(os.path.join(os.getcwd()))
from src.agents.DQAgent.metrics import sustainability, equality, peace, efficiency
from src.agents.DQAgent.singleAgent import DQNAgent


class MultiDQNAgent:
    def __init__(self, input_shape, num_actions, gamma=0.99, lr=0.00075, update_every=50, n_players=1,
                 max_episodes=50000, render_every=np.inf, ep_length=150, loss="mse",  save_every=100,
                 models_directory="models", learn_every=1, greedy=False, batch_size=256, buffer_size=1e6,
                 min_to_learn=100, log_dir=r"logs/"):
        """
        @param input_shape: tuple, dimensions of states in the environment
        @param num_actions: int, amount of possible actions
        @param gamma: float, DQN parameter (usually in range of 0.95-1)
        @param lr: float, learning rate of neural-network
        @param update_every: int, specified the frequency of updating the target network
        @param n_players: int, amount of agents
        @param max_episodes: int
        @param render_every: int, in case that you want to render episodes during learning
        @param ep_length: int, length of each episode
        @param loss: str/ tk.keras.losses.<loss> , specified the loss function of the neural-network ("mse", "mae", "tf.keras.losses.Huber()")
        @param save_every: int, frequency of saving the neural-network of each agent to files
        @param models_directory: str, directory to save the models to
        @param learn_every: int, frequency of calling model.fit for each agent
        @param greedy: bool, in case of running the code for rendering
        @param batch_size: int, batch size of model.fit
        @param buffer_size: int, size of replay buffer
        @param min_to_learn: int, amount of episodes before calling model.fit
        @param log_dir: str, directory name to save logs
        """
        # init class params
        self.num_actions = num_actions
        self.gamma = gamma
        self.epsilon = 1
        self.n_players = n_players
        self.max_episodes = max_episodes
        self.render_every = render_every
        self.ep_length = ep_length
        self.update_every = update_every
        self.save_every = save_every
        self.models_directory = models_directory
        self.learn_every = learn_every
        self.greedy = greedy
        self.batch_siz = batch_size
        self.buffer_size = buffer_size
        self.min_to_learn = min_to_learn
        self.log_dir = os.path.join(log_dir)

        # create directory to save models
        if not os.path.isdir(models_directory):
            os.mkdir(models_directory)

        # callbacks dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

        # agents
        self.agents = [DQNAgent(input_shape, num_actions, lr=lr, loss=loss, gamma=gamma, greedy=greedy,
                                batch_size=batch_size, buffer_size=buffer_size,
                                save_directory=os.path.join(models_directory, f"model_{i}"))
                       for i in range(n_players)]

    def choose_actions(self, n_observations):
        """

        @param n_observations: list, list of observations one for each agent
        @return: int, chosen action according to agents prediction
        """
        actions = np.zeros(self.n_players, dtype=np.float32)
        for i, agent in enumerate(self.agents):
            actions[i] = agent.get_action(n_observations[i])
        return actions

    def train(self, env):
        """
            train MultiDQNAgent DQN on a given environment
        @param env: gym environment instance
        """

        loss = None
        max_predicted_reward = None
        # train loop
        for ep in range(self.max_episodes):
            # init parm for each new episode
            total_reward = np.array([0] * self.n_players)
            n_observations = np.array(env.reset())
            time_reward_collected = []
            sleepers_time = []
            start = int(time.time())

            # ep loop
            for t in range(self.ep_length):

                # acting in the environment
                actions = self.choose_actions(n_observations)

                # make actions
                next_n_observations, n_rewards, n_done, n_info = env.step(actions)

                # collect behavior data for analysis
                time_reward_collected.extend([t for r in n_rewards if r != 0])
                sleepers_time.extend([1 for obs in n_observations if obs is None])

                # store data for learning
                for i in range(self.n_players):
                    self.agents[i].store(state=n_observations[i], next_state=next_n_observations[i],
                                         action=actions[i], reward=n_rewards[i], done=n_done[i])

                # collect rewards
                total_reward += np.array(n_rewards)
                n_observations = next_n_observations

            # end of episode
            if ep % self.learn_every == 0 and self.min_to_learn < ep:
                # call fit each self.learn_every episodes
                loss, max_predicted_reward = self.fit(ep)

                # update all agent epsilon according to MultiDQNAgent policy
                [agent.set_epsilon(self.epsilon) for agent in self.agents]

            # collect episodes reward
            total_reward.sum()

            # update epsilon
            self.epsilon = np.exp(-ep / 5000)

            # save neural-networks to files
            if ep % self.save_every == 0:
                self.save()

            # tensorboard
            self.tensorboard(total_reward, time_reward_collected, sleepers_time, loss, max_predicted_reward, ep)
            end = time.time()
            print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end-start}")
            start = end

    def tensorboard(self, total_reward, time_reward_collected, sleepers_time, loss, max_predicted_reward, ep):
        """
            create tensorboard graph to display the algorithm progress
        @param total_reward: list, sum reward of full episode for each of the agents
        @param time_reward_collected: list,the time on which the rewards have collected
        @param sleepers_time: list, indicate when agent where out of the game due to tagging
        @param loss: float, sum of losses from all the agents
        @param max_predicted_reward: float, current estimate of q_values
        @param ep: int, indicate episode number
        """
        with self.writer.as_default() as writer:
            if loss is not None:
                learned_ep = ep - self.min_to_learn
                tf.summary.scalar(name="total_reward", data=total_reward.sum(), step=learned_ep)
                tf.summary.scalar(name="efficiency", data=efficiency(total_reward, self.n_players, self.ep_length), step=learned_ep)
                tf.summary.scalar(name="loss", data=loss, step=learned_ep)
                tf.summary.scalar(name="sustainability", data=sustainability(time_reward_collected, self.ep_length), step=learned_ep)
                tf.summary.scalar(name="equality", data=equality(total_reward, self.n_players), step=learned_ep)
                tf.summary.scalar(name="peace", data=peace(sleepers_time, self.n_players, self.ep_length), step=learned_ep)
                tf.summary.scalar(name="epsilon", data=self.epsilon, step=learned_ep)
                tf.summary.scalar(name="current_q_values", data=max_predicted_reward, step=learned_ep)

                dqn_variable = self.agents[0].q_net.trainable_variables
                for i in range(len(dqn_variable)):
                    tf.summary.histogram(name=f"layer_{i}", data=tf.convert_to_tensor(dqn_variable[i]), step=ep)
                writer.flush()

    def fit(self, ep):
        """
            call agent.fit for all the agents
        @param ep: int, current iteration in training
        @return: float, sum of all agent losses
        """
        update_target = (ep % self.update_every == 0)
        values = [agent.fit(update_target) for agent in self.agents]
        loss, max_predicted_reward = zip(*values)
        return sum(loss), np.mean(max_predicted_reward)

    def save(self):
        """
            save all agents neural-network to files
        """
        for agent in self.agents:
            agent.save()

    def load(self, path):
        """
            load neural-networks from files for all the agents
        @param path: str, path to where neural-network models locate at
        """
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f"model_{i}"))


class MultiDQNAgentTest:
    def __init__(self, n_players=1, models_directory="models"):
        # init class params
        self.n_players = n_players
        self.models_directory = models_directory
        models_in_directory = len(os.listdir(models_directory))
        # agents
        self.agents = [load_model(os.path.join(models_directory, f"model_{i % models_in_directory}", "q_net.h5"))
                       for i in range(n_players)]

    def choose_actions(self, n_observations):
        actions = np.zeros(self.n_players, dtype=np.float32)
        action = None
        for i, agent in enumerate(self.agents):
            if n_observations[i] is not None:
                actions[i] = np.argmax(agent.predict(n_observations[i][np.newaxis, :])[0])
            else:
                actions[i] = action
        return actions

    def choose_actions_r(self, n_observations, pr):
        actions = np.zeros(self.n_players, dtype=np.float32)
        action = None
        for i, agent in enumerate(self.agents):
            if n_observations[i] is not None:
                if random.random() < pr:
                    actions[i] = np.random.choice(8, size=1, p=[1/8]*8)[0]
                else:
                    actions[i] = np.argmax(agent.predict(n_observations[i][np.newaxis, :])[0])
            else:
                actions[i] = action
        return actions
