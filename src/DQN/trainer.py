import os
import PIL
import numpy as np
from PIL import Image
import tensorflow as tf
from collections import deque
from timeit import default_timer as timer

# local imports
from .singleAgents import DQNAgent, DDQNAgent, HCDDQN
from .metrics import get_metrics
from .utils import log_configuration, save_frames_as_gif, obs_to_grayscale


class TrainerDQNAgent:
    def __init__(self, input_shape, num_actions, lr, ep_length, n_players, gamma, max_episodes=5000, batch_size=512,
                 render_every=np.inf, save_every=5, models_directory="models", log_dir=r"logs/", min_to_learn=50,
                 buffer_size=int(1e6), epsilon_decay_rate=0.99, update_every=50, min_epsilon=.1, gifs="",
                 save_gif_every=100):
        """
        :param input_shape: tuple, dimensions of observation in the environment
        :param num_actions: int, amount of possible actions
        :param gamma: float, discount factor parameter (usually in range of 0.95-1)
        :param lr: float, learning rate of neural-network
        :param n_players: int, amount of agents
        :param max_episodes: int, maximum episodes to train on
        :param render_every: int, in case that you want to render episodes during learning otherwise won't render
        :param ep_length: int, length of each episode
        :param save_every: int, frequency of saving the neural-network of each agent to files
        :param models_directory: str, directory to save the models to
        :param log_dir: str, directory name to save logs
        """

        # init class params
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_players = n_players
        self.max_episodes = max_episodes
        self.render_every = render_every
        self.ep_length = ep_length
        self.min_to_learn = min_to_learn
        self.lr = lr
        self.seq_len = input_shape[0]
        self.save_every = save_every
        self.models_directory = models_directory
        self.log_dir = log_dir
        self.writer = None
        self.gifs = gifs
        self.save_gif_every = save_gif_every
        self.learn_every = 100
        self.buffer_size = buffer_size
        self.eps = 1
        self.eps_decay_rate = epsilon_decay_rate
        self.min_eps = min_epsilon

        if not os.path.isdir(models_directory):
            os.mkdir(models_directory)
        # simple agents
        self.agents = [DDQNAgent(input_shape, num_actions, batch_size=batch_size,
                                 lr=lr, gamma=gamma, save_directory=os.path.join(models_directory, f"DDQAgent_{i}"),
                                 update_every=update_every, buffer_size=buffer_size, min_to_learn=min_to_learn)
                       for i in range(n_players)]

        self.agents[0].q_predict.summary()

    def setup_results_directories(self):
        """
        make sure existing or create directories for tensboard logs and keras models
        :param models_directory:
        :return: tf.writer for handling tensorboard
        """

        # add model architecture to models file
        log_configuration(self)

        # callbacks dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # log dire
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

    def choose_actions(self, n_observations):
        """
        get observations list and call each-one of the agent.get_action method with the
        corresponding observation
        :param n_observations:
        :return: dictionary of <agent.id, agent action>
        """
        key = "agent-%d"
        return {key % i: agent.get_action(n_observations[key % i]) for i, agent in enumerate(self.agents)}

    def choose_actions_epsilon(self, n_observations):
        key = "agent-%d"
        return {key % i: agent.get_action_epsilon(n_observations[key % i], self.eps)
                for i, agent in enumerate(self.agents)}

    def train(self, env, ep0=1):
        """
        train MultiDQNAgent environment with Policy Gradient meta agent
         and Policy Gradient agents with given environment
        :param env: gym api environment instance
        :param ep0: int, indicate how many iterations the agents has already trained
        """
        results = None

        # tf summary writer
        with self.writer.as_default() as writer:
            # train loop
            for ep in range(1, self.max_episodes):

                # init parm for each new episode
                total_reward = np.zeros(shape=self.n_players)
                n_observations = env.reset()
                time_reward_collected = []
                sleepers_time = []
                fire_tracking = []
                frames = []
                start = timer()
                certainty = 0

                # ep loop
                for t in range(ep0 + 1, ep0 + self.ep_length + 1):
                    # render
                    if ep % self.render_every == 0 and ep > 0:
                        title = f"ep: {ep}, frame: {t}, score: {total_reward}"
                        env.render(title=title)

                    # acting in the environment
                    actions = self.choose_actions_epsilon(n_observations)

                    # make actions
                    next_n_observations, n_rewards, n_done, n_info = env.step(actions)

                    # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])
                    fire_tracking.extend(
                        [(env.get_current_hits(), (np.fromiter(actions.values(), dtype=float) == 7).sum())])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        if n_observations[index].sum() != 0:
                            self.agents[i].store(state=n_observations[index], next_state=next_n_observations[index],
                                                 action=actions[index],
                                                 reward=n_rewards[index],
                                                 done=n_done[index])

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
                    n_observations = next_n_observations

                    # save frames of this episode
                    if ep % self.save_gif_every == 0:
                        frames.append(Image.fromarray(
                            np.uint8(env.get_full_state())).resize(
                            size=(400, 180), resample=PIL.Image.BOX).convert("RGB"))

                    if ep > self.min_to_learn and t % self.learn_every == 0:
                        results = self.fit(ep)
                        certainty = results["certainty"]

                # end of episode
                # fit agents
                if ep > self.min_to_learn:
                    # update agents epsilon values
                    self.eps = self.eps * self.eps_decay_rate if self.eps >= self.min_eps else self.eps
                    # display results
                    fire_tracking = np.array(list(zip(*fire_tracking)), dtype=int).T.sum(axis=0)
                    self.tensorboard(total_reward=total_reward, ep=ep - self.min_to_learn, results=results,
                                     writer=writer,
                                     epsilon=self.eps,
                                     fire_efficiency=fire_tracking[0] / fire_tracking[1],
                                     time_reward_collected=time_reward_collected, sleepers_time=sleepers_time)

                # save agents models to files
                if ep > self.save_every:
                    self.save()

                # save episode to gif
                if ep % self.save_gif_every == 0:
                    save_frames_as_gif(frames=frames, path=self.gifs, filename=f"episode_{ep}.gif")

                # log
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {timer() - start:.2f},"
                      f" epsilon: {self.eps:.2e}, certainty: {int(certainty * 100)}%")

    def tensorboard(self, total_reward, ep, results, epsilon, writer, time_reward_collected, sleepers_time,
                    fire_efficiency):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param writer: tf writer
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param fire_efficiency: float, percentage of successful fire action
        :param epsilon, current exploration rate of DQN agents
        :param results: dictionary, results of fit method (agents and meta_agent loss and q_values)
        :param ep: int, indicate episode number
        :param env: environment instance
        """
        # metrics params
        eq, sus, p, ef = get_metrics(total_reward, self.n_players, self.ep_length, sleepers_time, time_reward_collected)
        rewards = total_reward.sum()
        learned_ep = ep

        # display fit results
        for k, v in results.items():
            tf.summary.scalar(name=k, data=v, step=learned_ep)

        # display metrics
        tf.summary.scalar(name="total reward", data=rewards, step=learned_ep)
        tf.summary.scalar(name="epsilon", data=epsilon, step=learned_ep)
        tf.summary.scalar(name="efficiency", data=ef, step=learned_ep)
        tf.summary.scalar(name="sustainability", data=sus, step=learned_ep)
        tf.summary.scalar(name="equality", data=eq, step=learned_ep)
        tf.summary.scalar(name="peace", data=p, step=learned_ep)
        tf.summary.scalar(name="fire efficiency", data=fire_efficiency, step=learned_ep),

        # write data to log file
        writer.flush()

    def fit(self, ep):
        """
        call agent.fit for all the agents including meta agent
        :param ep: int, current iteration in training
        :return: dictionary including all results from agent fit (losses...)
        """

        # fit agents
        loss, max_predicted_reward, certainty = list(zip(*[agent.fit(ep) for agent in self.agents]))
        return {"loss": np.mean(loss),
                "max_predicted_reward": np.mean(max_predicted_reward),
                "certainty": np.mean(certainty)}

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


class HCDWNTrainer:
    def __init__(self, input_shape, num_actions, ep_length, n_players, models_directory, log_dir, gamma=.99,
                 max_episodes=int(1e4), batch_size=64, render_every=100, save_every=100, min_to_learn=15000,
                 buffer_size=int(1e5), epsilon_decay_rate=0.999, update_every=200, min_epsilon=.1, gifs="",
                 save_gif_every=100, history_length=4, lr=0.00001, learn_every=100):
        """
        :param input_shape: tuple, dimensions of observation in the environment
        :param num_actions: int, amount of possible actions
        :param gamma: float, discount factor parameter (usually in range of 0.95-1)
        :param lr: float, learning rate of neural-network
        :param n_players: int, amount of agents
        :param max_episodes: int, maximum episodes to train on
        :param render_every: int, in case that you want to render episodes during learning otherwise won't render
        :param ep_length: int, length of each episode
        :param save_every: int, frequency of saving the neural-network of each agent to files
        :param models_directory: str, directory to save the models to
        :param log_dir: str, directory name to save logs
        """

        # init env params
        self.num_actions = num_actions
        self.gamma = gamma
        self.n_players = n_players
        self.max_episodes = max_episodes
        self.render_every = render_every
        self.ep_length = ep_length

        # init learning params
        self.min_to_learn = min_to_learn
        self.input_shape = input_shape
        self.lr = lr
        self.learn_every = learn_every
        self.buffer_size = buffer_size
        self.eps = 1
        self.eps_decay_rate = epsilon_decay_rate
        self.min_eps = min_epsilon
        self.history_length = history_length
        self.save_every = save_every
        self.batch_size = batch_size
        self.update_every = update_every

        # init outputs params
        self.models_directory = models_directory
        self.log_dir = log_dir
        self.gifs_dir = gifs
        self.writer = None
        self.save_gif_every = save_gif_every

        # init agents
        self.agents = None

    def setup_results_directories_with_tensorboard(self):
        """
        make sure existing or create directories for TensorBoard logs and keras models
        :return: tf.writer for handling tensorboard
        """

        # create directory to save agents
        if not os.path.isdir(self.models_directory):
            os.mkdir(self.models_directory)
        for i in range(self.n_players):
            dir_name = os.path.join(self.models_directory, f"DDQAgent_{i}")
            if not os.path.isdir(dir_name):
                os.mkdir(dir_name)

        # create directory for TensorBoard logs
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # empty previous content of log dir
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        self.writer = tf.summary.create_file_writer(logdir=self.log_dir)

    def init_agents(self):
        self.agents = [HCDDQN(self.input_shape, self.num_actions, batch_size=self.batch_size, lr=self.lr,
                              gamma=self.gamma, save_directory=os.path.join(self.models_directory, f"DDQAgent_{i}"),
                              experience_replay_size=self.buffer_size)
                       for i in range(self.n_players)]
        self.agents[0].summary()

    def choose_actions_eps_greedy(self, n_observations: dict):
        """
        get observations list and call each-one of the agent.get_action method with the
        corresponding observation
        :param n_observations:
        :return: dictionary of <agent.id, agent action>
        """
        key = "agent-%d"
        return {key % i: agent.choose_action_eps_greedy(n_observations[key % i].transpose(2, 1, 0), self.eps)
                for i, agent in enumerate(self.agents)}

    def choose_actions_softmax(self, n_observations: dict):
        key = "agent-%d"
        return {key % i: agent.choose_action_softmax(n_observations[key % i])
                for i, agent in enumerate(self.agents)}

    def train(self, env, ep0=0):
        """
        train MultiDQNAgent environment with Policy Gradient meta agent
        and Policy Gradient agents with given environment
        :param env: gym api environment instance
        :param ep0: int, indicate how many iterations the agents has already trained
        """
        results = None
        total_frames = 0
        agents_updates = 0
        # tf summary writer
        with self.writer.as_default() as writer:
            # train loop
            for ep in range(ep0 + 1, ep0 + self.max_episodes + 1):

                # init parm for each new episode
                total_reward = np.zeros(shape=self.n_players)
                n_observations = env.reset()
                n_observations_prepro = {agent: np.zeros(self.input_shape[::-1])
                                         for agent, obs in n_observations.items()}
                for agent, obs in n_observations.items():
                    n_observations_prepro[agent][-1, :, :] = obs_to_grayscale(obs)
                n_observations_prepro_next = n_observations_prepro.copy()

                time_reward_collected, sleepers_time, fire_tracking, frames = [], [], [], []
                start = timer()
                certainty = 0

                # ep loop
                for t in range(1, self.ep_length+1):
                    # render
                    if ep % self.render_every == 0 and ep > 0:
                        title = f"ep: {ep}, frame: {t}, score: {total_reward}"
                        env.render(title=title)

                    # acting in the environment
                    actions = self.choose_actions_eps_greedy(n_observations_prepro)

                    # make actions
                    next_n_observations, n_rewards, n_done, n_info = env.step(actions)
                    for agent, obs in next_n_observations.items():
                        n_observations_prepro_next[agent] = np.roll(n_observations_prepro_next[agent], 1, 0)
                        n_observations_prepro_next[agent][0, :, :] = obs_to_grayscale(obs)

                    # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])
                    fire_tracking.extend(
                        [(env.get_current_hits(), (np.fromiter(actions.values(), dtype=float) == 7).sum())])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        self.agents[i].store(state=n_observations_prepro[index].transpose(2, 1, 0),
                                             next_state=n_observations_prepro_next[index].transpose(2, 1, 0),
                                             action=actions[index],
                                             reward=n_rewards[index],
                                             done=n_done[index])

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)

                    # update observations
                    n_observations_prepro = n_observations_prepro_next
                    n_observations = next_n_observations

                    # save frames of this episode
                    if ep % self.save_gif_every == 0:
                        frames.append(Image.fromarray(
                            np.uint8(env.get_full_state())).resize(
                            size=(400, 180), resample=PIL.Image.BOX).convert("RGB"))

                    if total_frames > self.min_to_learn and total_frames % self.learn_every == 0:
                        results = self.fit(agents_updates % self.update_every == 0)
                        certainty = results["certainty"]
                        agents_updates += 1

                    total_frames += 1

            # end of episode
                # fit agents
                if total_frames > self.min_to_learn:
                    # update agents epsilon values
                    self.eps = self.eps * self.eps_decay_rate if self.eps >= self.min_eps else self.eps

                    # display results on TensorBoard
                    fire_tracking = np.array(list(zip(*fire_tracking)), dtype=int).T.sum(axis=0)
                    self.tensorboard(total_reward=total_reward, ep=ep, results=results,
                                     writer=writer,
                                     epsilon=self.eps,
                                     fire_efficiency=fire_tracking[0] / fire_tracking[1],
                                     time_reward_collected=time_reward_collected, sleepers_time=sleepers_time)

                # save agents models to files
                if ep > self.save_every:
                    self.save()

                # save episode to gif
                if ep % self.save_gif_every == 0:
                    save_frames_as_gif(frames=frames, path=self.gifs_dir,
                                       filename=f"episode_{ep}_score_{total_reward.sum()}.gif")

                # print train progression
                print(f"# {ep}, frames: {total_frames}, fit: {agents_updates},  total_rewards: {total_reward.sum()},"
                      f" time: {timer() - start:.2f}, epsilon: {self.eps:.2e}, certainty: {int(certainty * 100)}%")

    def tensorboard(self, total_reward, ep, results, epsilon, writer, time_reward_collected, sleepers_time,
                    fire_efficiency):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param writer: tf writer
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param fire_efficiency: float, percentage of successful fire action
        :param epsilon, current exploration rate of DQN agents
        :param results: dictionary, results of fit method (agents and meta_agent loss and q_values)
        :param ep: int, indicate episode number
        :param env: environment instance
        """
        # metrics params
        eq, sus, p, ef = get_metrics(total_reward, self.n_players, self.ep_length, sleepers_time, time_reward_collected)
        rewards = total_reward.sum()
        learned_ep = ep

        # display fit results
        for k, v in results.items():
            tf.summary.scalar(name=k, data=v, step=learned_ep)

        # display metrics
        tf.summary.scalar(name="total reward", data=rewards, step=learned_ep)
        tf.summary.scalar(name="epsilon", data=epsilon, step=learned_ep)
        tf.summary.scalar(name="efficiency", data=ef, step=learned_ep)
        tf.summary.scalar(name="sustainability", data=sus, step=learned_ep)
        tf.summary.scalar(name="equality", data=eq, step=learned_ep)
        tf.summary.scalar(name="peace", data=p, step=learned_ep)
        tf.summary.scalar(name="fire efficiency", data=fire_efficiency, step=learned_ep),

        # write data to log file
        writer.flush()

    def fit(self, ep):
        """
        call agent.fit for all the agents including meta agent
        :param ep: int, current iteration in training
        :return: dictionary including all results from agent fit (losses...)
        """

        # fit agents
        loss, max_predicted_reward, certainty = list(zip(*[agent.fit(ep) for agent in self.agents]))
        return {"loss": np.mean(loss),
                "max_predicted_reward": np.mean(max_predicted_reward),
                "certainty": np.mean(certainty)}

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


class TrainerDQNAgentFromPrevTraining(TrainerDQNAgent):
    def __init__(self, input_shape, num_actions, lr, ep_length, n_players, gamma, max_episodes=5000, batch_size=512,
                 render_every=np.inf, save_every=5, models_directory="models", log_dir=r"logs/", min_to_learn=50,
                 buffer_size=int(1e6), epsilon_decay_rate=0.99, update_every=50, min_epsilon=.1, writer=None, gifs="",
                 save_gif_every=100, eps0=1):
        """
        :param input_shape: tuple, dimensions of observation in the environment
        :param num_actions: int, amount of possible actions
        :param gamma: float, discount factor parameter (usually in range of 0.95-1)
        :param lr: float, learning rate of neural-network
        :param n_players: int, amount of agents
        :param max_episodes: int, maximum episodes to train on
        :param render_every: int, in case that you want to render episodes during learning otherwise won't render
        :param ep_length: int, length of each episode
        :param save_every: int, frequency of saving the neural-network of each agent to files
        :param models_directory: str, directory to save the models to
        :param log_dir: str, directory name to save logs
        """

        # init class params
        super().__init__(input_shape, num_actions, lr, ep_length, n_players, gamma, max_episodes, batch_size,
                         render_every, save_every, models_directory, log_dir, min_to_learn, buffer_size,
                         epsilon_decay_rate, update_every, min_epsilon, gifs, save_gif_every)
        self.writer = writer
        self.eps = eps0

        # load agents
        [agent.load(os.path.join(models_directory, f"DDQAgent_{i}")) for i, agent in enumerate(self.agents)]

    def setup_results_directories(self):
        """
        make sure existing or create directories for tensboard logs and keras models
        :param models_directory:
        :return: tf.writer for handling tensorboard
        """

        # create directory to save models
        if not os.path.isdir(self.models_directory):
            os.mkdir(self.models_directory)
