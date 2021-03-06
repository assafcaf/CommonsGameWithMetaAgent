import os
import PIL
import numpy as np
from PIL import Image
import tensorflow as tf
from timeit import default_timer as timer
# local imports
from .singleAgents import PGAgent, PGMetaAgent, DQNAgent
from .metrics import get_metrics
from .utils import put_kernels_on_grid, log_configuration, save_frames_as_gif


class TrainerWithMetaAgent:
    def __init__(self, input_shape, state_dim, num_actions, lr=0.00075, n_players=1, max_episodes=50000,
                 render_every=np.inf, ep_length=150, save_every=100, models_directory="models", log_dir=r"logs/",
                 gamma=0.99, act_every=25):
        """
        :param state_dim: tuple, dimensions of full states in the environment
        :param act_every: int, interval for meta agent to act on (will act on each act_every turns)
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
        self.meta_state_shape = (act_every, *state_dim)
        self.max_episodes = max_episodes
        self.render_every = render_every
        self.ep_length = ep_length
        self.save_every = save_every
        self.act_every = act_every
        self.models_directory = models_directory
        self.log_dir = os.path.join(log_dir)
        self.writer = self.handle_callbacks(models_directory)

        # simple agents
        self.agents = [PGAgent(input_shape, num_actions, lr=lr, gamma=gamma, normalize=False,
                               save_directory=os.path.join(models_directory, f"PGAgent_{i}")) for i in range(n_players)]
        self.agents[0].policy.summary()

        # meta agent
        self.meta_agent = PGMetaAgent(input_shape=self.meta_state_shape, act_every=act_every, normalize=True,
                                      save_directory=os.path.join(models_directory, f"MetaAgent"))

        self.meta_agent.policy.summary()

    def handle_callbacks(self, models_directory):
        """
        make sure existing or create directories for tensboard logs and keras models
        :param models_directory:
        :return: tf.writer for handling tensorboard
        """

        # create directory to save models
        if not os.path.isdir(models_directory):
            os.mkdir(models_directory)

        # callbacks dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # log dire
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        return tf.summary.create_file_writer(logdir=self.log_dir)

    def choose_actions(self, n_observations):
        """
        get observations list and call each-one of the agent.get_action method with the
        corresponding observation
        :param n_observations:
        :return: dictionary of <agent.id, agent action>
        """
        key = "agent-%d"
        return {key % i: agent.get_action(n_observations[key % i]) for i, agent in enumerate(self.agents)}

    def train(self, env):
        """
        train MultiAgent environment with Policy Gradient meta agent
         and Policy Gradient agents to a given environment
        :param env: gym api environment instance
        """
        results = None

        # tf summary writer
        with self.writer.as_default() as writer:
            # train loop
            for ep in range(1, self.max_episodes):

                # init parm for each new episode
                total_reward = np.zeros(shape=self.n_players)
                meta_rewards = np.zeros(shape=self.n_players)
                meta_total_rewards = 0
                n_observations = env.reset()
                meta_state = env.get_full_state()
                time_reward_collected = []
                sleepers_time = []
                start = timer()
                agg_meta_state = np.zeros(shape=self.meta_state_shape[:-1] + (3,))
                meta_actions = []

                # ep loop
                for t in range(self.ep_length):
                    # render
                    if ep % self.render_every == 0 and ep > 0:
                        title = f"ep: {ep}, frame: {t}, score: {total_reward}"
                        env.render(title=title)

                    # aggregate full states for metaAgent
                    agg_meta_state[t % self.act_every, :] = meta_state

                    # acting in the environment
                    actions = self.choose_actions(n_observations)

                    # make actions
                    next_n_observations, n_rewards, n_done, n_info = env.step(actions)
                    meta_state = env.get_full_state()

                    # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        self.agents[i].store(state=n_observations[index], action=actions[index],
                                             reward=n_rewards[index])

                    # meta action and storing to memory
                    if t % self.act_every == 0 and t != 0:
                        action, multiplier = self.meta_agent.act(agg_meta_state)
                        env.update_response_time(multiplier)
                        r = meta_rewards.sum() * equality(meta_rewards, self.n_players)
                        self.meta_agent.store(state=agg_meta_state, action=action, reward=r)
                        meta_rewards = np.zeros(shape=self.n_players)
                        meta_total_rewards += r
                        meta_actions.append(multiplier)

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
                    meta_rewards += np.fromiter(n_rewards.values(), dtype=np.int)
                    n_observations = next_n_observations

                ## end of episode
                # fit agents
                results = self.fit(ep)

                # display results
                self.tensorboard(total_reward=total_reward, time_reward_collected=time_reward_collected,
                                 sleepers_time=sleepers_time, ep=ep, results=results,
                                 meta_total_rewards=meta_total_rewards,
                                 meta_actions=np.array(meta_actions), writer=writer)
                # log
                end = timer()
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end - start:.2f}")

    def tensorboard(self, total_reward, time_reward_collected, sleepers_time, ep, results,
                    meta_total_rewards, meta_actions, writer):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param meta_total_rewards: float, total rewards of meta agent
        :param meta_actions: list, all actions meta agent performed
        :param writer: tf.writer
        :param results: dictionary, results of fit method (agents and meta_agent loss and q_values)
        :param ep: int, indicate episode number
        """

        learned_ep = ep
        for k, v in results.items():
            tf.summary.scalar(name=k, data=v, step=learned_ep)

        # display metrics
        tf.summary.scalar(name="total reward", data=total_reward.sum(), step=learned_ep)

        tf.summary.scalar(name="meta total reward", data=meta_total_rewards, step=learned_ep)
        tf.summary.scalar(name="meta avg actions", data=meta_actions.mean(), step=learned_ep)

        tf.summary.scalar(name="efficiency", data=efficiency(total_reward, self.n_players, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="sustainability", data=sustainability(time_reward_collected, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="equality", data=equality(total_reward, self.n_players), step=learned_ep)
        tf.summary.scalar(name="peace", data=peace(sleepers_time, self.n_players, self.ep_length),
                          step=learned_ep)

        writer.flush()

    def fit(self, ep):
        """
        call agent.fit for all the agents including meta agent
        :param ep: int, current iteration in training
        :return: dictionary including all results from agent fit (losses...)
        """

        # fit agents
        agents_loss = [agent.fit() for agent in self.agents]

        # fit meta agent
        meta_loss = self.meta_agent.fit()

        results = {"agents_loss": sum(agents_loss),
                   "meta_loss": meta_loss}
        return results

    def save(self):
        """
            save all agents neural-network to files
        """
        for agent in self.agents:
            agent.save()
        self.meta_agent.save()

    def load(self, path):
        """
            load neural-networks from files for all the agents
        @param path: str, path to where neural-network models locate at
        """
        for i, agent in enumerate(self.agents):
            agent.load(os.path.join(path, f"model_{i}"))
        self.meta_agent.load("meta.h5")


class TrainerNoMetaAgent:
    def __init__(self, input_shape, num_actions, lr=0.00075, n_players=1, max_episodes=50000,
                 render_every=np.inf, ep_length=150, save_every=100, models_directory="models", log_dir=r"logs/",
                 gamma=0.99):
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
        self.save_every = save_every
        self.models_directory = models_directory
        self.log_dir = os.path.join(log_dir)
        self.writer = self.handle_callbacks(models_directory)

        # simple agents
        self.agents = [PGAgent(input_shape, num_actions, lr=lr, gamma=gamma, normalize=False,
                               save_directory=os.path.join(models_directory, f"PGAgent_{i}")) for i in range(n_players)]
        self.agents[0].policy.summary()

        log_configuration(self)

    def handle_callbacks(self, models_directory):
        """
        make sure existing or create directories for tensboard logs and keras models
        :param models_directory:
        :return: tf.writer for handling tensorboard
        """

        # create directory to save models
        if not os.path.isdir(models_directory):
            os.mkdir(models_directory)

        # callbacks dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # log dire
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        return tf.summary.create_file_writer(logdir=self.log_dir)

    def choose_actions(self, n_observations):
        """
        get observations list and call each-one of the agent.get_action method with the
        corresponding observation
        :param n_observations:
        :return: dictionary of <agent.id, agent action>
        """
        key = "agent-%d"
        return {key % i: agent.get_action(n_observations[key % i]) for i, agent in enumerate(self.agents)}

    def train(self, env):
        """
        train MultiDQNAgent environment with Policy Gradient meta agent
         and Policy Gradient agents with given environment
        :param env: gym api environment instance
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
                start = timer()

                # ep loop
                for t in range(self.ep_length):
                    # render
                    if ep % self.render_every == 0 and ep > 0:
                        title = f"ep: {ep}, frame: {t}, score: {total_reward}"
                        env.render(title=title)

                    # acting in the environment
                    actions = self.choose_actions(n_observations)

                    # make actions
                    next_n_observations, n_rewards, n_done, n_info = env.step(actions)

                    # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        self.agents[i].store(state=n_observations[index], action=actions[index],
                                             reward=n_rewards[index])

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
                    n_observations = next_n_observations

                # end of episode
                # fit agents
                results = self.fit(ep)

                # display results
                self.tensorboard(total_reward=total_reward, time_reward_collected=time_reward_collected,
                                 sleepers_time=sleepers_time, ep=ep, results=results, writer=writer)

                # log
                end = timer()
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end - start:.2f}")

    def tensorboard(self, total_reward, time_reward_collected, sleepers_time, ep, results, writer):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param writer: tf.writer
        :param results: dictionary, results of fit method (agents and meta_agent loss and q_values)
        :param ep: int, indicate episode number
        """

        learned_ep = ep
        for k, v in results.items():
            tf.summary.scalar(name=k, data=v, step=learned_ep)

        # display metrics
        tf.summary.scalar(name="total reward", data=total_reward.sum(), step=learned_ep)

        tf.summary.scalar(name="efficiency", data=efficiency(total_reward, self.n_players, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="sustainability", data=sustainability(time_reward_collected, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="equality", data=equality(total_reward, self.n_players), step=learned_ep)
        tf.summary.scalar(name="peace", data=peace(sleepers_time, self.n_players, self.ep_length),
                          step=learned_ep)

        writer.flush()

    def fit(self, ep):
        """
        call agent.fit for all the agents including meta agent
        :param ep: int, current iteration in training
        :return: dictionary including all results from agent fit (losses...)
        """

        # fit agents
        agents_loss = [agent.fit() for agent in self.agents]

        results = {"agents_loss": sum(agents_loss)}
        return results

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


class TrainerDQNNoMetaAgent:
    def __init__(self, input_shape, num_actions, lr, ep_length, n_players, gamma, max_episodes=3000, batch_size=512,
                 render_every=np.inf, save_every=5, models_directory="models", log_dir=r"logs/", min_to_learn=50,
                 buffer_size=int(1e6), epsilon_decay_rate=0.99, update_every=50, min_epsilon=.1, conv=False, gifs="",
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
        self.log_dir = os.path.join(log_dir)
        self.writer = self.handle_callbacks(models_directory)
        self.conv = conv
        self.gifs = gifs
        self.save_gif_every = save_gif_every
        # simple agents
        self.agents = [DQNAgent(input_shape, num_actions, epsilon_decay_rate=epsilon_decay_rate, batch_size=batch_size,
                                lr=lr, gamma=gamma, save_directory=os.path.join(models_directory, f"DQAgent_{i}"),
                                update_every=update_every, buffer_size=buffer_size, min_to_learn=min_to_learn,
                                min_epsilon_value=min_epsilon)
                       for i in range(n_players)]

        self.agents[0].q_predict.summary()
        log_configuration(self)

    def handle_callbacks(self, models_directory):
        """
        make sure existing or create directories for tensboard logs and keras models
        :param models_directory:
        :return: tf.writer for handling tensorboard
        """

        # create directory to save models
        if not os.path.isdir(models_directory):
            os.mkdir(models_directory)

        # callbacks dir
        if not os.path.isdir(self.log_dir):
            os.mkdir(self.log_dir)

        # log dire
        elif len(os.listdir(self.log_dir)) != 0:
            [os.remove(os.path.join(self.log_dir, f)) for f in os.listdir(self.log_dir)]
        return tf.summary.create_file_writer(logdir=self.log_dir)

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
        return {key % i: agent.get_action_epsilon(n_observations[key % i]) for i, agent in enumerate(self.agents)}

    def train(self, env):
        """
        train MultiDQNAgent environment with Policy Gradient meta agent
         and Policy Gradient agents with given environment
        :param env: gym api environment instance
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
                apple_state_on_tagged = []
                frames = []
                start = timer()

                # ep loop
                for t in range(self.ep_length):
                    # render
                    if ep % self.render_every == 0 and ep > 0:
                        title = f"ep: {ep}, frame: {t}, score: {total_reward}"
                        env.render(title=title)

                    # acting in the environment
                    actions = self.choose_actions(n_observations)

                    # make actions
                    next_n_observations, n_rewards, n_done, n_info = env.step(actions)

                    # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])
                    apple_state_on_tagged.extend([env.get_agent_current_rewards(a)/env.get_mean_total_current_rewards()
                                                  if env.get_mean_total_current_rewards() != 0 else 0
                                                  for a, obs in n_observations.items()
                                                  if obs.sum() == 0])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        self.agents[i].store(state=n_observations[index], next_state=next_n_observations[index],
                                             action=actions[index], reward=n_rewards[index], done=n_done[index])

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
                    n_observations = next_n_observations

                    # save frames of this episode
                    frames.append(Image.fromarray(np.uint8(env.get_full_state())).resize(size=(400, 180),
                                  resample=PIL.Image.BOX).convert("RGB"))

                # end of episode
                # fit agents
                if ep > self.min_to_learn:
                    results = self.fit(ep)

                    # display results
                    self.tensorboard(total_reward=total_reward, ep=ep - self.min_to_learn, results=results,
                                     writer=writer,
                                     epsilon=self.get_current_epsilon(), apple_state_on_tagged=apple_state_on_tagged,
                                     time_reward_collected=time_reward_collected, sleepers_time=sleepers_time)

                # save agents models to files
                if ep > self.save_every:
                    self.save()

                # save episode to gif
                if ep % self.save_gif_every == 0:
                    save_frames_as_gif(frames=frames, path=self.gifs, filename=f"episode_{ep}.gif")
                # log
                end = timer()
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end - start:.2f},"
                      f" epsilon: {self.agents[0].epsilon:.2f}")

    def tensorboard(self, total_reward, ep, results, epsilon, writer, time_reward_collected, sleepers_time,
                    apple_state_on_tagged):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param writer: tf writer
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param apple_state_on_tagged: list store total rewards of each agent tagged
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
        tf.summary.scalar(name="apple state on tagged", data=np.mean(apple_state_on_tagged), step=learned_ep),


        # display conv filters as images
        if ep % 20 == 0 and self.conv is True:
            grid = put_kernels_on_grid(self.agents[0].q_predict.trainable_weights[0])
            tf.summary.image(f"filters layer 1", grid, step=ep)

        # write data to log file
        writer.flush()

    def get_current_epsilon(self):
        return self.agents[0].epsilon

    def fit(self, ep):
        """
        call agent.fit for all the agents including meta agent
        :param ep: int, current iteration in training
        :return: dictionary including all results from agent fit (losses...)
        """

        # fit agents
        loss, max_predicted_reward = list(zip(*[agent.fit(ep) for agent in self.agents]))
        return {"loss": np.mean(loss), "max_predicted_reward": np.mean(max_predicted_reward)}

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
