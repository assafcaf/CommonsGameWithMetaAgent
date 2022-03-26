import os
import tensorflow as tf
from timeit import default_timer as timer
from collections import deque
# local imports
from .singleAgents import PGAgent, PGMetaAgent, DQNAgent
from .metrics import *
from .utils import put_kernels_on_grid


class TrainerWithMetaAgent:
    def __init__(self, input_shape, state_dim, num_actions, lr=0.00075,  n_players=1, max_episodes=50000,
                 render_every=np.inf, ep_length=150, save_every=100,  models_directory="models",  log_dir=r"logs/",
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
                                 sleepers_time=sleepers_time, ep=ep, results=results, meta_total_rewards=meta_total_rewards,
                                 meta_actions=np.array(meta_actions), writer=writer)
                # log
                end = timer()
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end-start:.2f}")

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
    def __init__(self, input_shape, state_dim, num_actions, lr=0.00075,  n_players=1, max_episodes=50000,
                 render_every=np.inf, ep_length=150, save_every=100,  models_directory="models",  log_dir=r"logs/",
                 gamma=0.99):
        """
        :param state_dim: tuple, dimensions of full states in the environment
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
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end-start:.2f}")

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
    def __init__(self, input_shape, num_actions, lr, ep_length, n_players, gamma, max_episodes=50000, batch_size=512,
                 render_every=np.inf, save_every=100,  models_directory="models",  log_dir=r"logs/", min_to_learn=50,
                 buffer_size=int(1e6), epsilon_decay_rate=0.99, update_every=50, min_epsilon=.1, conv=False):
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
        self.seq_len = input_shape[-1]
        self.save_every = save_every
        self.models_directory = models_directory
        self.log_dir = os.path.join(log_dir)
        self.writer = self.handle_callbacks(models_directory)
        self.conv = conv

        # simple agents
        self.agents = [DQNAgent(input_shape, num_actions, epsilon_decay_rate=epsilon_decay_rate, batch_size=batch_size,
                                lr=lr, gamma=gamma, save_directory=os.path.join(models_directory, f"DQAgent_{i}"),
                                update_every=update_every, buffer_size=buffer_size, min_to_learn=min_to_learn,
                                min_epsilon_value=min_epsilon)
                       for i in range(n_players)]

        self.agents[0].q_predict.summary()

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
                n_observations_deque = {f'agent-{i}': deque(maxlen=self.seq_len) for i in range(len(self.agents))}
                n_observations_next_deque = n_observations_deque.copy()

                # state = dict()
                # for agent in n_observations_deque.keys():
                #     for _ in range(self.seq_len):
                #         n_observations_deque[agent].append(n_observations[agent])
                #         n_observations_next_deque[agent].append(n_observations[agent])
                #     state[agent] = np.stack(n_observations_deque[agent], axis=-1).squeeze(-2)

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

                    # state, next_state = dict(), dict()
                    # for agent in n_observations_deque.keys():
                    #     n_observations_deque[agent].append(n_observations[agent])
                    #     state[agent] = np.stack(n_observations_deque[agent], axis=-1).squeeze(-2)
                    #
                    #     n_observations_next_deque[agent].append(next_n_observations[agent])
                    #     next_state[agent] = np.stack(n_observations_next_deque[agent], axis=-1).squeeze(-2)

                # collect behavior data for analysis
                    time_reward_collected.extend([t for a, r in n_rewards.items() if r != 0])
                    sleepers_time.extend([1 for a, obs in n_observations.items() if obs.sum() == 0])

                    # store agents (observation, action, rewards) for learning
                    for i in range(self.n_players):
                        index = f"agent-{i}"
                        self.agents[i].store(state=n_observations[index], next_state=next_n_observations[index],
                                             action=actions[index], reward=n_rewards[index], done=n_done[index])

                    # collect rewards
                    total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
                    n_observations = next_n_observations

                # end of episode
                # fit agents
                if ep > self.min_to_learn:
                    results = self.fit(ep)

                    # display results
                    self.tensorboard(total_reward=total_reward, time_reward_collected=time_reward_collected,
                                     sleepers_time=sleepers_time, ep=ep-self.min_to_learn, results=results,
                                     writer=writer, epsilon=self.agents[0].epsilon)

                # log
                end = timer()
                print(f"# {ep}, total_rewards: {total_reward.sum()}, time: {end-start:.2f},"
                      f" epsilon: {self.agents[0].epsilon:.2f}")

    def tensorboard(self, total_reward, time_reward_collected, sleepers_time, ep, results, epsilon, writer):
        """
            create tensorboard graph to display the algorithm progress
        :param total_reward: list, sum reward of full episode for each of the agents
        :param time_reward_collected: list,the time on which the rewards have collected
        :param sleepers_time: list, indicate when agent where out of the game due to tagging
        :param writer: tf.writer
        :param epsilon, current exploration rate of DQN agents
        :param results: dictionary, results of fit method (agents and meta_agent loss and q_values)
        :param ep: int, indicate episode number
        """

        learned_ep = ep
        for k, v in results.items():
            tf.summary.scalar(name=k, data=v, step=learned_ep)

        # display metrics
        tf.summary.scalar(name="total reward", data=total_reward.sum(), step=learned_ep)
        tf.summary.scalar(name="epsilon", data=epsilon, step=learned_ep)
        tf.summary.scalar(name="efficiency", data=efficiency(total_reward, self.n_players, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="sustainability", data=sustainability(time_reward_collected, self.ep_length),
                          step=learned_ep)

        tf.summary.scalar(name="equality", data=equality(total_reward, self.n_players), step=learned_ep)
        tf.summary.scalar(name="peace", data=peace(sleepers_time, self.n_players, self.ep_length),
                          step=learned_ep)
        if ep % 20 == 0 and self.conv is True:
            grid = put_kernels_on_grid(self.agents[0].q_predict.trainable_weights[0])
            tf.summary.image(f"filters layer 1", grid, step=ep)
        writer.flush()

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

