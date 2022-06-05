import os
import numpy as np
import time

from src.CommonsGame3.commons_env import MapEnv  # my env
from src.CommonsGame3.constants import *  # my env
from src.CommonsGame.game_environment.constants import MEDIUM_HARVEST_MAP
from src.DQN.singleAgents import DQNAgent
from src.DQN.utils import obs_to_grayscale
# init script's param
MAX_EPISODES = 20
EP_LENGTH = 300
n_players = 5

# init i/o paths
current_run = "2022-05-16_noMeta_5_DQN_32_64_32_changingLR"
pwd = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame"
agents_dir = os.path.join(fr"{pwd}", "models", current_run)
gif_file_name = "giff_{0}.gif"
gif_path = os.path.join(fr"{pwd}", "gifs", current_run)


# init env and load models
env = MapEnv(bass_map=MEDIUM_HARVEST_MAP, num_agents=n_players, color_map=DEFAULT_COLOURS)
n_agents = len([name for name in os.listdir(agents_dir) if not os.path.isfile(os.path.join(agents_dir, name))])
input_shape = env.observation_space_shape
num_actions = env.action_space_n
agents = [DQNAgent(input_shape, num_actions,
                   save_directory=os.path.join(agents_dir, f"DQAgent_{i}"))
          for i in range(n_players)]

[agent.load(os.path.join(agents_dir, f"DQAgent_{i}")) for i, agent in enumerate(agents)]

# games loop
total_rewards = []
for ep in range(MAX_EPISODES):
    total_reward = np.array([0] * n_players)
    n_observations = env.reset()
    actions = [0] * n_players
    start = time.time()
    render_memory = []

    # ep loop
    for t in range(EP_LENGTH):
        env_state = env.render(wait_time=1)
        #
        # acting in the environment
        key = "agent-%d"
        actions = {key % i: agent.get_action(n_observations[key % i]) for i, agent in enumerate(agents)}
        next_n_observations, n_rewards, n_done, n_info = env.step(actions)

        # collect rewards
        total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
        n_observations = next_n_observations

    end = time.time()
    print(f"ep: {ep}, reward_total: {sum(total_reward)}, rewards: {total_reward}, time {end - start}")
    start = end
