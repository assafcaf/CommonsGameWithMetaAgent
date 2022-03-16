import os.path

import numpy as np
import gym
import time
import warnings
from PIL import Image
import PIL
import tensorflow as tf
import sys
warnings.filterwarnings("ignore")

# local imports
sys.path.append(os.path.join(os.getcwd()))
from Danfoa_CommonsGame.agents.MetaDQNAgent.utils import save_frames_as_gif
from Danfoa_CommonsGame.agents.MetaDQNAgent.metaAgent import MultiDQNAgentTest


# tensorflow run params
# os.environ['CUDA_VISIBLE_DEVICES'] = '-1'
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.25)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))


# init script's param
random_chance = 0
MAX_EPISODES = 20
EP_LENGTH = 200
n_players = 10

# init i/o paths
current_run = "2022-01-26_dense_256x2_10_agents"
AGENTS_FOLDER = os.path.join(fr"C:\studies\IDC_dataScience\thesis\CommonsGameDQN\models", current_run)
gif_file_name = "players_{0}_ep_{1}_tr_{2}_randomChance_{3}.gif"
gif_path = os.path.join(fr"C:\studies\IDC_dataScience\thesis\CommonsGameDQN\gifs", current_run)


# init env and load models
env = gym.make('CommonsGame:CommonsGame-v0', numAgents=n_players, visualRadius=10)
multi_agent = MultiDQNAgentTest(n_players=n_players, models_directory=AGENTS_FOLDER)

# games loop
total_rewards = []
for ep in range(MAX_EPISODES):
    total_reward = np.array([0] * n_players)
    n_observations = np.array(env.reset())
    actions = [0] * n_players
    start = time.time()
    render_memory = []

    # ep loop
    for t in range(EP_LENGTH):
        env_state = env.render(wait_time=0.0001)
        render_memory.append(Image.fromarray(np.uint8(env_state)).resize(size=(400, 180), resample=PIL.Image.BOX).convert("RGB"))

    # acting in the environment
        actions = multi_agent.choose_actions_r(n_observations, random_chance)
        next_n_observations, n_rewards, n_done, n_info = env.step(actions)

        # collect rewards
        total_reward += np.array(n_rewards)
        n_observations = next_n_observations

    end = time.time()
    print(f"ep: {ep}, reward_total: {sum(total_reward)}, rewards: {total_reward}, time {end - start}")
    start = end
    save_frames_as_gif(frames=render_memory, path=gif_path, filename=gif_file_name.format(n_players, ep, sum(total_reward), random_chance))