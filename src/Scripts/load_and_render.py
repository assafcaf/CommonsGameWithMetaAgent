import os

from src.CommonsGame.game_environment.commons_env import HarvestCommonsEnv
from src.CommonsGame.game_environment.constants import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP
from src.PolicyGradientTrainerWithMetaAgent.singleAgents import DQNAgent

# init script's param
MAX_EPISODES = 20
EP_LENGTH = 300
n_players = 5

# init i/o paths
current_run = "2022-03-30_noMeta_5_DQN_512_512_seqLen_1"
pwd = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame"
agents_dir = os.path.join(fr"{pwd}", "models", current_run)
gif_file_name = "giff_{0}.gif"
gif_path = os.path.join(fr"{pwd}", "gifs", current_run)


# init env and load models
env = HarvestCommonsEnv(ascii_map=MEDIUM_HARVEST_MAP, num_agents=n_players)
n_agents = len([name for name in os.listdir(agents_dir) if not os.path.isfile(os.path.join(agents_dir, name))])

agents = [DQNAgent(env.observation_space.shape, env.action_space.n,
                   save_directory=os.path.join(agents_dir, f"DQAgent_{i}"))
          for i in range(n_players)]

[agent.load(os.path.join(agents_dir, f"DQAgent_{i}")) for i, agent in enumerate(agents)]

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