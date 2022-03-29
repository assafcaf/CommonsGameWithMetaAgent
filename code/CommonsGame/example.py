from Danfoa_CommonsGame.code.CommonsGame.game_environment.commons_env import HarvestCommonsEnv
from Danfoa_CommonsGame.code.CommonsGame.game_environment.constants import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP
import numpy as np
EPISODES = 10
N_PLAYERS = 5
EP_LENGTH = 500
RENDER = True
env = HarvestCommonsEnv(ascii_map=MEDIUM_HARVEST_MAP, num_agents=N_PLAYERS)

for ep in range(1, EPISODES):
    env.reset()
    total_reward = np.zeros(shape=(1, N_PLAYERS))
    # ep loop
    for t in range(EP_LENGTH):
        # render
        if RENDER:
            title = f"ep: {ep}, frame: {t}, score: {total_reward}"
            env.render(title=title)

        # acting in the environment
        actions = {f"agent-{i}": np.random.randint(env.action_space.n) for i in range(N_PLAYERS)}

        # make actions
        next_n_observations, n_rewards, n_done, n_info = env.step(actions)

        # collect rewards
        total_reward += np.fromiter(n_rewards.values(), dtype=np.int)
        n_observations = next_n_observations
