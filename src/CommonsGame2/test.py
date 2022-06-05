import random
from commons_env import MapEnv
from constants import *

num_agents = 6
vision = 11
env = MapEnv(color_map=DEFAULT_COLOURS, bass_map=SMALL_HARVEST_MAP, num_agents=num_agents, agents_vision=vision)
for ep in range(5):
    env.reset()
    for t in range(1000):
        actions = {f"agent-{i}": random.randint(0, 7) for i in range(num_agents)}
        # actions = {f"agent-{i}": int(input(f"{t}: ")) for i in range(num_agents)}

        n_observation, n_rewards, n_done, _ = env.step(actions=actions)
        env.render(wait_time=1, title=f"time: {t}, rewards: {env.total_rewards().values()}")



