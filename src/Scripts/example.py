import matplotlib.pyplot as plt
from Danfoa_CommonsGame.commons_game.game_environment.commons_env import HarvestCommonsEnv
from Danfoa_CommonsGame.commons_game.game_environment.constants import HARVEST_MAP, MEDIUM_HARVEST_MAP, SMALL_HARVEST_MAP
num_agents = 4

# Create a custom Gym environment for the commons game
env = HarvestCommonsEnv(ascii_map=HARVEST_MAP, num_agents=num_agents)
# Reset environment and get agents observations
observations = env.reset()
# Plot how the env looks like
env.render(title="The Commons Game")

# Plot each of the agent observation's
fig, ax = plt.subplots(1, num_agents, figsize=(8,4), squeeze=True)
for i, agent_id in enumerate(env.agents.keys()):
    ax[i].imshow(observations[agent_id])
    ax[i].set_title(agent_id)
plt.show()