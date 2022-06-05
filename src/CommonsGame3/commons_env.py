from CommonsGame3.utils import *
from CommonsGame3.commons_agent import Agent
import matplotlib.pyplot as plt
from CommonsGame3.constants import *


class MapEnv:

    def __init__(self, bass_map, num_agents=1, color_map=None, agents_vision=11, beam_size=(1, 6),
                 characters_map=CHARACTERS_MAP, gray_scale=True):
        """

        Parameters
        ----------
        bass_map: list of strings
            Specify what the map should look like. Look at constant.py for
            further explanation
        num_agents: int
            Number of agents to have in the system.
        color_map: dict
            Specifies how to convert between ascii chars and colors
        """
        self.num_agents = num_agents
        self.agents_vision = (agents_vision-1)//2
        self.beam_size = beam_size
        self.spawn_props = SPAWN_PROB
        self.characters_map = characters_map
        self.gray_scale = gray_scale

        self.base_map = map_to_ascii(bass_map)
        self.extended_grid = build_extended_grid(self.base_map, self.agents_vision)

        self.agents_respawn_position = list(zip(*np.where(self.extended_grid == "P")))
        self.apples_respawn_position = list(zip(*np.where(self.extended_grid == self.characters_map["apple"])))
        self.apple_eaten = set()

        self.wales_rows = self.agents_vision, self.agents_vision + len(bass_map)-1
        self.wales_columns = self.agents_vision, self.agents_vision + len(bass_map[0])-1

        self.agents = {f"agent-{i}": Agent(f"agent-{i}", characters_map=characters_map) for i in range(num_agents)}

        self.color_map = color_map
        self.observation_space_shape = (2*self.agents_vision+1, 2*self.agents_vision+1, 3)
        self.action_space_n = 8

        self.current_hits = 0
        self.reset()
        plt.ion()

    def total_rewards(self):
        return {agent_name: agent.total_rewards for agent_name, agent in self.agents.items()}

    def get_mean_total_current_rewards(self):
        return np.mean(list(self.total_rewards().values()))

    def get_agent_current_rewards(self, agent_id):
        return self.agents[agent_id].current_reward

    def reset(self):
        self.extended_grid = build_extended_grid(self.base_map, self.agents_vision)
        self.extended_grid = np.char.replace(self.extended_grid, "P", self.characters_map["space"])
        self.apple_eaten = set()
        positions = random.sample(self.agents_respawn_position, self.num_agents)
        for i, agent in enumerate(self.agents.values()):
            agent.reset(positions[i], self.base_map)
        return {agent_id:self.get_partial_observation(agent_id) for agent_id in self.agents.keys()}

    def get_partial_observation(self, agent_id):
        p = self.agents[agent_id].position
        v = self.agents_vision
        if self.agents[agent_id].is_sleep():
            observation = np.zeros(self.observation_space_shape)
        else:
            observation = self.map_to_colors(self.extended_grid[p[0]-v:p[0]+v+1, p[1]-v:p[1]+v+1])
            observation[v, v] = self.color_map['O']

        if self.gray_scale:
            observation = obs2gray_scale(observation)
        return observation

    def update_apples(self):
        spawned_apples = []
        # redraw uneaten apple that wasn't on the map (due to sight stepped over it)
        for apple_pos in set(self.apples_respawn_position) - set(self.apple_eaten):
            self.extended_grid[apple_pos] = self.characters_map["apple"]

        # respawn eaten apples by predefined probabilities
        for apple_pos in self.apple_eaten:
            if self.extended_grid[apple_pos] != self.characters_map["apple"]:
                if random.random() < self.spawn_props[min((self.n_neighbors(apple_pos), 5))]:
                    self.extended_grid[apple_pos] = self.characters_map["apple"]
                    spawned_apples.append(apple_pos)

        # remove respawned apple from eaten apple set
        for apple_pos in spawned_apples:
            self.apple_eaten.remove(apple_pos)

    def n_neighbors(self, pos):
        neighbors = self.extended_grid[pos[0]-1: pos[0]+2, pos[1]-1: pos[1]+2]
        return np.sum(neighbors == self.characters_map["apple"])

    def clean_beams(self):
        if self.characters_map["beam"] in self.extended_grid:
            self.extended_grid = np.char.replace(self.extended_grid, self.characters_map["beam"],
                                                 self.characters_map["space"])

    def map_to_colors(self, grid):
        """Converts a map to an array of RGB values.
        Parameters
        ----------
        grid: np.ndarray
            map to convert to RGB colors
        Returns
        -------
        arr: np.ndarray
            3-dim numpy array consisting of color map
        """
        shape = grid.shape
        rgb_arr = np.zeros((*shape, 3), dtype=int)
        for row_elem in range(shape[0]):
            for col_elem in range(shape[1]):
                rgb_arr[row_elem, col_elem, :] = self.color_map[grid[row_elem, col_elem]]

        return rgb_arr

    def render(self, filename=None, title=None, wait_time=0.05):
        """ Creates an image of the map to plot or save.

        Args:
            filename: If a string is passed, will save the image
                to disk at this location.
            title: string, title for plotting
            wait_time: time to sleep between each frame
        """
        rgb_arr = self.map_to_colors(self.extended_grid)
        if title:
            plt.title(title)
        plt.imshow(rgb_arr, aspect='auto')
        if filename is None:
            plt.show()
            plt.axis('off')
            plt.pause(wait_time)
            plt.clf()

        else:
            plt.savefig(filename, dpi=80)
            plt.close()
        # if sum([agent.is_sleep() for agent in self.agents.values()]):
        #     time.sleep(7)

    def step(self, actions):
        # clean previous beam and update apples on board
        self.clean_beams()
        self.update_apples()

        #

        # create en empty gym-like parameters
        n_observation = {f"agent-{i}": None for i in range(self.num_agents)}
        n_rewards = {f"agent-{i}": None for i in range(self.num_agents)}
        n_done = {f"agent-{i}": None for i in range(self.num_agents)}
        self.current_hits = 0

        # agents actions
        for agent_id, agent in self.agents.items():
            # make agent act and check if an apple was eaten in that turn
            agent.act(actions[agent_id], self.extended_grid)

        # checks for conflicts
        self.checks_conflicts()

        # update gym-like parameters
        for agent_id, agent in self.agents.items():
            # make agent act and check if an apple was eaten in that turn
            n_observation[agent_id] = self.get_partial_observation(agent_id)
            n_rewards[agent_id] = agent.current_reward
            n_done[agent_id] = False
        return n_observation, n_rewards, n_done, None

    def get_full_state(self):
        extended_grid = self.map_to_colors(self.extended_grid)
        n, m, c = extended_grid.shape
        v = self.agents_vision
        return extended_grid[v: n-v, v:m-v, :]

    def handle_fire(self, agent_id):
        f = self.characters_map["beam"]
        if self.agents[agent_id].fire and not self.agents[agent_id].is_sleep():
            x, y = self.agents[agent_id].position
            left_wall, right_wall = self.wales_columns
            upper_wall, down_wall = self.wales_rows
            beam_width, beam_height = self.beam_size

            if self.agents[agent_id].orientation[0] == -1:  # sight is up
                x_start, x_end = sorted([x+1, max(upper_wall+1, x - beam_height)])
                y_start, y_end = sorted([max(left_wall+1, y - beam_width), min(right_wall, y + beam_width+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            elif self.agents[agent_id].orientation[1] == 1:  # sight is right
                x_start, x_end = sorted([max(upper_wall+1, x - beam_width), min(down_wall, x + beam_width+1)])
                y_start, y_end = sorted([y, min(right_wall, y + beam_height+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            elif self.agents[agent_id].orientation[0] == 1:  # sight is down
                x_start, x_end = sorted([x, min(down_wall, x + beam_height)])
                y_start, y_end = sorted([max(left_wall+1, y - beam_width), min(right_wall, y + beam_width+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            else:  # sight is left
                x_start, x_end = sorted([max(upper_wall+1, x - beam_width), min(down_wall, x + beam_width+1)])
                y_start, y_end = sorted([max(left_wall+1, y - beam_height),y+1])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            self.extended_grid[x, y] = self.characters_map["agent"]

    def clean_position(self, position, sight):
        self.extended_grid[position] = self.characters_map["space"]

        # handle previous position of sight
        if sight[0] in self.wales_rows or sight[1] in self.wales_columns:
            self.extended_grid[sight] = self.characters_map["wall"]
        elif sight not in self.apple_eaten and sight in self.apples_respawn_position:
            self.extended_grid[sight] = self.characters_map["apple"]
        else:
            self.extended_grid[sight] = self.characters_map["space"]

    def checks_conflicts(self):
        """
        run over positions of the agents and update the state of the game accordingly
        """

        # applying shooting first
        for agent_id, agent in self.agents.items():
            self.handle_fire(agent_id)

        beam_positions = list(zip(*np.where(self.extended_grid == self.characters_map["beam"])))

        # random shuffle order of agents to apply actions on
        random_agent_order = list(self.agents.items())
        random.shuffle(random_agent_order)

        # make action of non shooting agents
        for agent_id, agent in random_agent_order:

            # checks if agent has been tagged
            if agent.position in beam_positions:
                self.clean_position(position=agent.position,
                                    sight=tuple(np.array(agent.position) + np.array(agent.orientation)))
                self.clean_position(position=agent.prev_position,
                                    sight=tuple(np.array(agent.prev_position) + np.array(agent.prev_orientation)))
                agent.hit()
                self.current_hits += 1
                continue

            if not agent.fire:
                # remove from grid previous position and sight position of the agent
                self.clean_position(position=agent.prev_position,
                                    sight=tuple(np.array(agent.prev_position) + np.array(agent.prev_orientation)))

                if not agent.is_sleep():
                    # update grid according to current state of the game
                    self.update_state_state_by_agent(agent)
                elif agent.is_time_to_wake_up():
                    pos = random.sample(self.agents_respawn_position, 1)[0]
                    agent.wake_up(pos)

        for agent_id, agent in self.agents.items():
            self.handle_fire(agent_id)

    def get_current_hits(self):
        return self.current_hits

    def update_agent_position_on_grid(self, position, sight):
        self.extended_grid[position] = self.characters_map["agent"]

        if self.extended_grid[sight] == self.characters_map["wall"] or\
           self.extended_grid[sight] == self.characters_map["agent"] or \
            self.extended_grid[sight] == self.characters_map["apple"] or \
            self.extended_grid[sight] == self.characters_map["beam"]:
            pass
        else:
            self.extended_grid[sight] = self.characters_map["sight"]
            
    def update_state_state_by_agent(self, agent):
        if self.extended_grid[agent.position] == self.characters_map["space"]:  # if agent new position is empty
            sight_pos = tuple(np.array(agent.position) + np.array(agent.orientation))
            self.update_agent_position_on_grid(agent.position, sight_pos)

        elif self.extended_grid[agent.position] == self.characters_map["apple"]:  # if agent new position is on apple
            sight_pos = tuple(np.array(agent.position) + np.array(agent.orientation))
            self.update_agent_position_on_grid(agent.position, sight_pos)
            agent.consume()
            self.apple_eaten.add(agent.position)

        elif self.extended_grid[agent.position] == self.characters_map["agent"]:  # if agent new position is taken by other agent
            agent.position = agent.prev_position
            agent.orientation = agent.prev_orientation
            sight_pos = tuple(np.array(agent.position) + np.array(agent.orientation))
            self.update_agent_position_on_grid(agent.position, sight_pos)
            self.update_agent_position_on_grid(agent.position, sight_pos)
