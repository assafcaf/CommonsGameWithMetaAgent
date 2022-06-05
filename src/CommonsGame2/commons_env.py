from utils import *
from commons_agent import Agent
import matplotlib.pyplot as plt
from constants import *
import time


class MapEnv:

    def __init__(self, bass_map, num_agents=1, color_map=None, agents_vision=6, beam_size=(1, 6),
                 characters_map=CHARACTERS_MAP):
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

        self.base_map = map_to_ascii(bass_map)
        self.extended_grid = build_extended_grid(self.base_map, self.agents_vision)

        self.agents_respawn_position = list(zip(*np.where(self.extended_grid == "P")))
        self.apples_respawn_position = list(zip(*np.where(self.extended_grid == self.characters_map["apple"])))
        self.apple_eaten = set()

        self.wales_rows = self.agents_vision, self.agents_vision + len(bass_map)-1
        self.wales_columns = self.agents_vision, self.agents_vision + len(bass_map[0])-1

        self.agents = {f"agent-{i}": Agent(f"agent-{i}", characters_map=characters_map) for i in range(num_agents)}
        self.reset()

        self.color_map = color_map
        plt.ion()

    def reset(self):
        self.extended_grid = build_extended_grid(self.base_map, self.agents_vision)
        self.extended_grid = np.char.replace(self.extended_grid, "P", self.characters_map["space"])
        self.apple_eaten = set()
        positions = random.sample(self.agents_respawn_position, self.num_agents)
        for i, agent in enumerate(self.agents.values()):
            agent.reset(positions[i], self.base_map)

    def get_partial_observation(self, agent_id):
        p = self.agents[agent_id].position
        v = self.agents_vision
        if self.agents[agent_id].is_sleep():
            observation = np.zeros((2*v+1, 2*v+1, 3))
        else:
            observation = self.map_to_colors(self.extended_grid[p[0]-v:p[0]+v+1, p[1]-v:p[1]+v+1])
        return observation

    def update_apples(self):
        spawned_apples = []
        for apple_pos in set(self.apples_respawn_position) - set(self.apple_eaten):
            self.extended_grid[apple_pos] = self.characters_map["apple"]

        for apple_pos in self.apple_eaten:
            if self.extended_grid[apple_pos] != self.characters_map["apple"]:
                s = self.n_neighbors(apple_pos)
                if random.random() < self.spawn_props[min(s, 5)]:
                    self.extended_grid[apple_pos] = self.characters_map["apple"]
                    spawned_apples.append(apple_pos)

        for apple_pos in spawned_apples:
            self.apple_eaten.remove(apple_pos)

    def n_neighbors(self, pos):
        neighbors = self.extended_grid[pos[0]-1: pos[0]+2, pos[1]-1: pos[1]+2]
        return (neighbors == self.characters_map["apple"]).sum()

    def convert_to_extended_position(self, pos):
        return pos[0] + self.agents_vision, pos[1] + self.agents_vision

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

    def total_rewards(self):
        return {agent_name: agent.total_rewards for agent_name, agent in self.agents.items()}

    def step(self, actions):
        if self.characters_map["beam"] in self.extended_grid:
            self.extended_grid = np.char.replace(self.extended_grid, self.characters_map["beam"],
                                                 self.characters_map["space"])
        self.update_apples()

        agent_shuffle_order = random.sample(self.agents.keys(), len(self.agents))
        n_observation = {f"agent-{i}": None for i in range(self.num_agents)}
        n_rewards = {f"agent-{i}": None for i in range(self.num_agents)}
        n_done = {f"agent-{i}": None for i in range(self.num_agents)}

        for agent_id in agent_shuffle_order:
            # make agent act and check if an apple was eaten in that turn
            if self.agents[agent_id].act(actions[agent_id], self.extended_grid):
                if self.agents[agent_id].is_rewarded():
                    self.apple_eaten.add(self.agents[agent_id].position)
                self.updated_grid(agent_id)
            n_observation[agent_id] = self.get_partial_observation(agent_id)
            n_rewards[agent_id] = self.agents[agent_id].current_reward
            n_done[agent_id] = False
        self.checks_sleeping_agents()
        return n_observation, n_rewards, n_done, None

    def handle_fire(self, agent_id):
        f = self.characters_map["beam"]
        if self.agents[agent_id].fire:
            pos = self.agents[agent_id].position
            left_wall, right_wall = self.wales_columns
            upper_wall, down_wall = self.wales_rows
            beam_width, beam_height = self.beam_size
            if self.agents[agent_id].orientation[0] == -1:  # sight is up
                x_start, x_end = sorted([pos[0]+1, max(upper_wall+1, pos[0] - beam_height)])
                y_start, y_end = sorted([max(left_wall+1, pos[1] - beam_width), min(right_wall, pos[1] + beam_width+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            elif self.agents[agent_id].orientation[1] == 1:  # sight is right
                x_start, x_end = sorted([max(upper_wall+1, pos[0] - beam_width), min(down_wall, pos[0] + beam_width+1)])
                y_start, y_end = sorted([pos[1], min(right_wall, pos[1] + beam_height+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            elif self.agents[agent_id].orientation[0] == 1:  # sight is down
                x_start, x_end = sorted([pos[0], min(down_wall, pos[0] + beam_height)])
                y_start, y_end = sorted([max(left_wall+1, pos[1] - beam_width), min(right_wall, pos[1] + beam_width+1)])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

            else:  # sight is left
                x_start, x_end = sorted([max(upper_wall+1, pos[0] - beam_width), min(down_wall, pos[0] + beam_width+1)])
                y_start, y_end = sorted([max(left_wall+1, pos[1] - beam_height), pos[1]+1])
                self.extended_grid[x_start: x_end, y_start: y_end] = f

    def clean_position(self, position, sight):
        self.extended_grid[position] = self.characters_map["space"]  # handle previous position of sight

        if sight[0] in self.wales_rows or sight[1] in self.wales_columns:
            self.extended_grid[sight] = self.characters_map["wall"]
        elif sight not in self.apple_eaten and sight in self.apples_respawn_position:
            self.extended_grid[sight] = self.characters_map["apple"]
        else:
            self.extended_grid[sight] = self.characters_map["space"]

    def checks_sleeping_agents(self):
        for agent_id, agent in self.agents.items():
            if self.extended_grid[agent.position] == self.characters_map["beam"]:
                agent.hit()
                pos = self.agents[agent_id].position
                sight_pos = tuple(np.array(self.agents[agent_id].position) + np.array(self.agents[agent_id].orientation))
                self.clean_position(pos, sight_pos)
                print(f"{agent_id} got hot shoot")

    def updated_grid(self, agent_id):
        pos = self.agents[agent_id].position
        prev_pos = self.agents[agent_id].prev_position
        sight_pos = tuple(np.array(self.agents[agent_id].position) + np.array(self.agents[agent_id].orientation))
        prev_sight_pos = tuple(
            np.array(self.agents[agent_id].prev_position) + np.array(self.agents[agent_id].prev_orientation))

        # TODO: change generic agent symbol to colored char

        # self.extended_grid[prev_pos] = " "  # handle previous position of sight
        #
        # if prev_sight_pos[0] in self.wales_rows or prev_sight_pos[1] in self.wales_columns:
        #     self.extended_grid[prev_sight_pos] = "@"
        # elif prev_sight_pos not in self.apple_eaten and prev_sight_pos in self.apples_respawn_position:
        #     self.extended_grid[prev_sight_pos] = "A"
        # else:
        #     self.extended_grid[prev_sight_pos] = " "
        self.clean_position(prev_pos, prev_sight_pos)

        if self.agents[agent_id].is_sleep():
            print("123")
            return

        self.handle_fire(agent_id)

        if self.extended_grid[pos] != self.characters_map["beam"]:
            self.extended_grid[pos] = self.characters_map["agent"]
            self.extended_grid[sight_pos] = self.characters_map["sight"] \
                if self.extended_grid[sight_pos] == self.characters_map["space"] \
                else self.extended_grid[sight_pos]