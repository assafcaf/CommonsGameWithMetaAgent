from CommonsGame3.utils import *
from CommonsGame3.constants import *
# all actions
ACTIONS = { 0: 'MOVE_LEFT',  # Move left
            1: 'MOVE_RIGHT',  # Move right
            2: 'MOVE_UP',  # Move up
            3: 'MOVE_DOWN',  # Move down
            4: 'STAY',  # don't move
            5: 'SIGHT_CLOCKWISE',  # Rotate clockwise
            6: 'SIGHT_COUNTERCLOCKWISE',  # Rotate counter-clockwise
            7: "SHOOT"}  # Shoot a laser

# all orientations
ORIENTATION = {0: (-1, 0),  # up
               1:  (0, 1),  # right
               2:  (1, 0),  # down
               3: (0, -1)}  # left


class Agent(object):
    def __init__(self, name, position=None, _sleep_duration=25, sleep=False, characters_map=CHARACTERS_MAP):
        self.name = name
        self.characters_map = characters_map
        self._position = position
        self._prev_position = position
        self._orientation = rand_orientation()
        self._prev_orientation = self._orientation
        self._total_rewards = 0
        self._current_reward = False
        self._sleep_flag = sleep
        self._sleep_duration = _sleep_duration
        self._hit_time = None
        self.current_timestamp = 0
        self._fire = False
        self._action_applied = False

    @property
    def action_applied(self):
        return self._action_applied

    @action_applied.setter
    def action_applied(self, r):
        self._action_applied = r

    @property
    def total_rewards(self):
        return self._total_rewards

    @total_rewards.setter
    def total_rewards(self, r):
        self._total_rewards = r

    @property
    def fire(self):
        return self._fire

    @fire.setter
    def fire(self, f):
        self._fire = f

    @property
    def position(self):
        return self._position

    @position.setter
    def position(self, pos):
        self._position = pos

    @property
    def prev_position(self):
        return self._prev_position

    @prev_position.setter
    def prev_position(self, pos):
        self._prev_position = pos

    @property
    def current_reward(self):
        return self._current_reward

    @current_reward.setter
    def current_reward(self, r):
        self._current_reward = r

    @property
    def orientation(self):
        return ORIENTATION[self._orientation]

    @orientation.setter
    def orientation(self, o):
        if len(o) == 1:
            self._orientation = o
        elif o == ORIENTATION[0]:
            self._orientation = 0
        elif o == ORIENTATION[1]:
            self._orientation = 1
        elif o == ORIENTATION[2]:
            self._orientation = 2
        else:
            self._orientation = 3

    @property
    def prev_orientation(self):
        return ORIENTATION[self._prev_orientation]

    @prev_orientation.setter
    def prev_orientation(self, o):
        if len(o) == 1:
            self._prev_orientation = o
        elif o == ORIENTATION[0]:
            self._prev_orientation = 0
        elif o == ORIENTATION[1]:
            self._prev_orientation = 1
        elif o == ORIENTATION[2]:
            self._prev_orientation = 2
        else:
            self._prev_orientation = 3

    def reset(self, pos, grid):
        self.position = pos
        self.prev_position = pos
        self._orientation = rand_orientation()
        self._prev_orientation = self._orientation
        self._total_rewards = 0
        self._current_reward = False
        self._sleep_flag = False
        self._hit_time = None

    def hit(self):
        self._sleep_flag = True
        self._hit_time = self.current_timestamp
        self.position = (-1, -1)
        # print(f"agent {self.name} is sleeping at time {self.current_timestamp}")

    def is_sleep(self):
        return self._sleep_flag

    def is_time_to_wake_up(self):
        return self._hit_time + self._sleep_duration <= self.current_timestamp

    def wake_up(self, pos):
        self.position = pos
        self.orientation = ORIENTATION[rand_orientation()]
        self._sleep_flag = False
        self._hit_time = None
        # print(f"agent {self.name} is woke up")

    def act(self, action, grid):
        """
        making agent action by specified action and the current state of grid
        @param action: integer from action space
        @param grid: the current state of the grid
        @return bool, indicate if agent made action or not
        """
        self.fire = False
        # update current time-stamp

        self.current_timestamp += 1
        self.current_reward = 0
        self._prev_position = self._position
        self._prev_orientation = self._orientation
        self._action_applied = False
        if self.is_sleep():
            return

        # move left
        if action == 0:
            new_pos = (self.position[0], self.position[1]-1)
            if self.checks_new_position(new_pos, grid):
                self.position = new_pos

        # move right
        elif action == 1:
            new_pos = (self.position[0], self.position[1]+1)
            if self.checks_new_position(new_pos, grid):
                self.position = new_pos

        # move up
        elif action == 2:
            new_pos = (self.position[0]-1, self.position[1])
            if self.checks_new_position(new_pos, grid):
                self.position = new_pos

        # move down
        elif action == 3:
            new_pos = (self.position[0]+1, self.position[1])
            if self.checks_new_position(new_pos, grid):
                self.position = new_pos

        # stay
        elif action == 4:
            pass

        # rotate clockwise
        elif action == 5:
            self._orientation += 1
            self._orientation %= 4

        # rotate counter-clockwise
        elif action == 6:
            # make sure _orientation will be in range of [0, 3]
            self._orientation -= 1 if self._orientation != 0 else -3
            self._orientation %= 4

        # fire
        elif action == 7:
            self.fire = True

    def checks_new_position(self, new_pos, grid):
        """
        checks if the new position is not on the wall
        @param new_pos: new position
        @param grid: current state of the grid
        @return:
        """
        return not grid[new_pos] == self.characters_map["wall"]

    def consume(self):
        self.current_reward = 1
        self.total_rewards += self.current_reward

    def is_rewarded(self):
        return self.current_reward == 1
