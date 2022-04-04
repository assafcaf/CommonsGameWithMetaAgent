import numpy as np


class QAgentBuffer:
    def __init__(self, buffer_size, input_shape, n_actions, discrete=False):
        self.mem_size = int(buffer_size)
        self.discrete = discrete
        self.n_actions = n_actions
        self.mem_cntr = 0

        self.state_memory = np.zeros(((self.mem_size,) + input_shape))
        self.new_state_memory = np.zeros(((self.mem_size,) + input_shape))
        dtype = np.float32 if self.discrete else np.int8
        if discrete:
            self.action_memory = np.zeros((self.mem_size, n_actions), dtype=dtype)
        else:
            self.action_memory = np.zeros(self.mem_size, dtype=dtype)
        self.reward_memory = np.zeros(self.mem_size)
        self.terminal_memory = np.zeros(self.mem_size)

    def store(self, state, state_, action, reward, done):
        index = self.mem_cntr % self.mem_size
        self.state_memory[index] = state
        self.new_state_memory[index] = state_
        self.reward_memory[index] = reward
        self.terminal_memory[index] = 1 - int(done)
        if self.discrete:
            actions = np.zeros(self.n_actions)
            actions[action] = 1
            self.action_memory[index] = actions
        else:
            self.action_memory[index] = action
            self.mem_cntr += 1

    def sample_buffer(self, batch_size):
        max_mem = min(self.mem_cntr, self.mem_size)
        batch = np.random.choice(max_mem, batch_size)
        state = self.state_memory[batch]
        state_ = self.new_state_memory[batch]
        reward = self.reward_memory[batch]
        actions = self.action_memory[batch]
        terminal = self.terminal_memory[batch]

        return state, state_, actions, reward, terminal

    def is_buffer_full(self):
        return self.mem_cntr >= self.mem_size
