import random
import numpy as np


def rand_orientation():
    return random.randint(0, 3)


def map_to_ascii(map_array):
    rows = len(map_array)
    columns = len(map_array[0])
    grid = np.hstack([list(row) for row in map_array]).reshape(rows, columns)
    return grid


def build_extended_grid(base_map, agents_vision):
    m, n = base_map.shape
    v = agents_vision
    extra = v*2
    grid = np.empty((m + extra, n + extra), dtype=str)
    m, n = grid.shape
    grid[v:m-v, v:n-v] = base_map
    return grid


def clean_map(grid):
    grid = grid.copy()
    for c in np.unique(grid):
        if c != "@":
            grid = np.char.replace(grid, c, "")
    return grid


def obs2gray_scale(obs):
    """"
    Return gray scale image from RBG by matplotlib implementation
    """
    return np.expand_dims(np.dot(obs[..., :3], [0.2989, 0.5870, 0.1140]), -1)
