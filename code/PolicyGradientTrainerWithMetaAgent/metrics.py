import numpy as np


def equality(rewards, n_players):
    """
    compute equality metric given rewards list and amount of players
    :param rewards: np.array, dim: (ep_length, n_players) summary of ep rewards flow
    :param n_players: int, represent amount of players participants during the episode
    :return: float, equality metric [0, 1]
    """
    s = sum([np.abs(r1-r2)for r1 in rewards for r2 in rewards])
    denominator = (2*n_players * sum(rewards))
    return (1 - s / denominator) if denominator != 0 else 0


def sustainability(time_reward_collected, t):
    """
    compute sustainability metric given list that indicates each time index that reward has given
    :param time_reward_collected: list, each element represents index which reward has given
    :param t: int, repeats the max time index possibly by episode configuration
    :return: float, sustainability metric [0, 1]
    """
    times = np.array(time_reward_collected).astype(np.float)
    avg = np.nanmean(times, axis=0)
    return avg / t


def peace(sleepers_time, n_players, t):
    """
    compute peace metric given list that indicates each time index that one of agent was out of the game
    :param sleepers_time: list, indices of time index that one of the agent was out of game
    :param n_players: int, repeats the amount of players by episode configuration
    :param t: int, repeats the max time index possibly by episode configuration
    :return: float, peace metric [0, 1]
    """
    times = np.array(sleepers_time).astype(np.float)
    players_time_sleep = np.nansum(times)
    nt = n_players * t
    return (nt - players_time_sleep) / (n_players * t)


def efficiency(total_rewards, n_players, ep_length):
    """
    compute efficiency metric given list that represents the amount of reward each player got duration episode
    :param total_rewards: np.array, repeats the amount of rewards each agent got
    :param n_players: int, repeats the amount of players by episode configuration
    :param ep_length: int, repeats the max time index possibly by episode configuration
    :return: float, efficiency metric [0, 1]
    """
    return sum(total_rewards) / (n_players * ep_length)



