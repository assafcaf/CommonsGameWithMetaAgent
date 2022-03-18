import os
import numpy as np


def save_frames_as_gif(frames, path='./', filename='gym_animation.gif'):
    if not os.path.isdir(path):
        os.mkdir(path)
    frames[0].save(os.path.join(path, filename), save_all=True, append_images=frames, duration=15)


def clean_memory():
    """
    :return: return 3 empty lists
    """
    return [], [], []


def discounted_reward(rewards, gamma, normalize=False):
    """
    take 1D float array of rewards and compute discounted reward
    :param rewards: list, indicate each time index a reward has given
    :param gamma: float, discount factor
    :param normalize: boolean, indicate if perform mean normalization
    :return np.array, discounted rewards
    """
    r = np.array(rewards)
    discounted_r = np.zeros_like(r)
    running_add = 0
    cnt = 0
    for t in reversed(range(len(r))):
        if cnt > 25:
            running_add = 0
            cnt = 0
        if r[t] == 1:
            vvv=1
        running_add = running_add * gamma + r[t]
        discounted_r[t] = running_add
        cnt += 1
    if sum(discounted_r) and normalize is True:
        return (discounted_r - discounted_r.mean()) / discounted_r.std()
    else:
        return discounted_r


