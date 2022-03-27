import os
import numpy as np
from math import sqrt
import tensorflow as tf
import json

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


def rgb2gray(rgb_arr):
    return np.expand_dims(rgb_arr.mean(axis=-1), axis=-1)/255


def put_kernels_on_grid(kernel, pad=1):

    '''Visualize conv. filters as an image (mostly for the 1st layer).
    Arranges filters into a grid, with some paddings between adjacent filters.
    Args:
      kernel:            tensor of shape [Y, X, NumChannels, NumKernels]
      pad:               number of black pixels around each filter (between them)
    Return:
      Tensor of shape [1, (Y+2*pad)*grid_Y, (X+2*pad)*grid_X, NumChannels].
    '''
    # get shape of the grid. NumKernels == grid_Y * grid_X
    def factorization(n):
        for i in range(int(sqrt(float(n))), 0, -1):
            if n % i == 0:
                if i == 1: print('Who would enter a prime number of filters')
                return (i, int(n / i))
    (grid_Y, grid_X) = factorization (kernel.get_shape()[3])
    print ('grid: %d = (%d, %d)' % (kernel.get_shape()[3], grid_Y, grid_X))

    x_min = tf.reduce_min(kernel)
    x_max = tf.reduce_max(kernel)
    kernel = (kernel - x_min) / (x_max - x_min)

    # pad X and Y
    x = tf.pad(kernel, tf.constant( [[pad,pad],[pad, pad],[0,0],[0,0]] ), mode = 'CONSTANT')

    # X and Y dimensions, w.r.t. padding
    Y = kernel.get_shape()[0] + 2 * pad
    X = kernel.get_shape()[1] + 2 * pad

    channels = kernel.get_shape()[2]

    # put NumKernels to the 1st dimension
    x = tf.transpose(x, (3, 0, 1, 2))
    # organize grid on Y axis
    x = tf.reshape(x, tf.stack([grid_X, Y * grid_Y, X, channels]))

    # switch X and Y axes
    x = tf.transpose(x, (0, 2, 1, 3))
    # organize grid on X axis
    x = tf.reshape(x, tf.stack([1, X * grid_X, Y * grid_Y, channels]))

    # back to normal order (not combining with the next step for clarity)
    x = tf.transpose(x, (2, 1, 3, 0))

    # to tf.image_summary order [batch_size, height, width, channels],
    #   where in this case batch_size == 1
    x = tf.transpose(x, (3, 0, 1, 2))

    # scaling to [0, 255] is not necessary for tensorboard
    return x


def log_configuration(trainer):
    config_path = os.path.join(trainer.models_directory, "config.txt")
    architecture_path = os.path.join(trainer.models_directory, "architecture.txt")
    architecture_code = trainer.agents[0].get_architecture_code()
    configuration = {"gamma": trainer.gamma,
                     "n_players": trainer.n_players,
                     "learning_rate": trainer.lr,
                     "input_shape": trainer.agents[0].input_shape,
                     "batch_siz": trainer.agents[0].batch_siz,
                     "update_every": trainer.agents[0].update_every,
                    }
    json_object = json.dumps(configuration, indent=4)

    # jason configuration file
    with open(config_path, "w") as outfile:
        outfile.write(json_object)

    # network architecture code
    with open(config_path, "a") as outfile:
        outfile.write(architecture_code)
