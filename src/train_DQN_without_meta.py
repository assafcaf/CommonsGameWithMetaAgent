#!/usr/bin/env python3
from datetime import date
from CommonsGame3.commons_env import MapEnv  # my env
from CommonsGame3.constants import HARVEST_MAP, DEFAULT_COLOURS
from DQN.trainer import TrainerDQNAgent as Trainer
import os
import warnings
import tensorflow as tf

# tensorflow gpu-memory usage
gpus = tf.config.list_physical_devices('GPU')
if gpus:
    # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
    try:
        tf.config.set_logical_device_configuration(
            gpus[0],
            [tf.config.LogicalDeviceConfiguration(memory_limit=4*1024)])
        logical_gpus = tf.config.list_logical_devices('GPU')
        print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
    except RuntimeError as e:
        # Virtual devices must be set before GPUs have been initialized
        print(e)


if __name__ == '__main__':
    # set up directories
    n_players = 6
    directory = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame"
    model_name = f"{date.today().strftime('%Y-%m-%d')}_noMeta_{n_players}_DDQN_conv_bigMap_8Agents"
    gifs = os.path.join(directory, "gifs", model_name)
    models_directory = os.path.join(directory, "models", model_name)
    log_dir = os.path.join(directory, "logs", "DQN", model_name)

    # init learning params
    ep_length = 750
    lr = 1e-5
    render_every = 100
    gamma = 0.995
    buffer_size = int(1e5)
    batch_size = 32
    epsilon_decay_rate = 0.9997
    update_every = 25
    min_to_learn = 10
    min_epsilon = 0.15

    # init env
    env = MapEnv(bass_map=HARVEST_MAP, num_agents=n_players, color_map=DEFAULT_COLOURS, agents_vision=21)
    num_actions = env.action_space_n

    # build model
    trainer = Trainer(input_shape=env.observation_space_shape, num_actions=num_actions, n_players=n_players,
                      ep_length=ep_length,  models_directory=models_directory, lr=lr, gifs=gifs,
                      update_every=update_every, gamma=gamma, min_to_learn=min_to_learn, min_epsilon=min_epsilon,
                      render_every=render_every, log_dir=log_dir, buffer_size=buffer_size, batch_size=batch_size,
                      epsilon_decay_rate=epsilon_decay_rate)

    trainer.setup_results_directories()
    # start training
    trainer.train(env)

