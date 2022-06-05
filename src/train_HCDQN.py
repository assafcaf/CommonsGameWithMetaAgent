#!/usr/bin/env python3
import os
import tensorflow as tf
from datetime import date
from CommonsGame3.commons_env import MapEnv  # my env
from CommonsGame3.constants import HARVEST_MAP, DEFAULT_COLOURS
from HLC.agent.multiAgent import MultiAgent

# handling errors
tf.compat.v1.logging.set_verbosity(tf.compat.v1.logging.ERROR)


# tensorflow gpu-memory usage
def limit_gpu():
    gpus = tf.config.list_physical_devices('GPU')
    if gpus:
        # Restrict TensorFlow to only allocate 1GB of memory on the first GPU
        try:
            tf.config.set_logical_device_configuration(
                gpus[0],
                [tf.config.LogicalDeviceConfiguration(memory_limit=6 * 1024)])
            logical_gpus = tf.config.list_logical_devices('GPU')
            print(len(gpus), "Physical GPUs,", len(logical_gpus), "Logical GPUs")
        except RuntimeError as e:
            # Virtual devices must be set before GPUs have been initialized
            print(e)
# limit_gpu()



if __name__ == '__main__':
    # init training params
    ep_length = 750
    n_agents = 9
    lr = 1e-4
    agents_vision = 11
    channels = 3
    input_shape = (agents_vision, agents_vision, channels)
    replay_start_size = int(1e6)
    save_every = 500
    model_layer = "conv"
    # set up directories
    out_put_directory = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame\logs\HLCDDQN"
    model_name = f"{date.today().strftime('%Y-%m-%d')}_{n_agents}_{model_layer}_agents_{str(lr)}lr_HLCDDQN"

    # init env
    env = MapEnv(bass_map=HARVEST_MAP, num_agents=n_agents, color_map=DEFAULT_COLOURS,
                 agents_vision=agents_vision, gray_scale=False)
    num_actions = env.action_space_n

    # build model
    trainer = MultiAgent(input_shape=input_shape,
                         num_actions=num_actions,
                         ep_steps=ep_length,
                         agent_history_length=channels,
                         model_name=model_name,
                         num_agents=n_agents,
                         lr=lr, save_weight_interval=save_every,
                         runnig_from=out_put_directory)

    # start training
    trainer.train_no_history(env)
