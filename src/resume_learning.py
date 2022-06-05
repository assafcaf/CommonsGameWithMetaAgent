#!/usr/bin/env python3
import os
import shutil
import warnings
import tensorflow as tf
from datetime import date
from tensorflow.core.util import event_pb2
from tensorflow.python.lib.io import tf_record
from tensorflow.python.framework import tensor_util
from CommonsGame3.commons_env import MapEnv  # my env
from CommonsGame3.constants import *  # my env
from CommonsGame3.constants import HARVEST_MAP
from DQN.trainer import TrainerDQNAgentFromPrevTraining as Trainer
warnings.filterwarnings("ignore")

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


def my_summary_iterator(path):
    for r in tf_record.tf_record_iterator(path):
        yield event_pb2.Event.FromString(r)


if __name__ == '__main__':
    writer_path = r"C:\studies\IDC_dataScience\thesis\Danfoa_CommonsGame\logs\DQN\2022-05-21_noMeta_6_DDQN_conv_bigMap_8Agents"
    event_filename = os.listdir(writer_path)[0]
    prev_log_path = os.path.join(writer_path, event_filename)
    run_name = writer_path.split("\\")[-1]
    current_log_path = os.path.join("\\".join(writer_path.split("\\")[:-1]), "Resume_"+run_name)
    step_mod = -1
    step = -1
    eps = 1
    print("Starting copy log file")
    if os.path.isdir(current_log_path):
        shutil.rmtree(current_log_path)
    w = tf.summary.create_file_writer(current_log_path)
    with w.as_default() as writer:
        for event in my_summary_iterator(prev_log_path):
            step = event.step
            for value in event.summary.value:
                t = tensor_util.MakeNdarray(value.tensor)
                if event.step % 100 == 0 and step_mod != event.step:
                    step_mod = event.step
                    print(f"{step_mod} steps hase copied")
                if value.tag != "batch_loss":
                    tf.summary.scalar(name=value.tag, data=t, step=event.step)
                    if value.tag == "epsilon":
                        eps = t
    print(f"Finished copy log file with {step} steps, epsilon {eps}")

    # trainer params
    n_players = 6
    ep_length = 750
    lr = 2.5e-5
    render_every = 50
    gamma = 0.99
    buffer_size = int(1e5)
    batch_size = 32
    epsilon_decay_rate = 0.9999

    update_every = 25
    min_to_learn = 10
    min_epsilon = 0.1
    seq_len = 1

    env = MapEnv(bass_map=HARVEST_MAP, num_agents=n_players, color_map=DEFAULT_COLOURS, agents_vision=21)
    num_actions = env.action_space_n
    gifs = os.path.join(os.getcwd(), os.pardir, "gifs", run_name)
    models_directory = os.path.join(os.getcwd(), os.pardir, "models", run_name)
    input_shape = env.observation_space_shape

    # build model
    trainer = Trainer(input_shape=input_shape,
                      num_actions=num_actions,
                      n_players=n_players,
                      ep_length=ep_length,
                      models_directory=models_directory,
                      lr=lr,
                      gifs=gifs,
                      update_every=update_every,
                      gamma=gamma,
                      min_to_learn=min_to_learn,
                      min_epsilon=min_epsilon,
                      render_every=render_every,
                      log_dir=current_log_path,
                      buffer_size=buffer_size,
                      batch_size=batch_size,
                      epsilon_decay_rate=epsilon_decay_rate,
                      eps0=eps,
                      writer=w)

    # start training
    trainer.train(env, step)





