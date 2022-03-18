from datetime import date
from Danfoa_CommonsGame.code.CommonsGame.game_environment.commons_env import HarvestCommonsEnv
from Danfoa_CommonsGame.code.CommonsGame.game_environment.constants import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP
from Danfoa_CommonsGame.code.PolicyGradientTrainerWithMetaAgent.trainer import TrainerNoMetaAgent as Trainer
import os
import warnings
import tensorflow as tf

# tensorflow gpu usage
warnings.filterwarnings("ignore")
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.5)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# init params
n_players = 5
ep_length = 750
lr = 0.0001
learn_every = 1
render_every = 100

# init env
env = HarvestCommonsEnv(ascii_map=MEDIUM_HARVEST_MAP, num_agents=n_players)
state_dim = env.state_dim

# set outputs directories
num_actions = env.action_space.n
folders_name = f"{date.today().strftime('%Y-%m-%d')}_no_meta_dense_256x2_{n_players}_agents"
models_directory = os.path.join(os.getcwd(), os.pardir, os.pardir, "models", folders_name)
log_dir = os.path.join(os.getcwd(), os.pardir, os.pardir, "logs", "without_meta", folders_name)
input_shape = env.observation_space.shape

# build model
meta_agent = Trainer(input_shape=input_shape,
                     num_actions=num_actions,
                     n_players=n_players,
                     ep_length=ep_length,
                     models_directory=models_directory,
                     lr=lr,
                     state_dim=state_dim,
                     render_every=render_every,
                     log_dir=log_dir)

# start training
meta_agent.train(env)
