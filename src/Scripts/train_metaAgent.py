from datetime import date
from Danfoa_CommonsGame.code.CommonsGame.game_environment.commons_env import HarvestCommonsEnv
from Danfoa_CommonsGame.code.CommonsGame.game_environment.constants import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP
from Danfoa_CommonsGame.code.PolicyGradientTrainerWithMetaAgent.trainer import TrainerWithMetaAgent as Trainer
import os
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
# tensorflow gpu-memory usage
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.4)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# init params
n_players = 5
ep_length = 750
lr = 1e-3
learn_every = 1
act_every = 25
render_every = 25
gamma = 0.9

# init env
env = HarvestCommonsEnv(ascii_map=MEDIUM_HARVEST_MAP, num_agents=n_players)
state_dim = env.state_dim[:-1] + (1,)

# set outputs directories
num_actions = env.action_space.n
folders_name = f"{date.today().strftime('%Y-%m-%d')}_withMeta_agents_{n_players}_notNormalize"
models_directory = os.path.join(os.getcwd(), os.pardir, os.pardir, "models", folders_name)
log_dir = os.path.join(os.getcwd(), os.pardir, os.pardir, "logs", "meta", folders_name)
input_shape = env.observation_space.shape[:-1] + (1,)

# build model
trainer = Trainer(input_shape=input_shape,
                  num_actions=num_actions,
                  n_players=n_players,
                  ep_length=ep_length,
                  models_directory=models_directory,
                  lr=lr,
                  gamma=gamma,
                  state_dim=state_dim,
                  act_every=act_every,
                  render_every=render_every,
                  log_dir=log_dir)

# start training
trainer.train(env)

