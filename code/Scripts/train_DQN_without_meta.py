from datetime import date
from Danfoa_CommonsGame.code.CommonsGame.game_environment.commons_env import HarvestCommonsEnv
from Danfoa_CommonsGame.code.CommonsGame.game_environment.constants import SMALL_HARVEST_MAP, MEDIUM_HARVEST_MAP
from Danfoa_CommonsGame.code.PolicyGradientTrainerWithMetaAgent.trainer import TrainerDQNNoMetaAgent as Trainer
import os
import warnings
import tensorflow as tf

warnings.filterwarnings("ignore")
# tensorflow gpu-memory usage
gpu_options = tf.compat.v1.GPUOptions(per_process_gpu_memory_fraction=0.6)
sess = tf.compat.v1.Session(config=tf.compat.v1.ConfigProto(gpu_options=gpu_options))

# init params
n_players = 5
ep_length = 750
lr = 1e-4
learn_every = 1
render_every = 500
gamma = 0.99
buffer_size = int(1e5)
batch_size = 64
epsilon_decay_rate = 0.9975
update_every = 25
min_to_learn = 50

# init env
env = HarvestCommonsEnv(ascii_map=MEDIUM_HARVEST_MAP, num_agents=n_players)
state_dim = env.state_dim

# set outputs directories
num_actions = env.action_space.n
folders_name = f"{date.today().strftime('%Y-%m-%d')}_withOutMeta_agents_{n_players}_DQN"
models_directory = os.path.join(os.getcwd(), os.pardir, os.pardir, "models", folders_name)
log_dir = os.path.join(os.getcwd(), os.pardir, os.pardir, "logs", "DQN", folders_name)
input_shape = env.observation_space.shape

# build model
trainer = Trainer(input_shape=input_shape,
                  num_actions=num_actions,
                  n_players=n_players,
                  ep_length=ep_length,
                  models_directory=models_directory,
                  lr=lr,
                  update_every=update_every,
                  gamma=gamma,
                  min_to_learn=min_to_learn,
                  render_every=render_every,
                  log_dir=log_dir,
                  buffer_size=buffer_size,
                  batch_size=batch_size,
                  epsilon_decay_rate=epsilon_decay_rate)

# start training
trainer.train(env)

