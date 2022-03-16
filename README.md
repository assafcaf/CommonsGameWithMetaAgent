
# Commons Game MultiagentDQN
 Multiagent task in CPR (common pool resources) using DQN algorithm
Using an existing environment from`https://github.com/tiagoCuervo/CommonsGame`
# Install Conda venv:

Download repository to local computer
1. Install Miniconda/Anaconda
	- Open conda command line
	- Create new conda venv using: `conda create --name <venv name> python=3.7`
	- Enter the environment we just created using: `conda activate <venv name>`

2. Install CommonesGame package
	- In conda command line go to `src\CommonsGame` where the project was saved using `cd` and then execute `pip install -e .`

3. Install Tensorflow
	- In conda command line execute: `conda install -c anaconda tensorflow-gpu=2.5.0`


# Run code from conda comman line:

First of all, make sure that conda command line is enabled on the environment we have created. If not, then use: `conda activate <venv name>`.

Then set conda path to where the project was saved using `cd` command (*.../CommonesGame/* ).

	- Training: In conda command line execute: `python src/scipts/train.py`
	- Rendering: In conda command line execute: `python src/scipts/load_and_render.py`
* the code designed to to run only from command line when the current directory is where the project was saved
# Analytics - Tnsorboard
First of all, make sure that conda command line is enabled on the environment we have created. If not, then use: `conda activate <venv name>`.
Then set conda path to where the project was saved using `cd`.

To activate Tensorboard:
1. In conda command line execute: `tensorboard --logdir logs`
2. Tensberboard is supposed to return something like:
	 
	"*2022-01-25 02:33:03.319349: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at `http://localhost:6006/` (Press CTRL+C to quit)"*

	The highlighted address should be copied to the browser
# GiF example:
![](/gifs/commons_game_my_dqn_dense4/model_dense4_players_20_ep_1_tr_295_rchance_0.gif)