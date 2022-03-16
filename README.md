
# Commons Game MultiAgent environment
Multiagent task in CPR (common pool resources) 
Using an existing environment from`https://github.com/Danfoa/commons_game`

# Install Conda venv:

1. Install Miniconda/Anaconda
	- Open conda command line
	- Create new conda venv using: `conda create --name <venv name> python=3.9`
	- Enter the environment we just created using: `conda activate <venv name>`

2. Install Tensorflow GPU
	- Tensorflow: `conda install -c anaconda tensorflow-gpu`
3. Install requirements
	- Tensorboard:  `conda install -c conda-forge tensorboard`
	- gym:  `conda install -c conda-forge gym`
	- cv2:  `conda install -c conda-forge opencv`
	- tqdm:  `conda install -c conda-forge tqdm`


# Run code from conda comman line:
* todo ....


# Analytics - Tensorboard
First of all, make sure that conda command line is enabled on the environment we have created. If not, then use: `conda activate <venv name>`.
Then set conda path to where the project was saved using `cd`.

To activate Tensorboard:
1. In conda command line execute: `tensorboard --logdir logsMeta`
2. Tensberboard is supposed to return something like:
	 
	"*2022-01-25 02:33:03.319349: I tensorflow/stream_executor/platform/default/dso_loader.cc:53] Successfully opened dynamic library cudart64_110.dll
Serving TensorBoard on localhost; to expose to the network, use a proxy or pass --bind_all
TensorBoard 2.5.0 at `http://localhost:6006/` (Press CTRL+C to quit)"*

	The highlighted address should be copied to the browser
