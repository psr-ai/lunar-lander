# Lunar Lander (CS221 Project)

This project focuses on using reinforcement learning to achieve highest score in [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/).
This is a part of our coursework at SCPD course CS221 at Stanford University. Visit the [website](http://www.prabhjotrai.com/lunar-lander) to check some videos, and view the [worksheet](https://worksheets.codalab.org/worksheets/0x1e3fc24cfa0d4ff3b492d0f47b6e0887/) on codalab.

### Setup Instructions (For Mac):

1. Make sure you have latest version of virtual env, `pip install --upgrade virtualenv`
2. Make sure `python3` is installed on your machine. To check, type `python3`. If not available, setup from [here](https://www.python.org/downloads/).
3. Create a virtual env by `virtualenv -p python3 venv` at the root of this repo (can be anywhere else too, but make sure not to commit this folder). This will create a directory `venv`, containing default dependencies for python3. If you create this folder at the root of this repo, it will be ignored by git before pushing.
4. Activate virtual env by `source venv/bin/activate`. This will activate the virtual env. Running python3 will open console with default dependencies from python.
5. Install dependencies for this project by `pip3 install -r requirements.txt --no-cache`. This will install all the dependencies inside the python virtual environment.
6. Note: Pip will have installed matpltlib, there's an issue with matpltlib with mac os. Follow the steps [here](https://stackoverflow.com/a/21789908/5159284) to fix this issue.

You are ready! We recommend using [pycharm community edition](https://www.jetbrains.com/pycharm/download/#section=mac). Once installed, follow the following steps to point project interpreter for pycharm:

1. Go to `Preferences`, by shortcut `Cmd + ,` or in Menu Bar `PyCharm > Preferences...`
2. Go to `Project Interpreter` under `Project:lunar-lander`
3. Click on settings cog wheel on right side of top Project Interpreter path specification and click on `Add...`. Choose existing environment and add the path of the one you created. [Here's](https://www.jetbrains.com/help/pycharm/creating-virtual-environment.html) detailed description of how to configure venv in pycharm.

### Setup Instructions (For Debian):

1. Prior to any step, please make sure you have [dependencies](https://github.com/openai/gym#installing-everything) specific to debian machine for installing gym. Specifically for debian, make sure the dependencies below are installed by running the command

````
apt-get install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev \
        libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev \
        freeglut3 xvfb libav-tools
````
2. Follow steps 1 to 4 in the setup instructions for mac above.
3. Install dependencies for this project by `pip3 install -r requirements-debian.txt --no-cache`. This will install all the dependencies inside the python virtual environment.
4. Note: Pip will have installed matpltlib, there's an issue with matpltlib with mac os. Follow the steps [here](https://stackoverflow.com/a/21789908/5159284) to fix this issue.

### Running instructions:

#### Simulator to play the code

To run the simulator to play the game: `python3 play/keyboard_agent.py`. Press key `1` to rotate left, `2` to thrust upwards and `3` to rotate right.

#### Deep Q-Learning

The main.py file can be run through command line interface from the root of the directory. There are a number of parameters which can be controlled through arguments of the run command.

##### Example running commands:

To run the DDQN agent on given weights (`implementation/experiments/DoubleDQN_Set4/weights_0750.hdf5`):

`python3 implementation/dqn/main.py --experiment_name=DoubleDQN_Set4 --agent=DDQN --should_learn=False --should_render=False --initial_weights=weights_0750.hdf5`

To run the DDQN agent and make it learn and output the weights to `new_experiment` directory:

`python3 implementation/dqn/main.py --experiment_name=new_experiment --agent=DDQN --should_learn=True --should_render=False`

To run the DDQN agent on `CartPole-v1` environment:

`python3 implementation/dqn/main.py --experiment_name=new_experiment --agent=DDQN --should_learn=True --should_render=False --problem=CartPole-v1`

##### Parameters description:

In order to change the hyperparameters, please visit the `implementation/dqn/hyperparameters.py` file directly and modify it. For the rest of the parameters,
they can be controlled directly from the command line: 

1. `--agent` (String): Specifies which learning agent you would like to use. The values can be specified from `FullDQN`, `DDQN` , `Dueling` and `Linear`. For example, if you want the agent to use Dueling Network Architecture learning approach, then specify `--agent=Dueling`.
Default is `DDQN`.

2. `--gpu` (Integer): Specifies which GPU to use, if you have multiple. For example: `--gpu=1`. Default is `0`.

3. `--num_episodes` (Integer): Specifies the number of episodes. Default is `800`.

4. `--should_render` (Boolean): Whether you want to render your episode. Default is `False`.

5. `--should_learn` (Boolean): Whether the agent should learn or should just exploit on given weights. Default is `True`.

6. `--experiment_name` (String): Unique name for your experiment. This will create a directory by this name in the experiments/ directory, which will hold all data related to this experiment. Default is `NewExperiment`.
When exploring, you should give an experiment name corresponding to existing folder name under `implementation/experiments`.

7. `--problem` (String): Can be any problem in open AI gym which is having vector based state description. Default is `lunar-lander` and we tried it on `CartPole-v1` too.

8. `--initial_weights` (String): Name of the weights the model should use. Default is `weights_0750.hdf5`.

#### Important Note

When the agent is not learning and only exploiting (`--should_learn=False`), by default, the `--experiment_name` should end with `Set4`, precisely being: `--experiment_name=[agent-type]_Set4`.
 This is because the hyperparameters are set to this set's configuration. Of course, feel free to change the hyperparameters to other sets, as described in `final.pdf`. 

#### Baselines

Baselines can be run directly from the `implementation/baselines` directory. For example, run `python3 implementation/baselines/random_baseline.py` to run the random baseline.

### Contributing:

Please follow the following guidelines to contribute.

1. Do not commit on `master` directly, although everyone has permission.
2. Create a new branch from `master`, make changes and open a PR.
3. Get it reviewed by contributors and we merge it in collaboration.

### Contributors:

Abhishek Bharani, Amey Naik and [Prabhjot Singh Rai](https://www.prabhjotrai.com).
