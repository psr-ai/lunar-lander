# Lunar Lander (CS221 Project)

This project focuses on using reinforcement learning to achieve highest score in [LunarLander-v2](https://gym.openai.com/envs/LunarLander-v2/).
This is a part of our coursework at SCPD course CS221 at Stanford University.

### Codalab Instructions:

### Setup Instructions:

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

### Google Cloud:

1. Prior to any step, please make sure you have [dependencies](https://github.com/openai/gym#installing-everything) specific to debian machine for installing gym.

apt-get install libjpeg-dev cmake swig python-pyglet python3-opengl libboost-all-dev \
        libsdl2-2.0.0 libsdl2-dev libglu1-mesa libglu1-mesa-dev libgles2-mesa-dev \
        freeglut3 xvfb libav-tools
        
2. Use `requirements-google-cloud.txt` to install requirements specific to debian.

### Contributing:

Please follow the following guidelines to contribute.

1. Do not commit on `master` directly, although everyone has permission.
2. Create a new branch from `master`, make changes and open a PR.
3. Get it reviewed by contributers and we merge it in collaboration.

### Contributors:

Abhishek Bharani, Amey Naik and [Prabhjot Singh Rai](www.github.com/raiprabh).
