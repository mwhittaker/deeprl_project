# Fall 2017 Deep RL Final Project

[Final Project Assignment](https://d1b10bmlvqabco.cloudfront.net/attach/j6l2zpz570w7jq/iy4vn27h37x7h4/j711skxb7k4n/final_project.pdf)

You can check style with `pylint --disable=locally-disabled,fixme src`.

Our dependencies require Python 3.5 (and assume appropriate GPU drivers have already been installed).

    conda create -y -n gpu-py3.5 python=3.5 anaconda
    source activate gpu-py3.5
    conda install -y gcc # prereq for numpy upgrade
    conda upgrade -y numpy # upgraded numpy is a prereq for gym
    pip install tensorflow-gpu # or tensorflow for CPU only
    pip install -r requirements.txt

Installing OpenAI baselines

    git clone https://github.com/openai/baselines.git
    cd baselines
    git checkout 4993286230ac92ead39a66005b7042b56b8598b0
    sudo apt install -y mpich # see Mac OS version below
    env MPICC=/usr/bin/mpicc pip install mpi4py # see Mac OS version below
    pip install --no-dependencies -e .
    cd ..

    ## Mac OS
    brew install mpich
    env MPICC=/usr/local/Cellar/mpich/3.2_3/bin/mpicc pip install mpi4py
    # (if that fails, use 'sudo find / -name mpicc' to find where MPICC is located + substitute accordingly)

Train PPO agent

    cd src
    python run_atari_ppo.py [--max_timesteps=1]
