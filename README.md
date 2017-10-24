# Fall 2017 Deep RL Final Project

[Final Project Assignment](https://d1b10bmlvqabco.cloudfront.net/attach/j6l2zpz570w7jq/iy4vn27h37x7h4/j711skxb7k4n/final_project.pdf)

Dependencies can be installed with `python setup.py install` (and TensorFlow).

You can check style with `pylint --disable=locally-disabled,fixme src`.

Our dependencies require Python 3.5 (and assume appropriate GPU drivers have already been installed).

    conda create -y -n gpu-py3.5 python=3.5 anaconda
    source activate gpu-py3.5
    conda install -y gcc # prereq for numpy upgrade
    conda upgrade -y numpy # upgraded numpy is a prereq for gym
