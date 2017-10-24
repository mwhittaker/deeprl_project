"""When run, this python file runs Pong and stores the observed games."""

import argparse

import tensorflow as tf
import numpy as np

from atari_env import gen_pong_env

def main():
    """Captures the transitions in episodes and writes them to disk."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--maxsteps', type=int, default=6000000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savefile', type=str, default='')
    args = parser.parse_args() # pylint: disable=unused-variable

    tf_config = tf.ConfigProto()
    tf_config.gpu_options.allow_growth = True
    session = tf.Session(config=tf_config) # pylint: disable=unused-variable

    seed = 0
    env = gen_pong_env(seed)
    tf.set_random_seed(seed)
    np.random.seed(seed)
    print(env.reset().shape)

if __name__ == "__main__":
    main()
