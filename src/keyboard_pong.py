"""
This file lets you play pong with the keyboard. Simply run this file and then
enter 0, 1, 2, 3, 4, or 5 on stdin to make an action.
"""

import argparse
import random
import time

import tensorflow as tf
import numpy as np

import atari_env
import utils

def _delayed_policy(policy, delay):
    def _policy(obs):
        time.sleep(delay)
        return policy(obs)
    return _policy

def main():
    """Plays pong with the keyboard."""
    parser = argparse.ArgumentParser()
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--random_policy', action="store_true")
    args = parser.parse_args()

    seed = args.seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = atari_env.gen_pong_env(args.seed)
    if args.random_policy:
        random_policy = utils.create_random_policy(env)
        policy = _delayed_policy(random_policy, 0.1)
    else:
        policy = utils.create_keyboard_policy(env)

    obs = env.reset()
    env.render()
    done = False
    while not done:
        action = policy(np.array([obs]))[0]
        obs, _reward, done, _info = env.step(action)
        env.render()

if __name__ == "__main__":
    main()
