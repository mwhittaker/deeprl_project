"""
This file lets you play pong with the keyboard. Simply run this file and then
enter 0, 1, 2, 3, 4, or 5 on stdin to make an action.
"""

import argparse
import random

import tensorflow as tf
import numpy as np

import atari_env

def read_action_from_stdin():
    allowable_actions = ["0", "1", "2", "3", "4", "5"]
    action = input()
    while action not in allowable_actions:
        print("Error: enter one of {}.".format(allowable_actions))
        action = input()
    return int(action)

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--render', action="store_true")
    parser.add_argument('--seed', type=int, default=0)
    args = parser.parse_args()

    seed = args.seed
    tf.set_random_seed(seed)
    np.random.seed(seed)
    random.seed(seed)
    env = atari_env.gen_pong_env(args.seed)

    obs = env.reset()
    if args.render:
        env.render()

    done = False
    while not done:
        act = read_action_from_stdin()
        obs, reward, done, info = env.step(act)
        if args.render:
            env.render()

if __name__ == "__main__":
    main()