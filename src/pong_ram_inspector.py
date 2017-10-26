"""This file helps you figure out the meaning of bytes in Pong's RAM."""

import argparse
import collections
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
    """Run pong and inspect the RAM."""
    parser = argparse.ArgumentParser()
    parser.add_argument("--seed", type=int, default=0)
    parser.add_argument("--keyboard", action="store_true")
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--delay_random_policy", action="store_true")
    parser.add_argument("--track_changes", action="store_true")
    args = parser.parse_args()

    # Generate the environment.
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = atari_env.gen_pong_ram_env(args.seed)

    # Generate the policy.
    if args.keyboard:
        policy = utils.create_keyboard_policy(env)
    else:
        policy = utils.create_random_policy(env)
        if args.delay_random_policy:
            policy = _delayed_policy(policy, 0.1)

    # For each byte in RAM, we see how many times its value changes throughout
    # the course of a game.
    num_changes = collections.defaultdict(int)

    obs = env.reset()
    if args.render:
        env.render()
    done = False
    while not done:
        # Take an action.
        old_obs = obs
        action = policy(np.array([obs]))[0]
        obs, _reward, done, _info = env.step(action)

        # Track RAM changes.
        if args.track_changes:
            for i in range(obs.shape[0]):
                if old_obs[i] != obs[i]:
                    num_changes[i] += 1

        if args.render:
            env.render()

    if args.track_changes:
        for (k, v) in sorted(num_changes.items(), key=lambda p: -p[1]):
            print("{}: {}".format(k, v))

if __name__ == "__main__":
    main()
