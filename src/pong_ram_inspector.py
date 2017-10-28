"""This file helps you figure out the meaning of bytes in Pong's RAM."""

import argparse
import collections
import itertools
import random
import time

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf

import atari_env
import utils

def _delayed_policy(policy, delay):
    def _policy(obs):
        time.sleep(delay)
        return policy(obs)
    return _policy

def _get_parser():
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--seed",
        type=int,
        default=0,
    )
    parser.add_argument(
        "--track",
        type=int,
        help="The byte in RAM to track.",
    )
    parser.add_argument(
        "--keyboard",
        action="store_true",
        help="Read actions from stdin instead of using a random policy",
    )
    parser.add_argument(
        "--render",
        action="store_true",
        help="Render the environment",
    )
    parser.add_argument(
        "--delay_random_policy",
        action="store_true",
        help="Pause briefly after every random policy action."
    )
    parser.add_argument(
        "--compute_change_frequencies",
        action="store_true",
        help="Compute the frequency with which bytes in RAM change",
    )

    return parser

# pylint: disable=too-many-branches
def main():
    """Run pong and inspect the RAM."""
    parser = _get_parser()
    args = parser.parse_args()

    # Generate the environment.
    tf.set_random_seed(args.seed)
    np.random.seed(args.seed)
    random.seed(args.seed)
    env = atari_env.gen_pong_ram_env(args.seed, cherry_picked=False)

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

    # We track the value of the byte specified by --track across timesteps.
    steps = []
    ram_values = []

    obs = env.reset()
    if args.render:
        env.render()
    done = False
    for step in itertools.count(0):
        # Take an action.
        old_obs = obs
        action = policy(np.array([obs]))[0]
        obs, _reward, done, _info = env.step(action)

        # Track RAM changes.
        if args.compute_change_frequencies:
            for i in range(obs.shape[0]):
                if old_obs[i] != obs[i]:
                    num_changes[i] += 1

        if args.track is not None:
            steps.append(step)
            ram_values.append(obs[args.track])

        if args.render:
            env.render()

        if done:
            break

    if args.compute_change_frequencies:
        for (k, v) in sorted(num_changes.items(), key=lambda p: -p[1]):
            print("{}: {}".format(k, v))

    if args.track is not None:
        plt.figure()
        plt.plot(steps, ram_values)
        plt.savefig("ram_{}_seed_{}.pdf".format(args.track, args.seed))

if __name__ == "__main__":
    main()
