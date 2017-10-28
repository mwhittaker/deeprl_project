"""When run, this python file runs Pong and stores the observed games."""

import argparse
import random
from multiprocessing import cpu_count

import numpy as np

from atari_env import (gen_pong_video_env, gen_pong_ram_env)
from dataset import Dataset
from sample import (sample, vsample)
from utils import create_random_policy

def _get_parser():
    parser = argparse.ArgumentParser(description=main.__doc__)

    parser.add_argument('--maxsteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savefile', type=str, required=True)
    nproc = max(cpu_count() - 1, 1)
    parser.add_argument('--maxprocs', type=int, default=nproc)
    types = ["video", "ram", "cherry_picked"]
    parser.add_argument('--pong_type', choices=types, default="video")

    return parser

def main():
    """
    Captures the transitions in episodes and writes them to disk.
    Uses a random agent.
    """
    parser = _get_parser()
    args = parser.parse_args()

    # Generate the environment.
    np.random.seed(args.seed)
    random.seed(args.seed)
    if args.pong_type == "video":
        env = gen_pong_video_env(args.seed, vectorized_n=args.maxprocs)
    elif args.pong_type == "ram":
        env = gen_pong_ram_env(
            args.seed,
            cherry_picked=False,
            vectorized_n=args.maxprocs)
    elif args.pong_type == "cherry_picked":
        env = gen_pong_ram_env(
            args.seed,
            cherry_picked=True,
            vectorized_n=args.maxprocs)
    else:
        raise ValueError("Illegal pong type {}.".format(args.pong_type))

    # Run the environment.
    paths = []
    policy = create_random_policy(env)
    num_timesteps = 0
    while num_timesteps < args.maxsteps:
        print('{: 10d} of {: 10d} steps'.format(num_timesteps, args.maxsteps))
        sample_fun = sample if args.maxprocs == 1 else vsample
        new_paths = sample_fun(env, policy)
        paths += new_paths
        num_timesteps += sum(len(path.obs) for path in new_paths)

    # Save the dataset.
    dataset = Dataset.from_paths(env, paths)
    print('Generated', len(dataset.obs), 'timesteps total.')
    print('Saving dataset to {}.'.format(args.savefile))
    dataset.save(args.savefile)

if __name__ == "__main__":
    main()
