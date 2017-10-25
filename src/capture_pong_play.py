"""When run, this python file runs Pong and stores the observed games."""

import argparse
from multiprocessing import cpu_count

import numpy as np

from atari_env import gen_vectorized_pong_env
from dataset import Dataset
from sample import vsample
from utils import create_random_policy

def main():
    """
    Captures the transitions in episodes and writes them to disk.
    Uses a random agent.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--maxsteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savefile', type=str, required=True)
    nproc = max(cpu_count() - 1, 1)
    parser.add_argument('--maxprocs', type=int, default=nproc)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)
    venv = gen_vectorized_pong_env(args.maxprocs)
    policy = create_random_policy(venv)

    num_timesteps = 0
    paths = []
    while num_timesteps < args.maxsteps:
        print('{: 10d} of {: 10d} steps'.format(
            num_timesteps, args.maxsteps))
        new_paths = vsample(venv, policy)
        paths += new_paths
        num_timesteps += sum(len(path.obs) for path in new_paths)

    dataset = Dataset.from_paths(venv, paths)
    print('Generated', len(dataset.obs), 'timesteps total')
    dataset.save(args.savefile)

if __name__ == "__main__":
    main()
