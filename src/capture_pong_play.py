"""When run, this python file runs Pong and stores the observed games."""

import argparse

import numpy as np

from atari_env import gen_pong_env
from dataset import Dataset
from sample import sample
from utils import create_random_policy, save_object

def main():
    """
    Captures the transitions in episodes and writes them to disk.
    Uses a random agent.
    """
    parser = argparse.ArgumentParser(description=main.__doc__)
    parser.add_argument('--maxsteps', type=int, default=100000)
    parser.add_argument('--seed', type=int, default=0)
    parser.add_argument('--savefile', type=str, required=True)
    args = parser.parse_args()

    seed = args.seed
    env = gen_pong_env(seed)
    np.random.seed(seed)

    num_timesteps = 0
    paths = []
    while num_timesteps < args.maxsteps:
        print('{: 10d} of {: 10d} steps'.format(
            num_timesteps, args.maxsteps))
        path = sample(env, create_random_policy(env), num_paths=1)[0]
        paths.append(path)
        num_timesteps += len(path.obs)

    dataset = Dataset.from_paths(env, paths)
    print('Generated', len(dataset.obs), 'timesteps total')

if __name__ == "__main__":
    main()
