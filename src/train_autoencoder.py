"""Train the autoencoder on a dataset of images."""

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
    parser.add_argument('--datafile', type=str, required=True)
    args = parser.parse_args()

    seed = args.seed
    np.random.seed(seed)

    dataset = Dataset.load(args.datafile)
    
    

if __name__ == "__main__":
    main()
