"""This module contains code for performing (vectorized) rollouts."""

import numpy as np

from dataset import Path

def sample(env, policy, num_paths=10):
    """
    Generates a list of num_paths paths from running a (batching) policy
    on the given environment. Policy should accept a batch of observations
    and output a corresponding batch of actions.
    """

    paths = []

    # potential optimization: use a vectorized environment (per vlad17's
    # HW4 code)

    for _ in range(num_paths):
        obs = env.reset()
        path = Path(env, obs)
        done = False
        while not done:
            ac = policy(obs[np.newaxis, :])[0]
            obs, reward, done, _ = env.step(ac)
            path.next(obs, reward, ac)
        paths.append(path)
    return paths
