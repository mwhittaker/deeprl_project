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

def vsample(venv, policy):
    """
    Samples from a vectorized environment, performing the rollouts
    simultaneously.
    """
    obs_n = venv.reset()
    paths = [Path(venv, obs) for obs in obs_n]
    n = len(paths)
    done_n = np.zeros(n, dtype=bool)
    while done_n.sum() < n:
        active = np.flatnonzero(~done_n)
        acs_active = policy(np.asarray(obs_n)[active])
        acs_n = np.full(n, None)
        acs_n[active] = acs_active
        obs_n, reward_n, new_done_n, _ = venv.step(acs_n)
        for i in active:
            paths[i].next(obs_n[i], reward_n[i], acs_n[i])
            if new_done_n[i] and not done_n[i]:
                venv.mask(i)
                done_n[i] = True
    return paths
