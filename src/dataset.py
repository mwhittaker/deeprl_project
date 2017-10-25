"""
The Path and Dataset contain the data that is operated on by all RL
agents.
"""
import os

import numpy as np
import h5py

from utils import get_ob_dim, get_num_acs

class Path:
    """
    A Path is a trace of the observations, actions, resulting
    observations, and rewards. The horizon can be expanded if the
    epsiode lasts longer. Assumes continuous Box observations
    and Discrete actions.
    """
    def __init__(self, env, initial_obs, horizon=8196):
        super().__init__()
        self._obs = np.empty((horizon,) + get_ob_dim(env))
        self._next_obs = np.empty((horizon,) + get_ob_dim(env))
        self._acs = np.empty((horizon,), dtype=int)
        self._rewards = np.empty(horizon)
        self._idx = 0
        self._horizon = horizon
        self._obs[0] = initial_obs

    def next(self, next_obs, reward, ac):
        """Add a new observation and reward resulting from an action to path"""
        if self._idx == self._horizon:
            self._resize()
        assert self._idx < self._horizon, (self._idx, self._horizon)
        self._next_obs[self._idx] = next_obs
        self._rewards[self._idx] = reward
        self._acs[self._idx] = ac
        self._idx += 1
        if self._idx < self._horizon:
            self._obs[self._idx] = next_obs

    def _resize(self):
        assert self._idx == self._horizon, (self._idx, self._horizon)

        self._horizon *= 2
        for arr in [self._next_obs, self._rewards, self._acs, self._obs]:
            arr.resize((self._horizon,) + arr.shape[1:], refcheck=False)
        self._obs[self._idx] = self._next_obs[self._idx - 1]

    @property
    def obs(self):
        """Time-indexed observations on path."""
        return self._obs[:self._idx]

    @property
    def acs(self):
        """Time-indexed actions on path."""
        return self._acs[:self._idx]

    @property
    def rewards(self):
        """Time-indexed rewards on path."""
        return self._rewards[:self._idx]

    @property
    def next_obs(self):
        """Time-indexed resulting observations on path."""
        return self._next_obs[:self._idx]


class Dataset:
    """Stores a set of paths, and (will) provide convient access to them."""

    def __init__(self, **kwargs):
        super().__init__()
        for k, v in kwargs.items():
            setattr(self, k, np.asarray(v))
        assert self._keys() == set(kwargs.keys())

    @staticmethod
    def _keys():
        return {'ob_dim', 'num_acs', 'obs', 'next_obs', 'rewards',
                'acs', 'ep_lens'}

    @staticmethod
    def from_paths(env, paths):
        """Generate a Dataset from paths."""
        kwargs = {
            'obs': np.concatenate([path.obs for path in paths]),
            'next_obs': np.concatenate([path.next_obs for path in paths]),
            'rewards': np.concatenate([path.rewards for path in paths]),
            'acs': np.concatenate([path.acs for path in paths]),
            'ep_lens': [len(path.obs) for path in paths],
            'ob_dim': get_ob_dim(env),
            'num_acs': get_num_acs(env)}
        return Dataset(**kwargs)

    def save(self, savefile):
        """Save a dataset to disk in h5py format"""
        with h5py.File(os.path.join(savefile), 'w') as f:
            ds = f.create_group('dataset')
            for attr_name in Dataset._keys():
                attr = getattr(self, attr_name)
                ds.create_dataset(attr_name, data=attr)

    @staticmethod
    def load(savefile):
        """Recover a saved dataset"""
        with h5py.File(os.path.join(savefile), 'r') as f:
            ds = f.require_group('dataset')
            kwargs = {attr_name: ds[attr_name][()]
                    for attr_name in Dataset._keys()}
        return Dataset(**kwargs)
