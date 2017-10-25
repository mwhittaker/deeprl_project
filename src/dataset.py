"""
The Path and Dataset contain the data that is operated on by all RL
agents.
"""

import numpy as np

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

    def __init__(self, ob_dim, num_acs, obs, next_obs, rewards, acs, ep_lens):
        super().__init__()
        self.ob_dim = ob_dim
        self.num_acs = num_acs
        self.ep_lens = ep_lens
        self.obs = obs
        self.next_obs = next_obs
        self.rewards = rewards
        self.acs = acs

    @staticmethod
    def from_paths(env, paths):
        """Generate a Dataset from paths."""
        obs = np.concatenate([path.obs for path in paths])
        next_obs = np.concatenate([path.next_obs for path in paths])
        rewards = np.concatenate([path.rewards for path in paths])
        acs = np.concatenate([path.acs for path in paths])
        ep_lens = np.array([len(path.obs) for path in paths])

        return Dataset(
            get_ob_dim(env), get_num_acs(env), obs, next_obs, rewards,
            acs, ep_lens)
