"""Wrappers for Atari playing, adopted from Deep RL HW3 starter code."""

import gym
import numpy as np

from multiprocessing_env import MultiprocessingEnv
from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
from baselines.common.atari_wrappers import (NoopResetEnv, FireResetEnv,
                                             EpisodicLifeEnv, MaxAndSkipEnv,
                                             ClipRewardEnv)

class _CherryPickedPongRamWrapper(gym.ObservationWrapper):
    """
    The Atari 2600 has 128 bytes of RAM. Thus, the Pong-ram-v0 environment's
    observation space includes numpy arrays of 128 bytes. Only some of these
    bytes are relevant to learning how to play Pong. Some are 0, some are
    constant, and only a handful actually change value over the course of a
    game. This wrapper extracts a set of hand-picked RAM values that seem to be
    important. The values were selected manually by using
    `pong_ram_inspector.py` to trace the values of RAM throughout the course of
    a game.
    """
    def __init__(self, env):
        super(_CherryPickedPongRamWrapper, self).__init__(env)
        self.indexes = [51, 50, 21, 60, 49, 4, 54, 11, 121, 12]
        n = len(self.indexes)
        low = np.zeros(n)
        high = np.zeros(n) + 255.0
        self.observation_space = spaces.Box(low, high)

    def _observation(self, observation):
        return observation[self.indexes]

def _wrap_deepmind_ram(env):
    """Applies various Atari-specific wrappers to make learning easier."""
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    return env

def wrap_train(env):
    """Helper function."""
    env = wrap_deepmind(env, clip_rewards=True)
    env = FrameStack(env, 4)
    return env

def gen_pong_env(seed):
    """Generate a pong environment, with all the bells and whistles."""
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[3]

    env_id = task.env_id
    env = gym.make(env_id)
    env.seed(seed)

    # Can wrap in gym.wrappers.Monitor here if we want to record.
    env = wrap_deepmind(env)
    return env

def gen_vectorized_pong_env(n):
    """
    Generate a vectorized pong environment, with n simultaneous
    differently-seeded envs. For deterministic seeding, you
    should seed np.random.seed beforehand.
    """
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[3]

    env_id = task.env_id
    envs = [wrap_deepmind(gym.make(env_id)) for _ in range(n)]
    env = MultiprocessingEnv(envs)

    seeds = [int(s) for s in np.random.randint(0, 2 ** 30, size=n)]
    env.seed(seeds)
    return env

def gen_pong_ram_env(seed):
    """Generate a pong RAM environment, with all the bells and whistles."""
    env = gym.make("Pong-ram-v0")
    env.seed(seed)
    # Can wrap in gym.wrappers.Monitor here if we want to record.
    env = _wrap_deepmind_ram(env)
    return env

def gen_cherrypicked_pong_ram_env(seed):
    """
    Generate a pong RAM environment, with all the bells and whistles (including
    salient cherrypicking RAM values).
    """
    env = gen_pong_ram_env(seed)
    return _CherryPickedPongRamWrapper(env)
