"""Wrappers for Atari playing, adopted from Deep RL HW3 starter code."""

import gym
import numpy as np

from multiprocessing_env import MultiprocessingEnv
from baselines.common.atari_wrappers import (wrap_deepmind, FrameStack)
from baselines.common.atari_wrappers import (NoopResetEnv, FireResetEnv,
                                             EpisodicLifeEnv, MaxAndSkipEnv,
                                             ClipRewardEnv)

def _wrap_deepmind_ram(env):
    """Applies various Atari-specific wrappers to make learning easier."""
    env = EpisodicLifeEnv(env)
    env = NoopResetEnv(env, noop_max=30)
    env = MaxAndSkipEnv(env, skip=4)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = FireResetEnv(env)
    env = ClipRewardEnv(env)
    return env

def _wrap_deepmind(env, frame_stack=False):
    """Analogous to _wrap_deepmind_ram, but for the original Atari env."""
    env = wrap_deepmind(env, clip_rewards=True)
    if frame_stack:
        env = FrameStack(env, 4)
    return env

def gen_pong_env(seed, frame_stack=False):
    """Generate a pong environment, with all the bells and whistles."""
    benchmark = gym.benchmark_spec('Atari40M')
    task = benchmark.tasks[3]

    env_id = task.env_id
    env = gym.make(env_id)
    env.seed(seed)

    # Can wrap in gym.wrappers.Monitor here if we want to record.
    env = _wrap_deepmind(env, frame_stack)
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
    # TODO: wrap_deepmind -> _wrap_deepmind, see issue #21
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
    # TODO: frame_stack, see issue #21
    env = _wrap_deepmind_ram(env)
    return env
