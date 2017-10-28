"""Wrappers for Atari playing, adopted from Deep RL HW3 starter code."""

import gym
import numpy as np

from multiprocessing_env import MultiprocessingEnv
# from baselines.common.atari_wrappers import wrap_deepmind
from baselines.common import atari_wrappers

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
        gym.ObservationWrapper.__init__(self, env)
        self.indexes = [51, 50, 21, 60, 49, 4, 54, 11, 121, 12]
        n = len(self.indexes)
        self.observation_space = gym.spaces.Box(0.0, 255.0, shape=(n,))

    def _observation(self, observation):
        return observation[self.indexes]

def _wrap_deepmind(env,
                   episode_life=True,
                   clip_rewards=True,
                   frame_stack=False,
                   scale=False,
                   warp_frame=False):
    """
    A variant of wrap_deepmind [1] but with the ability to disaple frame
    warping.

    [1]: https://goo.gl/CfetYi
    """
    env = atari_wrappers.NoopResetEnv(env, noop_max=30)
    env = atari_wrappers.MaxAndSkipEnv(env, skip=4)
    if episode_life:
        env = atari_wrappers.EpisodicLifeEnv(env)
    if 'FIRE' in env.unwrapped.get_action_meanings():
        env = atari_wrappers.FireResetEnv(env)
    if warp_frame:
        env = atari_wrappers.WarpFrame(env)
    if scale:
        env = atari_wrappers.ScaledFloatFrame(env)
    if clip_rewards:
        env = atari_wrappers.ClipRewardEnv(env)
    if frame_stack:
        env = atari_wrappers.FrameStack(env, 4)
    return env

def _wrap_pong_env(env_thunk,
                   vectorized_n=1,
                   episode_life=True,
                   clip_rewards=True,
                   frame_stack=False,
                   scale=False,
                   warp_frame=False):
    """
    _wrap_pong_env more or less wraps wrap_deepmind with some additional logic
    to vectorize environments. `env_thunk` is an arity-0 function that returns
    a fresh copy of an environment. The reason we cannot just pass in an
    environment is that a MultiprocessingEnv requires multiple copies of an
    environment, and there's not a nice way to copy an environment. See
    `gen_pong_video_env` for a description of the other flags.
    """
    def _env():
        env = env_thunk()
        env = _wrap_deepmind(env,
                             episode_life=episode_life,
                             clip_rewards=clip_rewards,
                             frame_stack=frame_stack,
                             scale=scale,
                             warp_frame=warp_frame)
        return env

    if vectorized_n == 1:
        return _env()

    envs = [_env() for _ in range(vectorized_n)]
    env = MultiprocessingEnv(envs)
    seeds = [int(s) for s in np.random.randint(0, 2 ** 30, size=vectorized_n)]
    env.seed(seeds)
    return env

def gen_pong_ram_env(seed,
                     cherry_picked=True,
                     vectorized_n=1,
                     episode_life=True,
                     clip_rewards=True):
    """
    Returns a Pong environment with an observation space of RAM snapshots. If
    `cherry_picked` is True, then only a hand-picked subset of the bytes in RAM
    are returned. See `gen_pong_video_env` for a described of the other flags.
    """
    # TODO: Potentially add frame_stack.
    # TODO: Consider if scale makes sense for RAM.

    def _env_thunk():
        env = gym.make("Pong-ram-v0")
        env.seed(seed)
        if cherry_picked:
            env = _CherryPickedPongRamWrapper(env)
        return env

    return _wrap_pong_env(_env_thunk,
                          vectorized_n=vectorized_n,
                          episode_life=episode_life,
                          clip_rewards=clip_rewards,
                          frame_stack=False,
                          scale=False,
                          warp_frame=False)

def gen_pong_video_env(seed,
                       vectorized_n=1,
                       episode_life=True,
                       clip_rewards=True,
                       frame_stack=False,
                       scale=False,
                       warp_frame=True):
    """
    Returns a Pong environment with an observation space of video frames.
    If `vectorized_n` > 1, then `vectorized_n` copies of the environment will
    be run in a `MultiprocessingEnv`. The `episode_life`, `clip_rewards`,
    `frame_stack`, and `scale` flags are described here: https://goo.gl/CfetYi.
    """
    def _env_thunk():
        benchmark = gym.benchmark_spec('Atari40M')
        task = benchmark.tasks[3]
        env_id = task.env_id
        env = gym.make(env_id)
        env.seed(seed)
        return env

    return _wrap_pong_env(_env_thunk,
                          vectorized_n=vectorized_n,
                          episode_life=episode_life,
                          clip_rewards=clip_rewards,
                          frame_stack=frame_stack,
                          scale=scale,
                          warp_frame=warp_frame)
