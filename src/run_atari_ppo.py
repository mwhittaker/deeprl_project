"""Trains PPO agent and plays Atari Pong.
Adapted from: baselines.ppo1.run_atari (MIT License)
"""

import os.path as osp
import logging
from mpi4py import MPI
import gym
from baselines.common import set_global_seeds
from baselines import bench
from baselines import logger
import atari_env


def train(env_id, num_frames, seed, max_ts, logdir):
    """Train agent."""
    from baselines.ppo1 import pposgd_simple, cnn_policy
    import baselines.common.tf_util as U
    rank = MPI.COMM_WORLD.Get_rank()
    sess = U.single_threaded_session()
    sess.__enter__()
    logger.configure(osp.join(logdir, "%i.log.json" % rank))
    workerseed = seed + 10000 * MPI.COMM_WORLD.Get_rank()
    set_global_seeds(workerseed)
    env = gym.make(env_id)
    def policy_fn(name, ob_space, ac_space):
        """Given an obs, returns an act."""
        return cnn_policy.CnnPolicy(name=name, ob_space=ob_space,
                                    ac_space=ac_space)
    env = bench.Monitor(env, logger.get_dir() and
                        osp.join(logger.get_dir(), "%i.monitor.json" % rank))
    env.seed(workerseed)
    gym.logger.setLevel(logging.WARN)

    env = atari_env.wrap_train(env)
    num_timesteps = max_ts or int(num_frames / 4 * 1.1)
    env.seed(workerseed)

    pposgd_simple.learn(env, policy_fn,
                        max_timesteps=num_timesteps,
                        timesteps_per_batch=256,
                        clip_param=0.2, entcoeff=0.01,
                        optim_epochs=4, optim_stepsize=1e-3,
                        optim_batchsize=64,
                        gamma=0.99, lam=0.95,
                        schedule='linear'
                       )
    env.close()

def main():
    """Train agent on given seed."""
    import argparse as ap
    prsr = ap.ArgumentParser(formatter_class=ap.ArgumentDefaultsHelpFormatter)
    prsr.add_argument('--logdir', help='logging directory',
                      default='/tmp/'
                     )
    prsr.add_argument('--env', help='environment ID',
                      default='PongNoFrameskip-v4'
                     )
    prsr.add_argument('--seed', help='RNG seed', type=int, default=0)
    prsr.add_argument("--max_timesteps", type=int)
    args = prsr.parse_args()
    train(args.env, num_frames=40e6, seed=args.seed,
          max_ts=args.max_timesteps, logdir=args.logdir)

if __name__ == '__main__':
    main()
