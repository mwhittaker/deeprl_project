"""Miscellaneous, small utilities"""
import pickle

import gym
import numpy as np

def save_object(obj, filename):
    """Pickle an object to a file, overwriting if the file exists"""
    with open(filename, 'wb') as f:
        pickle.dump(obj, f, pickle.HIGHEST_PROTOCOL)

def load_object(filename):
    """Load an object from a pickled file"""
    with open(filename, 'rb') as f:
        return pickle.load(f)

def get_ob_dim(env):
    """
    Retrieves observation space dimensions from environment, after verifying
    they constitute a continuous input space.
    """
    ob_space = env.observation_space
    assert isinstance(ob_space, gym.spaces.Box), type(ob_space)
    return ob_space.shape

def get_num_acs(env):
    """
    Retrieves the number of discrete disjoint actions that can be taken by the
    agent, after verifying that the action space is indeed discrete.
    """
    ac_space = env.action_space
    assert isinstance(ac_space, gym.spaces.Discrete), type(ac_space)
    return ac_space.n

def create_random_policy(env):
    """Generate a policy that randomly samples from the environment"""
    num_acs = get_num_acs(env)
    return lambda states_ns: (
        np.random.randint(num_acs, size=(states_ns.shape[0],)))

def check_shape(batch_tf, shape):
    """Checks that each item in a batch has specified shape"""
    item_shape = [dim.value for dim in batch_tf.shape[1:]]
    assert item_shape == shape, (item_shape, shape)
