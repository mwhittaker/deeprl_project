"""Useful TF constructions for Atari vision featurizing."""

import tensorflow as tf
from tensorflow.contrib import layers

def atari_features(img_in, output_dim, scope, reuse=False):
    """
    The base of the architecture described by
    Human-level control through deep reinforcement learning
    by Mnih et al 2015. Generates the tensor featurizing Atari
    input.
    """
    with tf.variable_scope(scope, reuse=reuse):
        out = img_in
        with tf.variable_scope('convnet'):
            out = layers.convolution2d(
                out, num_outputs=32, kernel_size=8, stride=4,
                activation_fn=tf.nn.relu)
            out = layers.convolution2d(
                out, num_outputs=64, kernel_size=4, stride=2,
                activation_fn=tf.nn.relu)
            out = layers.convolution2d(
                out, num_outputs=64, kernel_size=3, stride=1,
                activation_fn=tf.nn.relu)
        out = layers.flatten(out)
        with tf.variable_scope('fc'):
            out = layers.fully_connected(
                out, num_outputs=512, activation_fn=tf.nn.relu)
            out = layers.fully_connected(
                out, num_outputs=output_dim, activation_fn=None)

    return out
