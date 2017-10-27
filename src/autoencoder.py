"""
Autoencoder, with shape specifically chosen for Atari 84x84 images.
"""
import numpy as np
import tensorflow as tf

from utils import check_shape

class AtariAutoencoder:
    """
    Autoencoder specified to Atari dimensions. Can have specified
    bottleneck dimension width.
    """

    def __init__(self,
                 bottleneck_dims=100,
                 learning_rate=1e-3,
                 batch_size=512):
        self._bottleneck_dims = bottleneck_dims
        self._batch_size = batch_size
        assert self._bottleneck_dims < 512, self._bottleneck_dims

        self._observation_input_ph_ns = tf.placeholder(
            tf.float32, (None, 84, 84, 1), 'autoencoder_input')

        # Training
        bottleneck_nb = self._encode(
            self._observation_input_ph_ns, reuse=None)
        reconstruction_ns = self._decode(bottleneck_nb, reuse=None)
        self._mse = tf.losses.mean_squared_error(
            reconstruction_ns,
            self._observation_input_ph_ns)
        self._update_op = tf.train.AdamOptimizer(learning_rate).minimize(
            self._mse)

    def _encode(self, obs_ns, reuse=True):
        with tf.variable_scope('autoencoder', reuse=reuse):
            out = obs_ns
            with tf.variable_scope('convnet'):
                check_shape(out, [84, 84, 1])
                out = tf.layers.conv2d(
                    out, 32, 8, strides=4, use_bias=False,
                    activation=tf.nn.relu, name='c32x8x4')
                check_shape(out, [20, 20, 32])
                out = tf.layers.conv2d(
                    out, 64, 4, strides=2, use_bias=False,
                    activation=tf.nn.relu, name='c64x4x2')
                check_shape(out, [9, 9, 64])
                out = tf.layers.conv2d(
                    out, 64, 3, strides=1, use_bias=False,
                    activation=tf.nn.relu, name='c64x3x1')
                check_shape(out, [7, 7, 64])
                out = tf.reshape(out, [-1, 7 * 7 * 64])
            with tf.variable_scope('fc'):
                # note fully connected layers are not tied
                # so we give them different names
                out = tf.layers.dense(
                    out, 512, activation=tf.nn.relu, name='enc2')
                out = tf.layers.dense(
                    out, self._bottleneck_dims, activation=tf.nn.relu,
                    name='enc1')
        return out

    def _decode(self, encoded_nb, reuse=True):
        with tf.variable_scope('autoencoder'):
            out = encoded_nb
            with tf.variable_scope('fc', reuse=reuse):
                # only reuse if we're re-constructing in a second _decode call
                check_shape(out, [self._bottleneck_dims])
                out = tf.layers.dense(
                    out, 512, activation=tf.nn.relu, name='dec1')
                out = tf.layers.dense(
                    out, 7 * 7 * 64, activation=tf.nn.relu, name='dec2')
            with tf.variable_scope('convnet', reuse=True):
                # always reuse scope since conv layers are tied
                out = tf.reshape(out, [-1, 7, 7, 64])
                # TODO: add in bias manually? (can't reuse + tie)
                out = tf.layers.conv2d_transpose(
                    out, 64, 3, strides=1, use_bias=False,
                    activation=tf.nn.relu, name='c64x3x1')
                check_shape(out, [9, 9, 64])
                out = tf.layers.conv2d_transpose(
                    out, 32, 4, strides=2, use_bias=False,
                    activation=tf.nn.relu, name='c64x4x2')
                check_shape(out, [20, 20, 32])
                out = tf.layers.conv2d_transpose(
                    out, 1, 8, strides=4, use_bias=False,
                    activation=tf.nn.relu, name='c32x8x4')
                check_shape(out, [84, 84, 1])
        return out

    def encode(self, obs_ns):
        """
        Given a TF tensor of observations, this returns a corresponding
        tensor encoding the observations with the current autoencoder
        parameters.
        """
        return self._encode(obs_ns, reuse=True)

    def _batched_mse(self, data):
        max_size = 8196
        mse = 0
        for i in range(0, len(data) - max_size + 1, max_size):
            data_slice = data[i:i+max_size]
            slice_mse = tf.get_default_session().run(self._mse, feed_dict={
                self._observation_input_ph_ns: data_slice})
            mse += slice_mse * len(data_slice)
        return mse / len(data)

    def summarize_mse(self, training_data, val_data):
        """Return a tf.Summary proto with current parameter's MSEs."""
        train_mse = self._batched_mse(training_data)
        val_mse = self._batched_mse(val_data)
        summ = tf.Summary(value=[
            tf.Summary.Value(tag='train_mse', simple_value=train_mse),
            tf.Summary.Value(tag='val_mse', simple_value=val_mse)])
        return summ

    def fit(self, obs):
        """Fit the data for one epoch."""
        nexamples = len(obs)
        nbatches = max(nexamples // self._batch_size, 1)
        batches = np.random.randint(
            nexamples, size=(nbatches, self._batch_size))
        for batch_idx in batches:
            states = obs[batch_idx]
            tf.get_default_session().run(self._update_op, feed_dict={
                self._observation_input_ph_ns: states})
