"""Train the autoencoder on a dataset of images."""

import argparse
import os

from baselines.common import set_global_seeds
import numpy as np
from sklearn.model_selection import train_test_split
import tensorflow as tf

from autoencoder import AtariAutoencoder
from dataset import Dataset

def main():
    """
    Loads the specified dataset from the file system, then trains an
    autoencoder on the images, reporting results in tensorboard.
    """
    parser = argparse.ArgumentParser(
        description=main.__doc__,
        formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--datafile', type=str,
                        help='file with transition dataset', required=True)
    parser.add_argument('--batch_size', default=512, type=int,
                        help='batch size')
    parser.add_argument('--epochs', default=100, type=int,
                        help='number of epochs to train')
    parser.add_argument('--seed', default=0, type=int,
                        help='random seed')
    parser.add_argument('--bottleneck', default=100, type=int,
                        help='bottleneck')
    parser.add_argument('--lr', default=1e-2, type=float,
                        help='learning rate')
    parser.add_argument('--logdir', default='/tmp/log', type=str,
                        help='tensorboard log directory')
    parser.add_argument('--savedir', default='/tmp/save', type=str,
                        help='tf model save directory')
    args = parser.parse_args()

    set_global_seeds(args.seed)
    dataset = Dataset.load(args.datafile)
    train, val = train_test_split(dataset.obs)
    print('training dataset', train.shape)
    print('validation dataset', val.shape)

    ae = AtariAutoencoder(bottleneck_dims=args.bottleneck,
                          learning_rate=args.lr,
                          batch_size=args.batch_size)

    summary_writer = tf.summary.FileWriter(args.logdir)
    saver = tf.train.Saver()
    os.makedirs(args.savedir, exist_ok=True)
    savefile = os.path.join(args.savedir, 'model.ckpt')
    init = tf.global_variables_initializer()
    tf.get_default_graph().finalize()

    config = tf.ConfigProto()
    config.gpu_options.allow_growth = True
    sess = tf.Session(config=config)
    with sess.as_default():
        sess.run(init)
        for i in range(1, args.epochs + 1):
            ae.fit(train)
            saver.save(sess, savefile)
            summary = ae.summarize_mse(train, val)
            summary_writer.add_summary(summary, i)
            train_mse, val_mse = (x.simple_value for x in summary.value)
            print('epoch {: 4d} of {: 4d} train mse {:6.4f} val mse {:6.4f}'
                  .format(i, args.epochs, train_mse, val_mse))

if __name__ == '__main__':
    main()
