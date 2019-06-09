import numpy as np
import tensorflow as tf

class FeedForwardMNISTNetwork():

    def __init__(self, input_size, hidden_layers, output_size, **args):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('./tmp/data')

        self.activation = tf.nn.relu
        self.optimizer = tf.train.AdamOptimizer
        self.with_batch_norm = False
        self.with_dropout = False
        self.weights_initializer = tf.truncated_normal

        # set manually picked settings of the feedforward network
        if args is not None:

            if 'activation' in args:
                activation_choice = args['activation'].lower()

                if activation_choice == 'sigmoid':
                    self.activation = tf.nn.sigmoid()
                elif activation_choice == 'elu':
                    self.activation == tf.nn.elu()
                else:
                    raise NotImplementedError('Not implemented optimizer option')

            if 'optimizer' in args:
                optimizer_choice = args['optimizer'].lower()

                if optimizer_choice == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer()
                elif optimizer_choice == 'sgd':
                    self.optimizer == tf.train.GradientDescentOptimizer()
                else:
                    raise NotImplementedError('Not implemented optimizer option')

            if 'with_batch_norm' in args:
                if args['with_batch_norm']:
                    self.with_batch_norm = True

            if 'with_dropout' in args:
                if args['with_dropout']:
                    self.with_dropout = True
                    if 'dropout_rate' in args:
                        self.dropout_rate = args['dropout_rate']
                    else:
                        self.dropout_rate = 0.5

if __name__=='__main__':

    settings = {
        'with_batch_norm': False,
        'with_dropout': False,
    }
    FeedForwardMNISTNetwork(784, [300, 100], 10, settings= settings)