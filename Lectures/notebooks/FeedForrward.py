import numpy as np
import tensorflow as tf

class FeedForwardMNISTNetwork():

    def __init__(self, input_size, hidden_layers, output_size, **args):

        self.input_size = input_size
        self.hidden_layers = hidden_layers
        self.output_size = output_size

        from tensorflow.examples.tutorials.mnist import input_data
        self.mnist = input_data.read_data_sets('./tmp/data')

        self.weights_initializer = tf.truncated_normal
        self.activation = tf.nn.relu
        self.optimizer = tf.train.AdamOptimizer
        self.regularization = False
        self.with_batch_norm = False
        self.with_dropout = False
        self.with_lr_decay = False

        # set manually picked settings of the feedforward network
        if args is not None:

            if 'initializer' in args:
                initializer_choice = args['initializer'].lower()

                if initializer_choice == 'normal':
                    pass
                elif initializer_choice == 'he':
                    self.initializer = tf.contrib.layers.variance_scaling_initializer()
                else:
                    raise NotImplementedError('Not implemented initializer option')

            if 'activation' in args:
                activation_choice = args['activation'].lower()

                if activation_choice == 'sigmoid':
                    self.activation = tf.nn.sigmoid
                elif activation_choice == 'elu':
                    self.activation == tf.nn.elu
                elif activation_choice == 'relu':
                    pass
                else:
                    raise NotImplementedError('Not implemented activation function option')

            if 'optimizer' in args:
                optimizer_choice = args['optimizer'].lower()

                if optimizer_choice == 'adagrad':
                    self.optimizer = tf.train.AdagradOptimizer
                elif optimizer_choice == 'sgd':
                    self.optimizer == tf.train.GradientDescentOptimizer
                elif optimizer_choice == 'adam':
                    pass
                else:
                    raise NotImplementedError('Not implemented optimizer option')

            if 'with_regularization' in args:
                if args['with_regularization']:
                    self.regularization= True
                    if 'regularization_amount' in args:
                        self.reg_scale = args['regularization_amount']
                    else:
                        self.reg_scale = 0.1

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

            if 'with_lr_decay' in args:
                if args['with_lr_decay']:
                    self.with_lr_decay = True

                    if 'initial_eta' in args:
                        self.initial_eta = args['initial_eta']
                    else:
                        self.initial_eta = 1e-2

                    if 'decay_step' in args:
                        self.decay_step = args['decay_step']
                    else:
                        self.decay_step = 10000

                    if 'decay_rate' in args:
                        self.decay_rate = args['decay_rate']
                    else:
                        self.decay_rate = 0.1

if __name__=='__main__':

    settings = {
        'initializer': 'Normal',
        'activation': 'ReLU',
        'optimizer': 'SGD',
        'with_batch_norm': False,
        'with_dropout': False
    }

    FeedForwardMNISTNetwork(784, [300, 100], 10, settings= settings)