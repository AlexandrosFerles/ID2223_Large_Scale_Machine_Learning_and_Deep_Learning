import tensorflow as tf
import numpy as np
from matplotlib import pyplot as plt
from __future__ import division, print_function, unicode_literals

def load_mnist():

    from tensorflow.examples.tutorials.mnist import input_data as mnist_data
    mnist = mnist_data.read_data_sets("./Datasets/MNIST/", one_hot=True)
    return mnist


if __name__=='__main__':
    print('Finished!')