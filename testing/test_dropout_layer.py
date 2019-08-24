# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'DropOut Layer testing'


from keras.models import Model
from keras.layers import Input, Activation
import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.dropout_layer import Dropout_layer


import numpy as np


def test_dropout_layer():
  '''
  Tests:
    Properties of the output, since the seed in tensorflow behaves differently
    from numpy.

    I'm not sure what I should test here.
  '''

  # Set of probabilities
  probabilities = np.round(np.linspace(0.,1.,20),2)

  for prob in probabilities:

    batch = np.random.randint(low=1, high=10)

    # Random input
    inpt     = np.random.uniform(0.,1., size=(batch, 200, 201, 3))
    _, w, h, c = inpt.shape

    # Initialize the numpy_net model
    numpynet = Dropout_layer(prob)

    # Tensor Flow dropout, just to see if it works
    # forward_out_keras = K.eval(tf.nn.dropout(inpt, seed = None, keep_prob=prob))

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    zeros_out = np.count_nonzero(forward_out_numpynet == 0)

    if   prob == 1.:
      assert zeros_out ==  batch * w * h * c
    elif prob == 0.:
      assert zeros_out == 0
    else:
      assert forward_out_numpynet.shape == inpt.shape
      assert zeros_out != 0


if __name__ == '__main__':
  test_dropout_layer()
