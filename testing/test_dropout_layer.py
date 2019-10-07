# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.dropout_layer import Dropout_layer

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'DropOut Layer testing'

@given(batch = st.integers(min_value=1, max_value=15 ),
       w     = st.integers(min_value=1, max_value=100),
       h     = st.integers(min_value=1, max_value=100),
       c     = st.integers(min_value=1, max_value=10 ))
@settings(max_examples=10,
          deadline=1000)
def test_dropout_layer(batch, w, h, c):
  '''
  Tests:
    Properties of the output, since the seed in tensorflow behaves differently
    from numpy.

    I'm not sure what I should test here.
  '''
  np.random.seed(123)

  # Set of probabilities
  probabilities = [0., .25, .5, .75, 1.]

  for prob in probabilities:

    # Random input
    inpt     = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

    # Initialize the numpy_net model
    numpynet = Dropout_layer(prob)

    # Tensor Flow dropout, just to see if it works
    # forward_out_keras = K.eval(tf.nn.dropout(inpt, seed=None, keep_prob=prob))

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    zeros_out = np.count_nonzero(forward_out_numpynet)

    if   prob == 1.:
      assert zeros_out == 0
    elif prob == 0.:
      assert zeros_out == batch * w * h * c
    else:
      assert forward_out_numpynet.shape == inpt.shape


if __name__ == '__main__':

  test_dropout_layer()
