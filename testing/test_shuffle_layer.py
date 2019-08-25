# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.shuffler_layer import Shuffler_layer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Shuffle Layer testing'

def test_shuffle_layer():
  '''
  Tests:
    if the forward out of the shuffle layer is the same as tensorflow
    if the backward out of the shuffle layer give the same output
  to be:
  '''
  np.random.seed(123)

  couples = [(2,12),(4,32),(4,48),(6,108)]

  for scale, channels in couples:

    # input initialization
    batch = 2
    inpt = np.random.uniform(0.,1., size=(batch, 100, 101, channels))

    # numpynet model
    numpynet = Shuffler_layer(scale)

    # FORWARD

    # Keras operation
    forward_out_keras = K.eval(tf.depth_to_space(inpt, block_size=scale))

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras)

    # BACKWARD

    delta = np.random.uniform(0.,1., size=forward_out_keras.shape)

    delta_keras = K.eval(tf.space_to_depth(delta, block_size = scale))

    numpynet.delta = delta
    delta = delta.reshape(inpt.shape)

    numpynet.backward(delta)

    assert delta_keras.shape == delta.shape
    assert np.allclose(delta_keras, delta)


if __name__ == '__main__':
  test_shuffle_layer()
