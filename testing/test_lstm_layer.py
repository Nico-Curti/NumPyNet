#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import LSTM, Input
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.lstm_layer import LSTM_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestLSTMlayer :
  '''
  Tests:
    - costructor of RNN_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    update function.
  '''

  def test_constructor (self):
    pass

  def test_printer (self):
    pass

  def test_forward (self):

    batch     = 100
    timesteps = 1
    features  = 10
    outputs   = 20

    np.random.seed(123)

    weights = [np.random.uniform(size=(features, outputs)), np.random.uniform(size=(outputs,outputs))]
    # bias = [np.random.uniform(size=(outputs,)), np.random.uniform(size=(outputs,))]
    bias = [np.zeros(shape=(outputs,)), np.zeros(shape=outputs)]

    # assign same weights to all the kernel in keras as for NumPyNet
    keras_weights1_concat = np.concatenate([weights[0] for i in range(4)], axis=1)
    keras_weights2_concat = np.concatenate([weights[1] for i in range(4)], axis=1)
    keras_bias = np.concatenate([bias[0] for i in range(4)])

    inpt = np.random.uniform(size=(batch, timesteps, features)).astype(np.float32)

    inp   = Input(shape=(inpt.shape[1:]))
    lstm  = LSTM(units=outputs, implementation=1)(inp)
    model = Model(inputs=[inp], outputs=[lstm])

    model.set_weights([keras_weights1, keras_weights2, keras_bias])

    forward_out_keras = model.predict(inpt)

    forward_out_keras.shape

    inpt = inpt.reshape(batch, 1, 1, timesteps*features)
    layer = LSTM_layer(outputs=outputs, steps=timesteps, weights=weights, bias=bias, input_shape=inpt.shape)

    layer.forward(inpt)
    forward_out_numpynet = layer.output.reshape(batch, outputs)

    forward_out_numpynet.shape

    np.allclose(forward_out_numpynet, forward_out_keras)

    np.abs(forward_out_keras - forward_out_numpynet).max()


  def test_backward (self):
    pass
