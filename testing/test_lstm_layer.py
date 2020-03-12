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
from NumPyNet.utils import data_to_timesteps

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

    batch     = 11
    timesteps = 5
    features  = 3
    outputs   = 5

    np.random.seed(123)

    data = np.random.uniform(size=(batch, features))

    inpt_keras = data_to_timesteps(data, timesteps)

    assert inpt_keras.shape == (batch - timesteps + 1, timesteps, features)

    weights = [np.random.uniform(size=(features, outputs)), np.random.uniform(size=(outputs,outputs))]
    bias    = [np.zeros(shape=(outputs,)), np.zeros(shape=outputs)]

    # assign same weights to all the kernel in keras as for NumPyNet
    keras_weights1 = np.concatenate([weights[0] for i in range(4)], axis=1)
    keras_weights2 = np.concatenate([weights[1] for i in range(4)], axis=1)
    keras_bias     = np.concatenate([bias[0] for i in range(4)])

    for i in range(4):
      assert np.allclose(keras_weights1[:,outputs*i:outputs*(i+1)], weights[0])

    for i in range(4):
      assert np.allclose(keras_weights2[:,outputs*i:outputs*(i+1)], weights[1])

    inp   = Input(shape=(inpt_keras.shape[1:]))
    lstm  = LSTM(units=outputs, implementation=1, use_bias=False)(inp)
    model = Model(inputs=[inp], outputs=[lstm])

    model.set_weights([keras_weights1, keras_weights2])

    inpt_numpynet = data.reshape(batch, 1, 1, features)
    layer = LSTM_layer(outputs=outputs, steps=timesteps, weights=weights, bias=bias, input_shape=inpt_numpynet.shape)

    assert np.allclose(layer.uf.weights, model.get_weights()[0][:, :outputs])
    assert np.allclose(layer.ui.weights, model.get_weights()[0][:, outputs:2*outputs])
    assert np.allclose(layer.ug.weights, model.get_weights()[0][:, 2*outputs:3*outputs])
    assert np.allclose(layer.uo.weights, model.get_weights()[0][:, 3*outputs:4*outputs])

    assert np.allclose(layer.wf.weights, model.get_weights()[1][:, :outputs])
    assert np.allclose(layer.wi.weights, model.get_weights()[1][:, outputs:2*outputs])
    assert np.allclose(layer.wg.weights, model.get_weights()[1][:, 2*outputs:3*outputs])
    assert np.allclose(layer.wo.weights, model.get_weights()[1][:, 3*outputs:4*outputs])

    forward_out_keras = model.predict(inpt_keras)

    layer.forward(inpt_numpynet)
    forward_out_numpynet = layer.output.reshape(batch, outputs)

    # np.allclose(forward_out_numpynet, forward_out_keras)

    # np.abs(forward_out_keras - forward_out_numpynet).max()


  def test_backward (self):
    pass
