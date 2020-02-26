#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import SimpleRNN
import tensorflow.keras.backend as K

from NumPyNet.activations import Activations
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.rnn_layer import RNN_layer
from tensorflow.keras.layers import RNN

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestRNNLayer:
  '''
  Tests:
    - costructor of RNN_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    update function.
  '''

  @given(outputs= st.integers(min_value=-3, max_value=10),
         steps  = st.integers(min_value=1, max_value=4),
         b      = st.integers(min_value=5, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, outputs, steps, b, w, h, c):

    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    if outputs > 0:
      weights_choice = [np.random.uniform(low=-1, high=1., size=( w * h * c, outputs)), None]
      bias_choice = [np.random.uniform(low=-1, high=1., size=(outputs,)), None]

    else :
      with pytest.raises(ValueError):
        layer = RNN_layer(outputs=outputs, steps=steps)

      outputs += 10
      weights_choice = [[np.random.uniform(low=-1, high=1., size=(w * h * c, outputs))]*3, None]
      bias_choice = [[np.random.uniform(low=-1, high=1., size=(outputs,))]*3, None]

    weights = choice(weights_choice)
    bias    = choice(bias_choice)

    for numpynet_act in numpynet_activ:

      layer = RNN_layer(outputs=outputs, steps=steps, activation=numpynet_act,
                        input_shape=(b, w, h, c),
                        weights=weights, bias=bias)

      if weights is not None:
        assert np.allclose(layer.input_layer.weights, weights[0])
        assert np.allclose(layer.self_layer.weights, weights[1])
        assert np.allclose(layer.output_layer.weights, weights[2])

      if bias is not None:
        assert np.allclose(layer.input_layer.bias, bias[0])
        assert np.allclose(layer.self_layer.bias, bias[1])
        assert np.allclose(layer.output_layer.bias, bias[2])

      assert layer.output         == None
      assert layer.optimizer      == None


  @given(outputs= st.integers(min_value=3, max_value=10),
         steps  = st.integers(min_value=1, max_value=4),
         b      = st.integers(min_value=5, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_printer (self, outputs, steps, b, w, h, c):

    layer = RNN_layer(outputs=outputs, steps=steps, activation=Linear)

    with pytest.raises(TypeError):
      print(layer)

    layer = RNN_layer(outputs=outputs, steps=steps, activation=Linear, input_shape=(b, w, h, c))

    print(layer)


  def forward (self):

    inpt = np.random.uniform(size=(50, 1, 1))

    model = Sequential()
    model.add(SimpleRNN(units=32, input_shape=(1, 1), activation='linear'))

    forward_out_keras = model.predict(inpt)

    layer = RNN_layer(outputs=32, steps=1, input_shape=(1, 50, 1, 1), activation='linear')

    inpt.shape = (50, 1, 1, 1)

    layer.forward(inpt)
    forward_out_numpynet = layer.output

    np.allclose(forward_out_numpynet, forward_out_keras)

    forward_out_keras - forward_out_numpynet
