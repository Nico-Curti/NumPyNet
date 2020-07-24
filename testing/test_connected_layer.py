#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.activations import Activations
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.connected_layer import Connected_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


nn_activation = [Relu, Logistic, Linear]
tf_activation = ['relu', 'sigmoid','linear']

class TestConnectedLayer:
  '''
  Tests:
    - costructor of Connected_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    update function.
  '''

  @given(outputs = st.integers(min_value=-3, max_value=10),
         b       = st.integers(min_value=1, max_value=15),
         w       = st.integers(min_value=15, max_value=100),
         h       = st.integers(min_value=15, max_value=100),
         c       = st.integers(min_value=1, max_value=10),
         act_fun = st.sampled_from(nn_activation))
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, outputs, b, w, h, c, act_fun):

    if outputs > 0:
      weights_choice = [np.random.uniform(low=-1, high=1., size=( w * h * c, outputs)), None]
      bias_choice = [np.random.uniform(low=-1, high=1., size=(outputs,)), None]

    else :
      with pytest.raises(ValueError):
        layer = Connected_layer(outputs=outputs)

      outputs += 10
      weights_choice = [np.random.uniform(low=-1, high=1., size=(w * h * c, outputs)), None]
      bias_choice = [np.random.uniform(low=-1, high=1., size=(outputs,)), None]

    weights = choice(weights_choice)
    bias    = choice(bias_choice)

    layer = Connected_layer(outputs=outputs, activation=act_fun,
                            input_shape=(b, w, h, c),
                            weights=weights, bias=bias)

    if weights is not None:
      assert np.allclose(layer.weights, weights)

    if bias is not None:
      assert np.allclose(layer.bias, bias)
    else :
      assert np.allclose(layer.bias, np.zeros(shape=(outputs,)))

    assert layer.output         == None
    assert layer.weights_update == None
    assert layer.bias_update    == None
    assert layer.optimizer      == None

    assert layer.activation == act_fun.activate
    assert layer.gradient   == act_fun.gradient


  @given(outputs = st.integers(min_value=1, max_value=10),
         b       = st.integers(min_value=1, max_value=15),
         w       = st.integers(min_value=15, max_value=100),
         h       = st.integers(min_value=15, max_value=100),
         c       = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_printer (self, outputs, b, w, h, c):

    layer = Connected_layer(outputs=outputs, activation=Linear)

    with pytest.raises(TypeError):
      print(layer)

    layer = Connected_layer(outputs=outputs, activation=Linear, input_shape=(b, w, h, c))

    print(layer)


  @given(outputs = st.integers(min_value=1, max_value=10),
         b       = st.integers(min_value=1, max_value=15),
         w       = st.integers(min_value=15, max_value=100),
         h       = st.integers(min_value=15, max_value=100),
         c       = st.integers(min_value=1, max_value=10),
         idx_act = st.integers(min_value=0, max_value=len(nn_activation)-1))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, outputs, b, w, h, c, idx_act):

    weights = np.random.uniform(low=-1., high=1., size=(w * h * c, outputs)).astype(float)
    bias    = np.random.uniform(low=-1., high=1., size=(outputs)).astype(float)

    inpt = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)

    # Numpy_net model
    layer = Connected_layer(outputs, input_shape=inpt.shape,
                                     activation=nn_activation[idx_act],
                                     weights=weights, bias=bias)


    # Tensorflow Layer
    model = tf.keras.layers.Dense(outputs, activation=tf_activation[idx_act],
                                  kernel_initializer=lambda shape, dtype=None : weights,
                                  bias_initializer=lambda shape, dtype=None : bias)

    # FORWARD

    # Keras forward output
    forward_out_keras = model(inpt.reshape(b, -1))

    # Numpy_net forward output
    layer.forward(inpt=inpt)
    forward_out_numpynet = layer.output

    assert forward_out_numpynet.shape == (b, 1, 1, outputs)
    np.testing.assert_allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, rtol=1e-5, atol=1e-2)


  @given(outputs = st.integers(min_value=1, max_value=10),
         b       = st.integers(min_value=1, max_value=15),
         w       = st.integers(min_value=15, max_value=100),
         h       = st.integers(min_value=15, max_value=100),
         c       = st.integers(min_value=1, max_value=10),
         idx_act = st.integers(min_value=0, max_value=len(nn_activation)-1))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, outputs, b, w, h, c, idx_act):

    weights = np.random.uniform(low=0., high=1., size=(w * h * c, outputs)).astype(float)
    bias    = np.random.uniform(low=0., high=1., size=(outputs)).astype(float)

    inpt = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)
    tf_input = tf.Variable(inpt.astype(float).reshape(b, -1))

    # NumPyNet model
    layer = Connected_layer(outputs, input_shape=inpt.shape,
                                     activation=nn_activation[idx_act],
                                     weights=weights, bias=bias)
    # Tensorflow layer
    model = tf.keras.layers.Dense(outputs, activation=tf_activation[idx_act],
                                  kernel_initializer=lambda shape, dtype=None : weights,
                                  bias_initializer=lambda shape, dtype=None : bias)

    # Try backward
    with pytest.raises(NotFittedError):
      delta = np.empty(shape=inpt.shape, dtype=float)
      layer.backward(inpt=inpt, delta=delta)

    # FORWARD

    # Keras forward output
    # Tensorflow forward and backward
    with tf.GradientTape(persistent=True) as tape :
      preds = model(tf_input)
      grad1 = tape.gradient(preds, tf_input)
      grad2 = tape.gradient(preds, model.trainable_weights)

      forward_out_keras = preds.numpy()
      delta_keras = grad1.numpy()
      updates     = grad2

    # Numpy_net forward output
    layer.forward(inpt=inpt)
    forward_out_numpynet = layer.output

    # Forward output Test
    np.testing.assert_allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, rtol=1e-5, atol=1e-2)

    # BACKWARD

    # Initialization of NumPyNet starting delta to ones
    layer.delta = np.ones(shape=layer.out_shape, dtype=float)

    # Initialization of global delta
    delta = np.zeros(shape=(b, w, h, c), dtype=float)

    # Computation of delta, weights_update and bias updates for numpy_net
    layer.backward(inpt=inpt, delta=delta)

    #Now the global variable delta is updated
    np.testing.assert_allclose(delta_keras.reshape(b, w, h, c), delta, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(updates[0], layer.weights_update, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(updates[1], layer.bias_update, rtol=1e-4, atol=1e-7)
