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

  @given(outputs= st.integers(min_value=-3, max_value=10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, outputs, b, w, h, c):

    numpynet_activ = [Relu, Logistic, Tanh, Linear]

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

    for numpynet_act in numpynet_activ:

      layer = Connected_layer(outputs=outputs, activation=numpynet_act,
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

      assert layer.activation == numpynet_act.activate
      assert layer.gradient   == numpynet_act.gradient


  @given(outputs= st.integers(min_value=1, max_value=10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_printer (self, outputs, b, w, h, c):

    layer = Connected_layer(outputs=outputs, activation=Linear)

    with pytest.raises(TypeError):
      print(layer)

    layer = Connected_layer(outputs=outputs, activation=Linear, input_shape=(b, w, h, c))

    print(layer)


  @given(outputs= st.integers(min_value=1, max_value=10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, outputs, b, w, h, c):

    keras_activ = ['relu', 'sigmoid', 'tanh','linear']
    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    weights = np.random.uniform(low=-1., high=1., size=(w * h * c, outputs))
    bias    = np.random.uniform(low=-1., high=1., size=(outputs))

    inpt = np.random.uniform(low=-1., high=1., size=(b, w, h, c))

    for keras_act, numpynet_act in zip(keras_activ, numpynet_activ):

      # Numpy_net model
      layer = Connected_layer(outputs, input_shape=inpt.shape,
                                       activation=numpynet_act,
                                       weights=weights, bias=bias)


      # Tensorflow Layer
      model = tf.keras.layers.Dense(outputs, activation=keras_act,
                                    kernel_initializer=lambda shape, dtype=None : weights,
                                    bias_initializer=lambda shape, dtype=None : bias)

      # FORWARD

      # Keras forward output
      forward_out_keras = model(inpt.reshape(b, -1))

      # Numpy_net forward output
      layer.forward(inpt)
      forward_out_numpynet = layer.output

      assert forward_out_numpynet.shape == (b, 1, 1, outputs)
      assert np.allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, atol=1e-2)


  @given(outputs= st.integers(min_value=1, max_value=10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, outputs, b, w, h, c):

    numpynet_act = Linear
    keras_act    = 'linear'

    weights = np.random.uniform(low=0., high=1., size=(w * h * c, outputs))
    bias    = np.random.uniform(low=0., high=1., size=(outputs))

    inpt = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    tf_input = tf.Variable(inpt.astype('float32').reshape(b,-1))

    # NumPyNet model
    layer = Connected_layer(outputs, input_shape=inpt.shape,
                                     activation=numpynet_act,
                                     weights=weights, bias=bias)
    # Tensorflow layer
    model = tf.keras.layers.Dense(outputs, activation=keras_act,
                                  kernel_initializer=lambda shape, dtype=None : weights,
                                  bias_initializer=lambda shape, dtype=None : bias)

    # Try backward
    with pytest.raises(NotFittedError):
      delta = np.empty(shape=inpt.shape)
      layer.backward(inpt, delta)

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
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Forward output Test
    assert np.allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, atol=1e-2)

    # BACKWARD

    # Initialization of NumPyNet starting delta to ones
    layer.delta = np.ones(shape=layer.out_shape, dtype=float)

    # Initialization of global delta
    delta = np.zeros(shape=(b, w, h, c), dtype=float)

    # Computation of delta, weights_update and bias updates for numpy_net
    layer.backward(inpt, delta=delta)

    #Now the global variable delta is updated
    assert np.allclose(delta_keras.reshape(b, w, h, c), delta, atol=1e-8)
    assert np.allclose(updates[0], layer.weights_update, atol=1e-6)
    assert np.allclose(updates[1], layer.bias_update, atol=1e-7)
