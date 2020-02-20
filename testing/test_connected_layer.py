#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.activations import Activations
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.connected_layer import Connected_layer
from tensorflow.keras.layers import Dense

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'Connected Layer testing'


class TestConnectedLayer:
  '''
  Tests:
    - costructor of Connected_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be tested:
    update function, keras update not clear.
  '''

  @given(outputs= st.integers(-3, 10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_costructor (self, outputs, b, w, h, c):

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


  @given(outputs= st.integers(1, 10),
         b      = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_printer (self, outputs, b, w, h, c):

    layer = Connected_layer(outputs=outputs, activation=Linear)

    with pytest.raises(AttributeError):
      print(layer)

    layer = Connected_layer(outputs=outputs, activation=Linear, input_shape=(b, w, h, c))

    print(layer)


  @given(outputs= st.integers(1, 10),
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

      # Keras Model
      inp   = Input(batch_shape=(b, w * h * c))
      x     = Dense(outputs, activation=keras_act)(inp)
      model = Model(inputs=[inp], outputs=x)

      # Set weights in Keras Model.
      model.set_weights([weights, bias])

      # FORWARD

      # Keras forward output
      forward_out_keras = model.predict(inpt.reshape(b, -1))

      # Numpy_net forward output
      layer.forward(inpt)
      forward_out_numpynet = layer.output

      assert forward_out_numpynet.shape == (b, 1, 1, outputs)
      assert np.allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, atol=1e-4)


  @given(outputs= st.integers(1, 10),
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

    # Numpy_net model
    layer = Connected_layer(outputs, input_shape=inpt.shape,
                                     activation=numpynet_act,
                                     weights=weights, bias=bias)
    # Keras Model
    inp   = Input(batch_shape=(b, w * h * c))
    x     = Dense(outputs, activation=keras_act)(inp)
    model = Model(inputs=[inp], outputs=x)

    # Set weights in Keras Model.
    model.set_weights([weights, bias])

    # Try backward
    with pytest.raises(NotFittedError):
      delta = np.empty(shape=inpt.shape)
      layer.backward(inpt, delta)

    # FORWARD

    # Keras forward output
    forward_out_keras = model.predict(inpt.reshape(b, -1))

    # Numpy_net forward output
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Forward output Test
    assert np.allclose(forward_out_numpynet[:, 0, 0, :], forward_out_keras, atol=1e-4)

    # BACKWARD

    # Output derivative in respect to input
    grad1 = K.gradients(model.output, [model.input])

    # Output derivative respect to trainable_weights(Weights and Biases)
    grad2 = K.gradients(model.output, model.trainable_weights)

    # Definning functions to compute those gradients
    func1 = K.function(model.inputs + [model.output], grad1)
    func2 = K.function(model.inputs + model.trainable_weights + [model.output], grad2)

    # Evaluation of Delta, weights_updates and bias_updates for Keras
    delta_keras = func1([inpt.reshape(b, -1)])
    updates     = func2([inpt.reshape(b, -1)])

    # Initialization of NumPyNet starting delta to ones
    layer.delta = np.ones(shape=layer.out_shape, dtype=float)

    # Initialization of global delta
    delta = np.zeros(shape=(b, w, h, c), dtype=float)

    # Computation of delta, weights_update and bias updates for numpy_net
    layer.backward(inpt, delta=delta)

    #Now the global variable delta is updated
    assert np.allclose(delta_keras[0].reshape(b, w, h, c), delta, atol=1e-8)
    assert np.allclose(updates[0], layer.weights_update, atol=1e-6)
    assert np.allclose(updates[1], layer.bias_update, atol=1e-7)
