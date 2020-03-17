#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import SimpleRNN, Input
from tensorflow.keras.layers import RNN
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.temp_rnn_layer import RNN_layer
from NumPyNet.utils import data_to_timesteps

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
  @settings(max_examples=10,
            deadline=None)
  def _constructor (self, outputs, steps, b, w, h, c):

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

      assert layer.output == None


  @given(outputs= st.integers(min_value=3, max_value=10),
         steps  = st.integers(min_value=1, max_value=4),
         b      = st.integers(min_value=5, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10))
  @settings(max_examples=10,
            deadline=None)
  def _printer (self, outputs, steps, b, w, h, c):

    layer = RNN_layer(outputs=outputs, steps=steps, activation=Linear)

    with pytest.raises(TypeError):
      print(layer)

    layer = RNN_layer(outputs=outputs, steps=steps, activation=Linear, input_shape=(b, w, h, c))

    print(layer)


  @given(steps    = st.integers(min_value=1, max_value=10),
         outputs  = st.integers(min_value=1, max_value=50),
         features = st.integers(min_value=1, max_value=50),
         batch    = st.integers(min_value=20, max_value=100),
         return_seq = st.booleans())
  @settings(max_examples=10, deadline=None)
  def test_forward(self, steps, outputs, features, batch, return_seq):

    activation = 'tanh'

    inpt = np.random.uniform(size=(batch, features))
    inpt_keras, _ = data_to_timesteps(inpt, steps=steps)

    assert inpt_keras.shape == (batch - steps, steps, features)

    # weights init
    kernel           = np.random.uniform(low=-1, high=1, size=(features, outputs))
    recurrent_kernel = np.random.uniform(low=-1, high=1, size=(outputs, outputs))
    bias             = np.random.uniform(low=-1, high=1, size=(outputs,))

    # create keras model
    inp   = Input(shape=inpt_keras.shape[1:])
    rnn   = SimpleRNN(units=outputs, activation=activation, return_sequences=return_seq)(inp)
    model = Model(inputs=inp, outputs=rnn)

    # set weights for the keras model
    model.set_weights([kernel, recurrent_kernel, bias])

    # create NumPyNet layer
    layer = RNN_layer(outputs=outputs, steps=steps, input_shape=(50, 1, 1, 6), activation=activation, return_sequence=return_seq)

    # set NumPyNet weights
    layer.set_weights([kernel, recurrent_kernel, bias])

    # FORWARD

    # forward for keras
    forward_out_keras = model.predict(inpt_keras)
    forward_out_keras.shape

    # forward NumPyNet
    layer.forward(inpt)
    forward_out_numpynet = layer.output.reshape(forward_out_keras.shape)

    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-4)


  @given(steps    = st.integers(min_value=1, max_value=10),
         outputs  = st.integers(min_value=1, max_value=50),
         features = st.integers(min_value=1, max_value=50),
         batch    = st.integers(min_value=20, max_value=100),
         return_seq = st.booleans())
  @settings(max_examples=10, deadline=None)
  def test_backward(self, steps, outputs, features, batch, return_seq):

    return_seq = False # fixed to "many_to_one" for now
    activation = 'linear'

    inpt = np.random.uniform(size=(batch, features))
    inpt_keras, _ = data_to_timesteps(inpt, steps=steps)

    assert inpt_keras.shape == (batch - steps, steps, features)

    # weights init
    kernel           = np.random.uniform(low=-1, high=1, size=(features, outputs))
    recurrent_kernel = np.random.uniform(low=-1, high=1, size=(outputs, outputs))
    bias             = np.random.uniform(low=-1, high=1, size=(outputs,))

    # create keras model
    inp   = Input(shape=inpt_keras.shape[1:])
    rnn   = SimpleRNN(units=outputs, activation=activation, return_sequences=return_seq)(inp)
    model = Model(inputs=inp, outputs=rnn)

    # set weights for the keras model
    model.set_weights([kernel, recurrent_kernel, bias])

    # create NumPyNet layer
    layer = RNN_layer(outputs=outputs, steps=steps, input_shape=(batch, 1, 1, features), activation=activation, return_sequence=return_seq)

    # set NumPyNet weights
    layer.set_weights([kernel, recurrent_kernel, bias])

    assert np.allclose(layer.weights, model.get_weights()[0])
    assert np.allclose(layer.recurrent_weights, model.get_weights()[1])
    assert np.allclose(layer.bias, model.get_weights()[2])

    # FORWARD

    # forward for keras
    forward_out_keras = model.predict(inpt_keras)

    # forward NumPyNet
    layer.forward(inpt)
    forward_out_numpynet = layer.output.reshape(forward_out_keras.shape)

    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-4)

    # BACKWARD

    # Compute the gradient of output w.r.t input
    gradient1 = K.gradients(model.output, [model.input])
    gradient2 = K.gradients(model.output, model.trainable_weights)

    # Define a function to evaluate the gradient
    func1 = K.function(model.inputs + [model.output], gradient1)
    func2 = K.function(model.inputs + model.trainable_weights + model.outputs, gradient2)

    # Compute delta for Keras
    delta_keras = func1([inpt_keras])[0]
    updates     = func2([inpt_keras])

    weights_update_keras           = updates[0]
    recurrent_weights_update_keras = updates[1]
    bias_update_keras              = updates[2]

    # backward pass for NumPyNet
    delta       = np.zeros(shape=inpt_keras.shape, dtype=float)
    layer.delta = np.ones(shape=layer.output.shape, dtype=float)
    layer.backward(inpt, delta, copy=True)

    assert np.allclose(layer.bias_update,    bias_update_keras,     atol=1e-8, rtol=1e-5)
    assert np.allclose(layer.weights_update, weights_update_keras,  atol=1e-8, rtol=1e-5)
    assert np.allclose(delta,                delta_keras,           atol=1e-8, rtol=1e-5)
    assert np.allclose(layer.recurrent_weights_update, recurrent_weights_update_keras, atol=1e-8, rtol=1e-5)
