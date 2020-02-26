# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

from NumPyNet.exception import NotFittedError
from NumPyNet.layers.shortcut_layer import Shortcut_layer
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from tensorflow.keras.layers import Add

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestShortcutLayer :
  '''
  Tests:
    - constructor of the layer
    - printer of the layer
    - forward of the layer against keras
    - backward of the layer against keras

  to be:
    - different shape input.
  '''

  @given(alpha = st.floats(min_value=0., max_value=1., width=32),
         beta  = st.floats(min_value=0., max_value=1., width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_constructor (self, alpha, beta):

    activation = Linear

    layer = Shortcut_layer(activation=activation, alpha=alpha, beta=beta)

    assert layer.activation == activation.activate
    assert layer.gradient   == activation.gradient
    assert layer.alpha == alpha
    assert layer.beta  == beta
    assert layer.output == None
    assert layer.delta  == None
    assert layer._out_shape == None

    assert (layer.ix, layer.jx, layer.kx) == (None, None, None)
    assert (layer.iy, layer.jy, layer.ky) == (None, None, None)


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         alpha = st.floats(min_value=0., max_value=1., width=32),
         beta  = st.floats(min_value=0., max_value=1., width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, b, w, h, c, alpha, beta):

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))

    layer = Shortcut_layer(activation=Linear, alpha=alpha, beta=beta)

    with pytest.raises(TypeError):
      print(layer)

    layer.forward(inpt1, inpt2)

    print(layer)

  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         alpha = st.floats(min_value=0., max_value=1., width=32),
         beta  = st.floats(min_value=0., max_value=1., width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c, alpha, beta):

    keras_activations = ['tanh','linear','relu', 'sigmoid']
    numpynet_activations = [Tanh, Linear, Relu, Logistic]

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))

    for k_activ, n_activ in zip(keras_activations, numpynet_activations):

      layer = Shortcut_layer(activation=n_activ, alpha=alpha, beta=beta)

      # Keras Model, double input
      inp1  = Input(batch_shape=inpt1.shape)
      inp2  = Input(batch_shape=inpt2.shape)
      x     = Add()([inp1,inp2])
      out   = Activation(activation=k_activ)(x)
      model = Model(inputs=[inp1,inp2], outputs=out)

      # FORWARD

      forward_out_keras = model.predict([alpha*inpt1, beta*inpt2])

      layer.forward(inpt1,inpt2)
      forward_out_numpynet = layer.output

      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-7)

      # # minor test with different input size (no keras version)
      #
      # inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
      # inpt2 = np.random.uniform(low=-1., high=1., size=(b, w//2, h//2, c))
      #
      # layer.forward(inpt1, inpt2)
      #
      # assert inpt1.shape == layer.output.shape


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         alpha = st.floats(min_value=0., max_value=1., width=32),
         beta  = st.floats(min_value=0., max_value=1., width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, alpha, beta):

    keras_activations = ['tanh','linear','relu', 'sigmoid']
    numpynet_activations = [Tanh, Linear, Relu, Logistic]

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c))

    for k_activ, n_activ in zip(keras_activations, numpynet_activations):

      # numpynet model
      layer = Shortcut_layer(activation=n_activ, alpha=alpha, beta=beta)

      # Keras Model, double input
      inp1  = Input(batch_shape=inpt1.shape)
      inp2  = Input(batch_shape=inpt2.shape)
      x     = Add()([inp1,inp2])
      out   = Activation(activation=k_activ)(x)
      model = Model(inputs=[inp1,inp2], outputs=out)

      # Try backward:
      with pytest.raises(NotFittedError):
        delta      = np.zeros(shape=inpt1.shape, dtype=float)
        prev_delta = np.zeros(shape=inpt2.shape, dtype=float)
        layer.backward(delta, prev_delta)

      # FORWARD

      # Perform Add() for alpha*inpt and beta*inpt
      forward_out_keras = model.predict([alpha*inpt1, beta*inpt2])

      layer.forward(inpt1,inpt2)
      forward_out_numpynet = layer.output

      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-7)

      # BACKWARD

      grad = K.gradients(model.output, model.inputs)
      func = K.function(model.inputs + [model.output],grad)

      delta1, delta2 = func([alpha*inpt1, beta*inpt2])

      delta1 *= alpha
      delta2 *= beta

      delta      = np.zeros(shape=inpt1.shape, dtype=float)
      prev_delta = np.zeros(shape=inpt2.shape, dtype=float)

      layer.delta = np.ones(shape=(b, w, h, c), dtype=float)
      layer.backward(delta, prev_delta)

      assert np.allclose(delta1, delta)
      assert np.allclose(delta2, prev_delta, atol=1e-8)

if __name__ == '__main__':

  test = TestShortcutLayer()

  test.test_forward()
