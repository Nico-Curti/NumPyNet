# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow as tf

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


nn_activations = [Tanh, Linear, Relu, Logistic]
tf_activations = ['tanh','linear','relu', 'sigmoid']


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

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)

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
         beta  = st.floats(min_value=0., max_value=1., width=32),
         idx_act = st.integers(min_value=0, max_value=len(tf_activations)-1))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c, alpha, beta, idx_act):

    nn_act = nn_activations[idx_act]
    tf_act = tf_activations[idx_act]

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)

    layer = Shortcut_layer(activation=nn_act, alpha=alpha, beta=beta)

    # Keras Model, double input
    inp1  = Input(batch_shape=inpt1.shape)
    inp2  = Input(batch_shape=inpt2.shape)
    x     = Add()([inp1, inp2])
    out   = Activation(activation=tf_act)(x)
    model = Model(inputs=[inp1, inp2], outputs=out)

    # FORWARD

    forward_out_keras = model.predict([alpha*inpt1, beta*inpt2])

    layer.forward(inpt1, inpt2)
    forward_out_numpynet = layer.output

    assert forward_out_keras.shape == forward_out_numpynet.shape
    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, rtol=1e-5, atol=1e-7)

    # # minor test with different input size (no keras version)
    #
    # inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)
    # inpt2 = np.random.uniform(low=-1., high=1., size=(b, w//2, h//2, c)).astype(float)
    #
    # layer.forward(inpt1, inpt2)
    #
    # assert inpt1.shape == layer.output.shape


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         alpha = st.floats(min_value=0., max_value=1., width=32),
         beta  = st.floats(min_value=0., max_value=1., width=32),
         idx_act = st.integers(min_value=0, max_value=len(tf_activations)-1))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, alpha, beta, idx_act):

    nn_act = nn_activations[idx_act]
    tf_act = tf_activations[idx_act]

    inpt1 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)
    inpt2 = np.random.uniform(low=-1., high=1., size=(b, w, h, c)).astype(float)

    tf_inpt1 = tf.Variable(inpt1)
    tf_inpt2 = tf.Variable(inpt2)

    # numpynet model
    layer = Shortcut_layer(activation=nn_act, alpha=alpha, beta=beta)

    # Keras Model, double input
    inp1  = Input(batch_shape=inpt1.shape)
    inp2  = Input(batch_shape=inpt2.shape)
    x     = Add()([inp1, inp2])
    out   = Activation(activation=tf_act)(x)
    model = Model(inputs=[inp1, inp2], outputs=out)

    # Try backward:
    with pytest.raises(NotFittedError):
      delta      = np.zeros(shape=inpt1.shape, dtype=float)
      prev_delta = np.zeros(shape=inpt2.shape, dtype=float)
      layer.backward(delta, prev_delta)

    # FORWARD

    # Perform Add() for alpha*inpt and beta*inpt
    with tf.GradientTape(persistent=True) as tape :
      preds = model([alpha*tf_inpt1, beta*tf_inpt2])
      grad1 = tape.gradient(preds, tf_inpt1)
      grad2 = tape.gradient(preds, tf_inpt2)

      forward_out_keras = preds.numpy()
      delta1 = grad1.numpy()
      delta2 = grad2.numpy()


    layer.forward(inpt1,inpt2)
    forward_out_numpynet = layer.output

    assert forward_out_keras.shape == forward_out_numpynet.shape
    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, rtol=1e-5, atol=1e-7)

    # BACKWARD

    delta      = np.zeros(shape=inpt1.shape, dtype=float)
    prev_delta = np.zeros(shape=inpt2.shape, dtype=float)

    layer.delta = np.ones(shape=(b, w, h, c), dtype=float)
    layer.backward(delta, prev_delta)

    np.testing.assert_allclose(delta1, delta, rtol=1e-5, atol=1e-8)
    np.testing.assert_allclose(delta2, prev_delta, rtol=1e-5, atol=1e-8)
