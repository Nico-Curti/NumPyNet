#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.layers.batchnorm_layer import BatchNorm_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestBatchnormLayer:
  '''
  Tests:
    - costructor of Batchnorm_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    update function.
  '''

  @given(b = st.integers(min_value=3, max_value=15 ),
         w = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         h = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_constructor(self, b, w, h, c):

    input_shape = choice([None, (b, w, h, c)])
    scales = choice([None, np.random.uniform(size=(w, h, c))])
    bias   = choice([None, np.random.uniform(size=(w, h, c))])

    layer = BatchNorm_layer(scales=scales, bias=bias, input_shape=input_shape)

    try :
      assert np.allclose(layer.scales, scales)
    except TypeError:
      assert layer.scales == None

    try :
      assert np.allclose(layer.bias, scales)
    except TypeError:
      assert layer.bias == None

    assert layer.input_shape == input_shape


  @given(b = st.integers(min_value=3, max_value=15 ),
         w = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         h = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_printer(self, b, w, h, c):

    layer = BatchNorm_layer()

    with pytest.raises(TypeError):
      print(layer)

    layer = BatchNorm_layer(input_shape=(b, w, h, c))

    print(layer)

  @given(b = st.integers(min_value=3, max_value=15 ),
         w = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         h = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_forward(self, b, w, h, c):

    inpt = np.random.uniform(low=1., high=10., size=(b, w, h, c)).astype(np.float32)

    bias   = np.random.uniform(low=0., high=1., size=(w, h, c)) # random biases
    scales = np.random.uniform(low=0., high=1., size=(w, h, c)) # random scales

    # inpt_tf = tf.convert_to_tensor(inpt.astype('float32'))

    # Numpy_net model
    numpynet = BatchNorm_layer(input_shape=inpt.shape, scales=scales, bias=bias)

    # initializers must be callable with this syntax, I need those for dimensionality problems
    def bias_init(shape, **kwargs):
      return np.expand_dims(bias, axis=0)

    def gamma_init(shape, dtype=None):
      return np.expand_dims(scales, axis=0)

    def mean_init(shape, dtype=None):
      return np.expand_dims(inpt.mean(axis=0), axis=0)

    def var_init(shape, dtype=None):
      return np.expand_dims(inpt.var(axis=0), axis=0)

    # Tensorflow Layer
    model = tf.keras.layers.BatchNormalization(momentum=1., epsilon=1e-8, center=True, scale=True,
                                               axis=[1, 2, 3],
                                               beta_initializer=bias_init,
                                               gamma_initializer=gamma_init,
                                               moving_mean_initializer=mean_init,
                                               moving_variance_initializer=var_init)

    # Keras forward
    forward_out_keras = model(inpt).numpy()

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    # Comparing outputs
    assert forward_out_numpynet.shape == (b, w, h, c)
    assert forward_out_numpynet.shape == forward_out_keras.shape            # same shape
    assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-3)  # same output

    x_norm = (numpynet.x - numpynet.mean)*numpynet.var

    # Own variable updates comparisons
    assert np.allclose(numpynet.x, inpt)
    assert numpynet.mean.shape == (w, h, c)
    assert numpynet.var.shape == (w, h, c)
    assert x_norm.shape == numpynet.x.shape
    assert np.allclose(numpynet.x_norm, x_norm)


  @given(b = st.integers(min_value=3, max_value=15 ),
         w = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         h = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_backward(self, b, w, h, c):

    inpt = np.random.uniform(low=1., high=10., size=(b, w, h, c))

    bias   = np.random.uniform(low=0., high=1., size=(w, h, c)) # random biases
    scales = np.random.uniform(low=0., high=1., size=(w, h, c)) # random scales

    tf_input = tf.Variable(inpt.astype('float32'))

    # Numpy_net model
    numpynet = BatchNorm_layer(input_shape=inpt.shape, scales=scales, bias=bias)

    # initializers must be callable with this syntax, I need those for dimensionality problems
    def bias_init(shape, **kwargs):
      return np.expand_dims(bias, axis=0)

    def gamma_init(shape, dtype=None):
      return np.expand_dims(scales, axis=0)

    def mean_init(shape, dtype=None):
      return np.expand_dims(inpt.mean(axis=0), axis=0)

    def var_init(shape, dtype=None):
      return np.expand_dims(inpt.var(axis=0), axis=0)

    # Keras Model
    model = tf.keras.layers.BatchNormalization(momentum=1., epsilon=1e-8, center=True, scale=True,
                                               trainable=True,
                                               axis=[1, 2, 3],
                                               beta_initializer=bias_init,
                                               gamma_initializer=gamma_init,
                                               moving_mean_initializer=mean_init,
                                               moving_variance_initializer=var_init)


    # Tensorflow forward and backward
    with tf.GradientTape(persistent=True) as tape :
      preds = model(tf_input)
      grad1 = tape.gradient(preds, tf_input)
      grad2 = tape.gradient(preds, model.trainable_weights)

      forward_out_keras = preds.numpy()
      delta_keras = grad1.numpy()
      updates     = grad2

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    # Comparing outputs
    assert forward_out_numpynet.shape == (b, w, h, c)
    assert forward_out_numpynet.shape == forward_out_keras.shape            # same shape
    assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-3)  # same output

    x_norm = (numpynet.x - numpynet.mean)*numpynet.var

    # Own variable updates comparisons
    assert np.allclose(numpynet.x, inpt)
    assert numpynet.mean.shape == (w, h, c)
    assert numpynet.var.shape == (w, h, c)
    assert x_norm.shape == numpynet.x.shape
    assert np.allclose(numpynet.x_norm, x_norm)

    # BACKWARD

    # Initialization of numpynet delta to one (multiplication) and an empty array to store values
    numpynet.delta = np.ones(shape=inpt.shape, dtype=float)
    delta_numpynet = np.zeros(shape=inpt.shape, dtype=float)

    # numpynet bacward, updates delta_numpynet
    numpynet.backward(delta_numpynet)

    # Testing delta, the precision change with the image
    assert delta_keras.shape == delta_numpynet.shape
    assert np.allclose(delta_keras, delta_numpynet, rtol=1e-1, atol=1e-1)

    # Testing scales updates
    assert updates[0][0].shape == numpynet.scales_update.shape
    assert np.allclose(updates[0], numpynet.scales_update, atol=1e-03)

    # Testing Bias updates
    assert updates[1][0].shape == numpynet.bias_update.shape
    assert np.allclose(updates[1], numpynet.bias_update, atol=1e-06)

    # All passed, but precision it's not consistent, missing update functions
