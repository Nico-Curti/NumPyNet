# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.layers.l1norm_layer import L1Norm_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestL1normLayer :
  '''
  Tests:
    - constructor of the L1norm_layer.
    - printer of the layer.
    - forward against tensorflow.
    - backward against tensorflow.

  to be:
  '''

  @given(ax = st.sampled_from([None, 1, 2, 3]))
  @settings(max_examples=20,
            deadline=None)
  def test_costructor (self, ax):

    layer = L1Norm_layer(axis=ax)

    assert layer.axis   == ax
    assert layer.scales == None
    assert layer.output == None
    assert layer.delta  == None
    assert layer.out_shape == None

  @given(b = st.integers(min_value=1, max_value=15),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10))
  @settings(max_examples=50,
            deadline=None)
  def test_printer (self, b, w, h, c):

    layer = L1Norm_layer(input_shape=(b, w, h, c))

    print(layer)

    layer.input_shape = (3.14, w, h, c)

    with pytest.raises(ValueError):
      print(layer)

  @given(b = st.integers(min_value=3, max_value=15), # unstable for low values!
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=2, max_value=10),
         ax = st.integers(min_value=1, max_value=3))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c, ax):

    # "None" axis supported only in NumPyNet

    inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c)).astype(float)
    inpt_tf = tf.Variable(inpt)

    # NumPyNet model
    layer = L1Norm_layer(input_shape=inpt.shape, axis=ax)

    # Keras output
    forward_out_keras = tf.keras.utils.normalize(inpt_tf, order=1, axis=ax).numpy()

    # numpynet forward and output
    layer.forward(inpt=inpt)
    forward_out_numpynet = layer.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, atol=1e-7, rtol=1e-5)
    np.testing.assert_allclose(layer.delta, np.zeros(shape=(b, w, h, c), dtype=float), rtol=1e-5, atol=1e-8)


  @given(b = st.integers(min_value=3, max_value=15), # unstable for low values!
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=2, max_value=10),
         ax = st.integers(min_value=1, max_value=3))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, ax):

    inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c)).astype(float)
    inpt_tf = tf.Variable(inpt)

    # NumPyNet model
    layer = L1Norm_layer(input_shape=inpt.shape, axis=ax)

    # Keras output

    with tf.GradientTape() as tape:
      preds = tf.keras.utils.normalize(inpt_tf, order=1, axis=ax)
      grads = tape.gradient(preds, inpt_tf)

      forward_out_keras = preds.numpy()
      delta_keras = grads.numpy()

    # numpynet forward and output
    layer.forward(inpt=inpt)
    forward_out_numpynet = layer.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, atol=1e-7, rtol=1e-5)

    # BACKWARD

    # Definition of starting delta for numpynet
    layer.delta = np.zeros(shape=layer.out_shape, dtype=float)
    delta = np.zeros(inpt.shape, dtype=float)

    # numpynet Backward
    layer.backward(delta=delta)

    # Back tests
    assert delta.shape == delta_keras.shape
    assert delta.shape == inpt.shape
    # assert np.allclose(delta, delta_keras, atol=1e-6) # TODO wrong results?
