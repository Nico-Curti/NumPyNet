# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

# import tensorflow as tf
# import tensorflow.keras.backend as K

from NumPyNet.exception import NotFittedError
from NumPyNet.layers.dropout_layer import Dropout_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestDropoutLayer:
  '''
  Tests:
    - constructor of the layer.
    - printer function.
    - Properties of the output. The tensorflow dropout layer exist, bust due
      to the random nature of the layer, is impossible to test.
    - Check that both forward and backards works.
  '''

  @given(prob = st.floats(min_value=-0.5, max_value=1.5))
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, prob):

    if prob < 0. or prob > 1:

      with pytest.raises(ValueError):
        layer = Dropout_layer(prob)

    else:
      layer = Dropout_layer(prob)

      assert layer.probability == prob

      assert layer.output == None
      assert layer.delta  == None

  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         prob = st.floats(min_value=0., max_value=1.))
  @settings(max_examples=50,
            deadline=None)
  def test_printer (self, b, w, h, c, prob):

    layer = Dropout_layer(input_shape=(b, w, h, c), prob=prob)

    print(layer)

    layer.input_shape = (3.14, w, h, c)

    with pytest.raises(ValueError):
      print(layer)


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         prob = st.floats(min_value=0., max_value=1.))
  @settings(max_examples=20,
            deadline=None)
  def test_forward (self, b, w, h, c, prob):

    # Random input
    inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c)).astype(float)

    # Initialize the numpy_net model
    layer = Dropout_layer(input_shape=inpt.shape, prob=prob)

    # Tensor Flow dropout, just to see if it works
    # forward_out_keras = K.eval(tf.nn.dropout(inpt, seed=None, keep_prob=prob))

    layer.forward(inpt=inpt)
    forward_out_numpynet = layer.output

    zeros_out = np.count_nonzero(forward_out_numpynet)

    if prob == 1.:
      assert zeros_out == 0
      assert not np.all(forward_out_numpynet)

    elif prob == 0.:
      assert zeros_out == b * w * h * c
      np.testing.assert_allclose(forward_out_numpynet, inpt, rtol=1e-5, atol=1e-8)

    else:
      assert forward_out_numpynet.shape == inpt.shape

    np.testing.assert_allclose(layer.delta, np.zeros(shape=(b, w, h, c), dtype=float), rtol=1e-5, atol=1e-8)


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ),
         prob = st.floats(min_value=0., max_value=1.))
  @settings(max_examples=20,
            deadline=None)
  def test_backward (self, b, w, h, c, prob):

    prob = 0.
    # Random input
    inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c)).astype(float)

    # Initialize the numpy_net model
    layer = Dropout_layer(input_shape=inpt.shape, prob=prob)

    # Try to backward
    with pytest.raises(NotFittedError):
      delta = np.zeros(shape=inpt.shape, dtype=float)
      layer.backward(delta)

    # Tensor Flow dropout, just to see if it works
    # forward_out_keras = K.eval(tf.nn.dropout(inpt, seed=None, keep_prob=prob))

    # FORWARD

    layer.forward(inpt)
    forward_out_numpynet = layer.output

    zeros_out = np.count_nonzero(forward_out_numpynet)

    if prob == 1.:
      assert zeros_out == 0
      assert not np.all(forward_out_numpynet)

    elif prob == 0.:
      assert zeros_out == b * w * h * c
      np.testing.assert_allclose(forward_out_numpynet, inpt, rtol=1e-5, atol=1e-8)

    else:
      assert forward_out_numpynet.shape == inpt.shape

    np.testing.assert_allclose(layer.delta, np.zeros(shape=(b, w, h, c), dtype=float), rtol=1e-5, atol=1e-8)

    # BACKWARD

    delta = np.random.uniform(low=0., high=1., size=(b, w, h, c))
    prev_delta = delta.copy()

    layer.backward(delta)

    assert delta.shape == inpt.shape

    if prob == 0.:
      assert np.allclose(delta, prev_delta)

    elif prob == 1.:
      np.testing.assert_allclose(delta, np.zeros(shape=inpt.shape, dtype=float), rtol=1e-5, atol=1e-8)

    else:
      assert ~np.allclose(delta, np.zeros(shape=inpt.shape, dtype=float), rtol=1e-5, atol=1e-8)
