# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.avgpool_layer import Avgpool_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestAvgpoolLayer:
  '''
  Tests:
    - costructor of Avgpool_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras
  '''

  @given(size   = st.integers(min_value=-10, max_value=10),
         stride = st.integers(min_value=0, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, size, stride, pad):

    if size <= 0 :

      with pytest.raises(LayerError):
        layer = Avgpool_layer(size=size, stride=stride, pad=pad)

    else:
      layer = Avgpool_layer(size=size, stride=stride, pad=pad)

      assert layer.size      == (size, size)
      assert len(layer.size) == 2

      if stride:
        assert layer.stride == (stride, stride)
      else:
        assert layer.size == layer.stride

      assert len(layer.stride) == 2

      assert layer.delta  == None
      assert layer.output == None

      assert layer.pad        == pad
      assert layer.pad_left   == 0
      assert layer.pad_right  == 0
      assert layer.pad_top    == 0
      assert layer.pad_bottom == 0


  @given(size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=0, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, size, stride, pad):

    layer = Avgpool_layer(size=size, stride=stride, pad=pad)

    with pytest.raises(TypeError):
      print(layer)

    layer.input_shape = (1, 2, 3, 4)

    print(layer)

  @given(batch  = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10),
         size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=1, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, batch, w, h, c, size, stride, pad):

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c)).astype(float)

    # Numpy_net model
    numpynet = Avgpool_layer(input_shape=inpt.shape, size=size, stride=stride, pad=pad)

    keras_pad = 'same' if pad else 'valid'

    # Keras model initialization.
    model = tf.keras.layers.AveragePooling2D(pool_size=(size, size), strides=stride, padding=keras_pad, data_format='channels_last')

    # Keras Output
    forward_out_keras = model(inpt).numpy()

    # numpynet forward and output
    numpynet.forward(inpt=inpt)
    forward_out_numpynet = numpynet.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, rtol=1e-5, atol=1e-8)


  @given(batch  = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10),
         size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=1, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, batch, w, h, c, size, stride, pad):

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c)).astype(float)
    tf_input = tf.Variable(inpt)

    # Numpy_net model
    numpynet = Avgpool_layer(input_shape=inpt.shape, size=size, stride=stride, pad=pad)

    keras_pad = 'same' if pad else 'valid'

    # Keras model initialization.
    model = tf.keras.layers.AveragePooling2D(pool_size=(size, size), strides=stride, padding=keras_pad, data_format='channels_last')

    # Keras Output
    with tf.GradientTape() as tape :
      preds = model(tf_input)
      grads = tape.gradient(preds, tf_input)

      forward_out_keras = preds.numpy()
      delta_keras = grads.numpy()


    # try to backward
    with pytest.raises(NotFittedError):
      # Global delta init.
      delta = np.empty(shape=inpt.shape, dtype=float)

      # numpynet Backward
      numpynet.backward(delta=delta)

    # numpynet forward and output
    numpynet.forward(inpt=inpt)
    forward_out_numpynet = numpynet.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, rtol=1e-5, atol=1e-8)

    # BACKWARD

    # Definition of starting delta for numpynet
    numpynet.delta = np.ones(shape=numpynet.out_shape, dtype=float)
    delta = np.zeros(shape=inpt.shape, dtype=float)

    # numpynet Backward
    numpynet.backward(delta=delta)

    # Back tests
    assert delta.shape == delta_keras.shape
    assert delta.shape == inpt.shape
    np.testing.assert_allclose(delta, delta_keras, rtol=1e-5, atol=1e-8)
