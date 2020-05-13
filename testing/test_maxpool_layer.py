# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.maxpool_layer import Maxpool_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestMaxpoolLayer :
  '''
  Tests:
    - costructor of the layer
    - printer function
    - forward against tensorflow
    - backward against tensorflow

  to do:
  '''
  @given(size   = st.integers(min_value=-10, max_value=10),
         stride = st.integers(min_value=0, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, size, stride, pad):

    if size <= 0 :

      with pytest.raises(LayerError):
        layer = Maxpool_layer(size=size, stride=stride, pad=pad)

    else:
      layer = Maxpool_layer(size=size, stride=stride, pad=pad)

      assert layer.size        == (size, size)
      assert len(layer.size)   == 2

      if stride:
        assert layer.stride == (stride, stride)

      else:
        assert layer.size == layer.stride

      assert len(layer.stride) == 2

      assert layer.delta   == None
      assert layer.output  == None

      assert layer.pad        == pad
      assert layer.pad_left   == 0
      assert layer.pad_right  == 0
      assert layer.pad_top    == 0
      assert layer.pad_bottom == 0

  @given(size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=1, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, size, stride, pad):

    layer = Maxpool_layer(size=size, stride=stride, pad=pad)

    with pytest.raises(TypeError):
      print(layer)

    layer.input_shape = (1, 2, 3, 4)

    print(layer)


  @given(b = st.integers(min_value=1, max_value=15),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1, max_value=10),
         size   = st.tuples(st.integers(min_value=1, max_value=10),
                            st.integers(min_value=1, max_value=10)),
         stride = st.tuples(st.integers(min_value=1, max_value=10),
                            st.integers(min_value=1, max_value=10)),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c, size, stride, pad):

    inpt    = np.random.uniform(low=-1., high=1., size=(b, w, h, c))

    # NumPyNet model
    layer = Maxpool_layer(input_shape=inpt.shape, size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'SAME'
    else :
      keras_pad = 'VALID'

    model = tf.keras.layers.MaxPool2D( pool_size=size, strides=stride,
                                       padding=keras_pad,
                                       data_format='channels_last')

    forward_out_keras = model(inpt).numpy()

    # numpynet forward and output
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-6)


  @given(b = st.integers(min_value=1, max_value=15),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1, max_value=10),
         size   = st.tuples(st.integers(min_value=1, max_value=10),
                            st.integers(min_value=1, max_value=10)),
         stride = st.tuples(st.integers(min_value=1, max_value=10),
                            st.integers(min_value=1, max_value=10)),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, size, stride, pad):

    inpt    = np.random.uniform(low=0., high=1., size=(b, w, h, c))
    inpt_tf = tf.Variable(inpt)

    # NumPyNet model
    layer = Maxpool_layer(input_shape=inpt.shape, size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'SAME'
    else :
      keras_pad = 'VALID'

    model = tf.keras.layers.MaxPool2D( pool_size=size, strides=stride,
                                       padding=keras_pad,
                                       data_format='channels_last')

    with tf.GradientTape() as tape:
      preds = model(inpt_tf)
      grads = tape.gradient(preds, inpt_tf)

      forward_out_keras = preds.numpy()
      delta_keras = grads.numpy()

    # numpynet forward and output
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-6)

    # BACKWARD

    # Definition of starting delta for numpynet
    layer.delta = np.ones(shape=layer.out_shape, dtype=float)
    delta = np.zeros(shape=inpt.shape, dtype=float)

    # numpynet Backward
    layer.backward(delta)

    # Back tests
    assert delta.shape == delta_keras.shape
    assert delta.shape == inpt.shape
    assert np.allclose(delta, delta_keras, atol=1e-8)
