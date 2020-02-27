# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.maxpool_layer import Maxpool_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'MaxPool Layer testing'


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

      assert layer.batch == None
      assert layer.w == None
      assert layer.h == None
      assert layer.c == None

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

    layer.batch, layer.w, layer.h, layer.c = (1, 2, 3, 4)

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
    inpt_tf = tf.convert_to_tensor(inpt)

    # NumPyNet model
    layer = Maxpool_layer(size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'SAME'
    else :
      keras_pad = 'VALID'

    out_keras = tf.nn.max_pool2d(input=inpt_tf,
                                 ksize=size, strides=stride,
                                 padding=keras_pad,
                                 data_format='NHWC')

    forward_out_keras = K.eval(out_keras)

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
    inpt_tf = tf.convert_to_tensor(inpt)

    # NumPyNet model
    layer = Maxpool_layer(size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'SAME'
    else :
      keras_pad = 'VALID'

    out_keras = tf.nn.max_pool2d(input=inpt_tf,
                                 ksize=size, strides=stride,
                                 padding=keras_pad,
                                 data_format='NHWC')

    forward_out_keras = K.eval(out_keras)

    # numpynet forward and output
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-6)

    # BACKWARD

    # Compute the gradient of output w.r.t input
    gradient = tf.gradients(out_keras, [inpt_tf])

    # Define a function to evaluate the gradient
    func = K.function([inpt_tf] + [out_keras], gradient)

    # Compute delta for Keras
    delta_keras = func([inpt])[0]

    # Definition of starting delta for numpynet
    layer.delta = np.ones(shape=layer.out_shape, dtype=float)
    delta = np.zeros(shape=inpt.shape, dtype=float)

    # numpynet Backward
    layer.backward(delta)

    # Back tests
    assert delta.shape == delta_keras.shape
    assert delta.shape == inpt.shape
    assert np.allclose(delta, delta_keras, atol=1e-8)
