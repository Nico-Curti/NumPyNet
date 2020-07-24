# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.exception import NotFittedError
from NumPyNet.layers.shuffler_layer import Shuffler_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example
from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestShuffleLayer :
  '''
  Tests:
    - constructor of the layer
    - printer of the layer
    - forward of the layer against keras
    - backward of the layer against keras

  to be:
  '''

  @given(scale = st.integers(min_value=-3, max_value=10))
  @settings(max_examples=5,
            deadline=None)
  def test_constructor (self, scale):

    if scale <= 1:
      with pytest.raises(ValueError):
        layer = Shuffler_layer(scale=scale)

    else:
      layer = Shuffler_layer(scale=scale)
      assert layer.scale      == scale
      assert layer.scale_step == scale * scale
      assert layer.input_shape == None
      assert layer.output == None
      assert layer.delta  == None


  def test_printer (self):

    scale = 2

    layer = Shuffler_layer(scale=scale)

    with pytest.raises(TypeError):
      print(layer)

    layer.input_shape = (1, 1, 1, 1)
    print(layer)


  @example(couple=(2, 12),  b=5, w=100, h=300)
  @example(couple=(4, 32),  b=5, w=100, h=300)
  @example(couple=(4, 48),  b=5, w=100, h=300)
  @example(couple=(6, 108), b=5, w=100, h=300)
  @given(couple = st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=1, max_value=200)),
         b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),)
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self,couple, b, w, h):

    scale, channels = couple

    # input initialization
    inpt  = np.random.uniform(low=0., high=1., size=(b, w, h, channels))

    # numpynet model
    layer = Shuffler_layer(input_shape=inpt.shape, scale=scale)

    # FORWARD

    if channels % (scale*scale):
      with pytest.raises(ValueError):
        layer.forward(inpt=inpt)

    else:
      layer.forward(inpt=inpt)
      forward_out_numpynet = layer.output

      forward_out_keras = tf.nn.depth_to_space(inpt, block_size=scale, data_format='NHWC')

      assert forward_out_numpynet.shape == forward_out_keras.shape
      np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, rtol=1e-5, atol=1e-8)


  @example(couple=(2, 12),  b=5, w=10, h=30)
  @example(couple=(4, 32),  b=5, w=10, h=30)
  @example(couple=(4, 48),  b=5, w=10, h=30)
  @example(couple=(6, 108), b=5, w=10, h=30)
  @given(couple = st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=1, max_value=100)),
         b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),)
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, couple):

    scale, channels = couple

    # input initialization
    inpt  = np.random.uniform(low=0., high=1., size=(b, w, h, channels))

    # numpynet model
    layer = Shuffler_layer(input_shape=inpt.shape, scale=scale)

    # FORWARD

    if channels % (scale*scale):
      with pytest.raises(ValueError):
        layer.forward(inpt)

    else:

      forward_out_keras = tf.nn.depth_to_space(inpt, block_size=scale, data_format='NHWC')

      # try to BACKWARD
      with pytest.raises(NotFittedError):
        delta = np.random.uniform(low=0., high=1., size=forward_out_keras.shape)
        delta = delta.reshape(inpt.shape)
        layer.backward(delta=delta)


      layer.forward(inpt=inpt)
      forward_out_numpynet = layer.output

      assert forward_out_numpynet.shape == forward_out_keras.shape
      np.testing.assert_allclose(forward_out_numpynet, forward_out_keras, rtol=1e-5, atol=1e-8)

      # BACKWARD

      delta = np.random.uniform(low=0., high=1., size=forward_out_keras.shape).astype(float)

      delta_keras = tf.nn.space_to_depth(delta, block_size=scale, data_format='NHWC')
      inpt_keras  = tf.nn.space_to_depth(forward_out_keras, block_size=scale, data_format='NHWC')

      layer.delta = delta
      delta = delta.reshape(inpt.shape)

      layer.backward(delta=delta)

      assert delta_keras.shape == delta.shape
      np.testing.assert_allclose(delta_keras, delta, rtol=1e-5, atol=1e-8)
      np.testing.assert_allclose(inpt_keras, inpt, rtol=1e-5, atol=1e-8)
