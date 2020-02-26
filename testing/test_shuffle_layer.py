# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K

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

  @given(scale=st.integers(min_value=-3, max_value=10))
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
      assert layer.batch == None
      assert layer.w     == None
      assert layer.h     == None
      assert layer.c     == None
      assert layer.output == None
      assert layer.delta  == None


  def test_printer (self):

    scale = 2

    layer = Shuffler_layer(scale=scale)

    with pytest.raises(TypeError):
      print(layer)

    layer.batch, layer.w, layer.h, layer.c = (1,1,1,1)
    print(layer)

    layer = Shuffler_layer(scale=scale)

    layer.forward(np.random.uniform(size=(5,10,10,12)))
    print(layer)


  @example(couple=(2,12),  b=5, w=100, h=300)
  @example(couple=(4,32),  b=5, w=100, h=300)
  @example(couple=(4,48),  b=5, w=100, h=300)
  @example(couple=(6,108), b=5, w=100, h=300)
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
    layer = Shuffler_layer(scale=scale)

    # FORWARD

    if channels % (scale*scale):
      with pytest.raises(ValueError):
        layer.forward(inpt)

    else:
      layer.forward(inpt)
      forward_out_numpynet = layer.output

      forward_out_keras = K.eval(tf.depth_to_space(inpt, block_size=scale))

      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras)


  @given(couple = st.tuples(st.integers(min_value=2, max_value=10), st.integers(min_value=1, max_value=100)),
         b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),)
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, couple):

    couples = choice([(2,12), (4,32), (4,48), (6,108), couple])

    scale, channels = couples

    # input initialization
    inpt  = np.random.uniform(low=0., high=1., size=(b, w, h, channels))

    # numpynet model
    layer = Shuffler_layer(scale=scale)

    # FORWARD

    if channels % (scale*scale):
      with pytest.raises(ValueError):
        layer.forward(inpt)

    else:

      forward_out_keras = K.eval(tf.depth_to_space(inpt, block_size=scale))

      # try to BACKWARD
      with pytest.raises(NotFittedError):
        delta = np.random.uniform(low=0., high=1., size=forward_out_keras.shape)
        delta = delta.reshape(inpt.shape)
        layer.backward(delta)


      layer.forward(inpt)
      forward_out_numpynet = layer.output

      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras)

      # BACKWARD

      delta = np.random.uniform(low=0., high=1., size=forward_out_keras.shape)

      delta_keras = K.eval(tf.space_to_depth(delta, block_size=scale))

      layer.delta = delta
      delta = delta.reshape(inpt.shape)

      layer.backward(delta)

      assert delta_keras.shape == delta.shape
      assert np.allclose(delta_keras, delta)
