# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K

from NumPyNet.layers.l2norm_layer import L2Norm_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'L2Norm Layer testing'

class TestL2normLayer :
  '''
  Tests:
    - constructor of the L1norm_layer.
    - printer of the layer.
    - forward against tensorflow.
    - backward against tensorflow.

  to be:
    backward doesn't work
  '''

  def test_costructor (self):

    for axis in [None, 1, 2, 3]:

      layer = L2Norm_layer(axis=axis)

      assert layer.axis   == axis
      assert layer.scales == None
      assert layer.output == None
      assert layer.delta  == None
      assert layer._out_shape == None

  def test_printer (self):

    layer = L2Norm_layer(axis=None)

    with pytest.raises(TypeError):
      print(layer)

    layer.forward(np.random.uniform(size=(1, 2, 3, 4)))

    print(layer)

    layer.forward(np.random.uniform(size=(1,2)))

    with pytest.raises(ValueError):
      print(layer)

  @given(b = st.integers(min_value=3, max_value=15 ), # unstable for low values!
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=2, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c):

    for axis in [None, 1, 2, 3]:

      inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c))
      inpt_tf = tf.convert_to_tensor(inpt)

      # NumPyNet model
      layer = L2Norm_layer(axis=axis)

      sess = tf.Session()

      # Keras output
      output_tf = K.l2_normalize(inpt_tf, axis=axis)

      forward_out_keras = K.eval(output_tf)

      # numpynet forward and output
      layer.forward(inpt)
      forward_out_numpynet = layer.output

      # Test for dimension and allclose of all output
      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-4, rtol=1e-3)
      assert np.allclose(layer.delta, np.zeros(shape=(b, w, h, c)))


  @given(b = st.integers(min_value=3, max_value=15 ), # unstable for low values!
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=2, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c):

    for axis in [None, 1, 2, 3]:

      inpt = np.random.uniform(low=0., high=1., size=(b, w, h, c))
      inpt_tf = tf.convert_to_tensor(inpt)

      # NumPyNet model
      layer = L2Norm_layer(axis=axis)

      sess = tf.Session()

      # Keras output
      output_tf, norm_tf = tf.linalg.normalize(inpt_tf, ord=2, axis=axis)

      forward_out_keras = output_tf.eval(session=sess)

      # numpynet forward and output
      layer.forward(inpt)
      forward_out_numpynet = layer.output

      # Test for dimension and allclose of all output
      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-4, rtol=1e-3)

      # BACKWARD

      grad = K.gradients(output_tf, [inpt_tf])
      func = K.function([inpt_tf] + [output_tf], grad)
      delta_keras = func([inpt])[0]

      # Definition of starting delta for numpynet
      layer.delta = np.zeros(shape=layer.out_shape)
      delta = np.zeros(inpt.shape)

      # numpynet Backward
      layer.backward(delta)

      # Back tests
      assert delta.shape == delta_keras.shape
      assert delta.shape == inpt.shape
      # assert np.allclose(delta, delta_keras, atol=1e-6)


if __name__ == '__main__':

  test = TestL2normLayer()

  test.test_forward()
