# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.maxpool_layer import Maxpool_layer

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'MaxPool Layer testing'

@given(batch  = st.integers(min_value=1, max_value=15),
       w      = st.integers(min_value=15, max_value=100),
       h      = st.integers(min_value=15, max_value=100),
       c      = st.integers(min_value=1, max_value=10),
       size   = st.integers(min_value=1, max_value=10),
       stride = st.integers(min_value=1, max_value=10),
       pad    = st.booleans())
@settings(max_examples=10,
          deadline=None)
def test_maxpool_layer(batch, w, h, c, size, stride, pad):
  '''
  Tests:
    if the NumPyNet maxpool layer forward is consistent with Keras
    if the NumPyNet maxpool layer backward is the same as Keras
      
    both for different sizes, strides and padding values
    
  TODO:
  '''

  inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

  # tensor value of inpt, used to computes gradients 
  inpt_tf = tf.convert_to_tensor(inpt) 

  # Numpy_net model
  numpynet = Maxpool_layer(size=size, stride=stride, padding=pad)

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
  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

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
  numpynet.delta = np.ones(shape=numpynet.out_shape, dtype=float)
  delta = np.zeros(shape=inpt.shape, dtype=float)

  # numpynet Backward
  numpynet.backward(delta)

  # Back tests
  assert delta.shape == delta_keras.shape
  assert delta.shape == inpt.shape
  assert np.allclose(delta, delta_keras, atol=1e-8)

if __name__ == '__main__':
  test_maxpool_layer()

  
  
  
  
