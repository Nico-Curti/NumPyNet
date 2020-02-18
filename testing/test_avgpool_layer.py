# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.layers.avgpool_layer import Avgpool_layer
from tensorflow.keras.layers import AvgPool2D

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'AvgPool Layer testing'

@given(batch  = st.integers(min_value=1, max_value=15),
       w      = st.integers(min_value=15, max_value=100),
       h      = st.integers(min_value=15, max_value=100),
       c      = st.integers(min_value=1, max_value=10),
       size   = st.integers(min_value=1, max_value=10),
       stride = st.integers(min_value=1, max_value=10),
       pad    = st.booleans())
@settings(max_examples=10,
          deadline=None)
def test_avgpool_layer(batch, w, h, c, size, stride, pad):
  '''
  Tests:
    if the average pool layer forwards and backward are consistent with keras

  to be:
  '''

  inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

  # Numpy_net model
  numpynet = Avgpool_layer(size=size, stride=stride, padding=pad)

  if pad:
    keras_pad = 'same'
  else :
    keras_pad = 'valid'

  # Keras model initialization.
  inp = Input(batch_shape=inpt.shape)
  x = AvgPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
  model = Model(inputs=[inp], outputs=x)

  # Keras Output
  forward_out_keras = model.predict(inpt)

  # numpynet forward and output
  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

  # Test for dimension and allclose of all output
  assert forward_out_numpynet.shape == forward_out_keras.shape
  assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-6)

  # BACKWARD

  # Compute the gradient of output w.r.t input
  gradient = K.gradients(model.output, [model.input])

  # Define a function to evaluate the gradient
  func = K.function(model.inputs + [model.output], gradient)

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

  test_avgpool_layer()
