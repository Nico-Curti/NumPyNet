# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from keras.layers import Activation
import keras.backend as K


from NumPyNet.layers.input_layer import Input_layer

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Input Layer testing'


@given(batch = st.integers(min_value=1, max_value=15 ),
       w     = st.integers(min_value=1, max_value=100),
       h     = st.integers(min_value=1, max_value=100),
       c     = st.integers(min_value=1, max_value=10 ))
@settings(max_examples=10,
          deadline=1000)
def test_input_layer(batch, w, h, c):
  '''
  Tests:
    if the forward and the backward of Numpy_net are consistent with keras.

  to be:
  '''

  inpt = np.random.uniform(low=-1, high=1., size=(batch, w, h, c))

  # numpynet model init
  numpynet = Input_layer(input_shape=inpt.shape)

  # Keras Model init
  inp   = Input(shape = (w, h, c), batch_shape = (batch, w, h, c))
  x     = Activation(activation='linear')(inp)
  model = Model(inputs=[inp], outputs=x)

  # FORWARD

  # Keras Forward
  forward_out_keras = model.predict(inpt)

  # numpynet forwrd
  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

  # Forward check (Shape and Values)
  assert forward_out_keras.shape == forward_out_numpynet.shape
  assert np.allclose(forward_out_keras, forward_out_numpynet)

  # BACKWARD

  # Gradient computation (Analytical)
  grad = K.gradients(model.output, [model.input])

  # Define a function to compute the gradient numerically
  func = K.function(model.inputs + [model.output], grad)

  # Keras delta
  keras_delta = func([inpt])[0] # It returns a list with one array inside.

  # numpynet delta init.
  numpynet.delta = np.ones(shape=inpt.shape)

  # Global delta init.
  delta = np.empty(shape=inpt.shape)

  # numpynet Backward
  numpynet.backward(delta)

  # Check dimension and delta
  assert keras_delta.shape == delta.shape
  assert np.allclose(keras_delta, delta)
  # all passed

if __name__ == '__main__':
  test_input_layer()

