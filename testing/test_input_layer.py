# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

from NumPyNet.exception import NotFittedError
from NumPyNet.layers.input_layer import Input_layer

from random import choice
import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'Input Layer testing'

class TestInputLayer :
  '''
  Tests:
    - constructor of the Input layer object
    - __str__ function of the Input layer
    - forward function againts keras
    - backward function against keras

  to be:
  '''

  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, b, w, h, c):

    input_shape = choice([(b, w, h,c), (b, w, h), b, None])

    if input_shape != (b, w, h, c):
      with pytest.raises(ValueError):
        layer = Input_layer(input_shape=input_shape)

    else:
      layer = Input_layer(input_shape=input_shape)

      layer.batch == b
      layer.w == w
      layer.h == h
      layer.c == c

      assert layer.output == None
      assert layer.delta  == None
      assert layer.out_shape == (b, w, h, c)


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=50,
            deadline=None)
  def test_printer (self, b, w, h, c):

    layer = Input_layer(input_shape=(b, w, h, c))

    print(layer)

    layer.batch = None

    with pytest.raises(TypeError):
      print(layer)

  @given(b = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=20,
            deadline=None)
  def test_forward (self, b, w, h, c):

    inpt = np.random.uniform(low=-1, high=1., size=(b, w, h, c))

    # numpynet model init
    layer = Input_layer(input_shape=inpt.shape)

    # Keras Model init
    inp   = Input(batch_shape=(b, w, h, c))
    x     = Activation(activation='linear')(inp)
    model = Model(inputs=[inp], outputs=x)

    # FORWARD

    # Keras Forward
    forward_out_keras = model.predict(inpt)

    # numpynet forwrd
    layer.forward(inpt)
    forward_out_numpynet = layer.output

    # Forward check (Shape and Values)
    assert forward_out_keras.shape == forward_out_numpynet.shape
    assert np.allclose(forward_out_keras, forward_out_numpynet)


  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=20,
            deadline=None)
  def test_backward (self, b, w, h, c):

    inpt = np.random.uniform(low=-1, high=1., size=(b, w, h, c))

    # numpynet model init
    layer = Input_layer(input_shape=inpt.shape)

    # Keras Model init
    inp   = Input(batch_shape=(b, w, h, c))
    x     = Activation(activation='linear')(inp)
    model = Model(inputs=[inp], outputs=x)

    # FORWARD

    # Keras Forward
    forward_out_keras = model.predict(inpt)

    # layer forward
    layer.forward(inpt)
    forward_out_numpynet = layer.output

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

    # layer delta init.
    layer.delta = np.ones(shape=inpt.shape, dtype=float)

    # Global delta init.
    delta = np.empty(shape=inpt.shape, dtype=float)

    # layer Backward
    layer.backward(delta)

    # Check dimension and delta
    assert keras_delta.shape == delta.shape
    assert np.allclose(keras_delta, delta)

    delta = np.zeros(shape=(1,2,3,4))

    with pytest.raises(ValueError):
      layer.backward(delta)
