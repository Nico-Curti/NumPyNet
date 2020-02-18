# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.exception import NotFittedError
from NumPyNet.activations import Activations
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.layers.activation_layer import Activation_layer
from tensorflow.keras.layers import Activation

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

np.random.seed(123)

class TestActivationLayer:
  '''
  Tests:
    - if the forward and the backward of Numpy_net are consistent with keras.
    - if all the possible activation functions works with different batch_size
  '''

  def test_constructor (self):

    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    for act_fun in range(0, 4):

      layer = Activation_layer(activation=numpynet_activ[act_fun])

      assert layer.output is None
      assert layer.delta is None
      assert layer._out_shape is None

      assert layer.activation is not Activations.activate
      assert layer.gradient is not Activations.gradient



  def test_printer (self):

    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    for act_fun in range(0, 4):

      layer = Activation_layer(activation=numpynet_activ[act_fun])

      with pytest.raises(TypeError):
        print(layer)

      layer._out_shape = 1
      with pytest.raises(TypeError):
        print(layer)

      layer._out_shape = (1, 2)
      with pytest.raises(ValueError):
        print(layer)

      layer._out_shape = (1, 2, 3)
      with pytest.raises(ValueError):
        print(layer)

      layer._out_shape = (1, 2, 3, 4)
      print(layer)

      assert layer.out_shape == (1, 2, 3, 4)


  @given(batch = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, batch, w, h, c):

    keras_activ = ['relu', 'sigmoid', 'tanh','linear']
    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c))

    for act_fun in range(0, 4):

      # numpynet model init
      numpynet = Activation_layer(activation=numpynet_activ[act_fun])

      # Keras Model init
      inp = Input(batch_shape=(batch, w, h, c))
      x = Activation(activation=keras_activ[act_fun])(inp)
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


  @given(batch = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, batch, w, h, c):

    keras_activ = ['relu', 'sigmoid', 'tanh','linear']
    numpynet_activ = [Relu, Logistic, Tanh, Linear]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c))

    for act_fun in range(0, 4):

      # numpynet model init
      numpynet = Activation_layer(activation=numpynet_activ[act_fun])

      # Keras Model init
      inp = Input(batch_shape=(batch, w, h, c))
      x = Activation(activation=keras_activ[act_fun])(inp)
      model = Model(inputs=[inp], outputs=x)

      # try to backward

      with pytest.raises(NotFittedError):
        # Global delta init.
        delta = np.empty(shape=inpt.shape, dtype=float)

        # numpynet Backward
        numpynet.backward(delta)

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

      # numpynet delta init. (Multiplication with gradients)
      numpynet.delta = np.ones(shape=inpt.shape, dtype=float)

      # Global delta init.
      delta = np.empty(shape=inpt.shape, dtype=float)

      # numpynet Backward
      numpynet.backward(delta)

      # Check dimension and delta
      assert keras_delta.shape == delta.shape
      assert np.allclose(keras_delta, delta)
