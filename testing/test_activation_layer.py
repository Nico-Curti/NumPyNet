# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.exception import NotFittedError
from NumPyNet.activations import Activations
from NumPyNet.activations import Logistic
from NumPyNet.activations import Loggy
from NumPyNet.activations import Relu
from NumPyNet.activations import Elu
from NumPyNet.activations import Relie
from NumPyNet.activations import Ramp
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.activations import Plse
from NumPyNet.activations import Leaky
from NumPyNet.activations import Stair
from NumPyNet.activations import Hardtan
from NumPyNet.activations import Lhtan
from NumPyNet.activations import Selu
from NumPyNet.activations import Elliot
from NumPyNet.activations import SymmElliot
from NumPyNet.activations import SoftPlus
from NumPyNet.activations import SoftSign
from NumPyNet.layers.activation_layer import Activation_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestActivationLayer:
  '''
  Tests:
    - constructor of Activation_layer object
    - print function
    - forward function against tf.keras for different activations
    - backward function against tf.keras for different activations
  '''

  def test_constructor (self):

    numpynet_activ = [Elliot, Elu, Hardtan,
                     Leaky, Lhtan, Linear, Loggy, Logistic,
                     Plse, Ramp, Relie, Relu,
                     Selu, SoftPlus, SoftSign, Stair,
                     SymmElliot, Tanh]

    for act_fun in numpynet_activ:

      layer = Activation_layer(activation=act_fun)

      assert layer.output is None
      assert layer.delta is None

      assert layer.activation is not Activations.activate
      assert layer.gradient is not Activations.gradient



  def test_printer (self):

    numpynet_activ = [Elliot, Elu, Hardtan,
                     Leaky, Lhtan, Linear, Loggy, Logistic,
                     Plse, Ramp, Relie, Relu,
                     Selu, SoftPlus, SoftSign, Stair,
                     SymmElliot, Tanh]

    for act_fun in numpynet_activ:

      layer = Activation_layer(activation=act_fun)

      with pytest.raises(TypeError):
        print(layer)

      layer.input_shape = 1
      with pytest.raises(TypeError):
        print(layer)

      layer.input_shape = (1, 2)
      with pytest.raises(ValueError):
        print(layer)

      layer.input_shape = (1, 2, 3)
      with pytest.raises(ValueError):
        print(layer)

      layer.input_shape = (1, 2, 3, 4)
      print(layer)

      assert layer.out_shape == (1, 2, 3, 4)


  @given(batch = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, batch, w, h, c):

    keras_activ = ['relu', 'sigmoid', 'tanh','linear'] # 'softplus', 'softsign', 'elu', 'selu']
    numpynet_activ = [Relu, Logistic, Tanh, Linear] # SoftPlus, SoftSign, Elu, Selu]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c))

    for act_fun in range(0, len(keras_activ)):

      # numpynet model init
      numpynet = Activation_layer(input_shape=inpt.shape, activation=numpynet_activ[act_fun])

      # tensorflow model
      model = tf.keras.layers.Activation(activation=keras_activ[act_fun])

      # FORWARD

      # Keras Forward
      forward_out_keras = model(inpt).numpy()

      # numpynet forwrd
      numpynet.forward(inpt)
      forward_out_numpynet = numpynet.output

      # Forward check (Shape and Values)
      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-4)


  @given(batch = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, batch, w, h, c):

    keras_activ = ['relu', 'sigmoid', 'tanh','linear'] # 'elu', 'selu', 'softsign', 'softplus']
    numpynet_activ = [Relu, Logistic, Tanh, Linear] # Elu,  Selu, SoftSign, SoftPlus]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c))
    tf_input = tf.Variable(inpt)

    for act_fun in range(0, len(keras_activ)):

      # numpynet model init
      numpynet = Activation_layer(input_shape=inpt.shape, activation=numpynet_activ[act_fun])

      # Keras Model init
      model = tf.keras.layers.Activation(activation=keras_activ[act_fun])

      # try to backward
      with pytest.raises(NotFittedError):
        # Global delta init.
        delta = np.empty(shape=inpt.shape, dtype=float)

        # numpynet Backward
        numpynet.backward(delta)

      # FORWARD

      # Tensorflow Forward and backward
      with tf.GradientTape() as tape :
        preds = model(tf_input)
        grads = tape.gradient(preds, tf_input)

        forward_out_keras = preds.numpy()
        delta_keras = grads.numpy()

      # numpynet forwrd
      numpynet.forward(inpt)
      forward_out_numpynet = numpynet.output

      # Forward check (Shape and Values)
      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet)

      # BACKWARD

      # numpynet delta init. (Multiplication with gradients)
      numpynet.delta = np.ones(shape=inpt.shape, dtype=float)

      # Global delta init.
      delta = np.empty(shape=inpt.shape, dtype=float)

      # numpynet Backward
      numpynet.backward(delta)

      # Check dimension and delta
      assert delta_keras.shape == delta.shape
      assert np.allclose(delta_keras, delta, atol=1e-7)
