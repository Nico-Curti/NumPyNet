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

activation = [Elliot, Elu, Hardtan,
              Leaky, Lhtan, Linear, Loggy, Logistic,
              Plse, Ramp, Relie, Relu,
              Selu, SoftPlus, SoftSign, Stair,
              SymmElliot, Tanh]

nn_activations = [Relu,   Logistic,  Tanh,   Linear,   Elu,   Selu,   Hardtan,
                  #, SoftSign,   SoftPlus]
                  Leaky]
tf_activations = ['relu', 'sigmoid', 'tanh', 'linear', 'elu', 'selu', 'hard_sigmoid']#, 'softsign', 'softplus']

class TestActivationLayer:
  '''
  Tests:
    - constructor of Activation_layer object
    - print function
    - forward function against tf.keras for different activations
    - backward function against tf.keras for different activations
  '''


  @given(act_fun = st.sampled_from(activation))
  @settings(max_examples=30, deadline=None)
  def test_constructor (self, act_fun):

    layer = Activation_layer(activation=act_fun)

    assert layer.output is None
    assert layer.delta is None

    assert layer.activation is not Activations.activate
    assert layer.gradient is not Activations.gradient



  @given(act_fun = st.sampled_from(activation))
  @settings(max_examples=30, deadline=None)
  def test_printer (self, act_fun):

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
         c     = st.integers(min_value=1, max_value=10 ),
         idx_act = st.integers(min_value=0, max_value=len(tf_activations)-1)
         )
  @settings(max_examples=100,
            deadline=None)
  def test_forward (self, batch, w, h, c, idx_act):

    nn_act = nn_activations[idx_act]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c)).astype(float)

    # numpynet model init
    numpynet = Activation_layer(input_shape=inpt.shape, activation=nn_act)

    # tensorflow model
    if isinstance(nn_act, Leaky):
      model = tf.keras.LeakyReLU()
    else:
      model = tf.keras.layers.Activation(activation=tf_activations[idx_act])

    # FORWARD

    # Keras Forward
    forward_out_keras = model(inpt).numpy()

    # numpynet forwrd
    numpynet.forward(inpt=inpt)
    forward_out_numpynet = numpynet.output

    # Forward check (Shape and Values)
    assert forward_out_keras.shape == forward_out_numpynet.shape
    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, atol=1e-4, rtol=1e-5)


  @given(batch = st.integers(min_value=1, max_value=15 ),
         w     = st.integers(min_value=1, max_value=100),
         h     = st.integers(min_value=1, max_value=100),
         c     = st.integers(min_value=1, max_value=10 ),
         idx_act = st.integers(min_value=0, max_value=len(tf_activations)-1)
         )
  @settings(max_examples=100,
            deadline=None)
  def test_backward (self, batch, w, h, c, idx_act):

    nn_act = nn_activations[idx_act]

    # negative value for Relu testing
    inpt = np.random.uniform(low=-1., high=1., size=(batch, w, h, c)).astype(float)
    tf_input = tf.Variable(inpt)

    # numpynet model init
    numpynet = Activation_layer(input_shape=inpt.shape, activation=nn_act)

    # tensorflow model
    if isinstance(nn_act, Leaky):
      model = tf.keras.LeakyReLU()
    else:
      model = tf.keras.layers.Activation(activation=tf_activations[idx_act])

    # try to backward
    with pytest.raises(NotFittedError):
      # Global delta init.
      delta = np.empty(shape=inpt.shape, dtype=float)

      # numpynet Backward
      numpynet.backward(delta=delta)

    # FORWARD

    # Tensorflow Forward and backward
    with tf.GradientTape() as tape :
      preds = model(tf_input)
      grads = tape.gradient(preds, tf_input)

      forward_out_keras = preds.numpy()
      delta_keras = grads.numpy()

    # numpynet forward
    numpynet.forward(inpt=inpt)
    forward_out_numpynet = numpynet.output

    # Forward check (Shape and Values)
    assert forward_out_keras.shape == forward_out_numpynet.shape
    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, atol=1e-4, rtol=1e-5)

    # BACKWARD

    # numpynet delta init. (Multiplication with gradients)
    numpynet.delta = np.ones(shape=inpt.shape, dtype=float)

    # Global delta init.
    delta = np.empty(shape=inpt.shape, dtype=float)

    # numpynet Backward
    numpynet.backward(delta=delta)

    # Check dimension and delta
    assert delta_keras.shape == delta.shape
    np.testing.assert_allclose(delta_keras, delta, atol=1e-4, rtol=1e-4)
