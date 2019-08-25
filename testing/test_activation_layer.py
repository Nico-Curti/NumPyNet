# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
import keras.backend as K

from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.layers.activation_layer import Activation_layer
from keras.layers import Activation

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Activation Layer testing'

def test_activation_layer():
  '''
  Tests:
    if the forward and the backward of Numpy_net are consistent with keras.
    if all the possible activation functions works with different batch_size
  to be:
  '''
  np.random.seed(123)

  keras_activ = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activ = [Relu, Logistic, Tanh, Linear]

  batch_sizes = [1,5,10]


  for batch in batch_sizes:
      # negative value for Relu testing
    inpt = np.random.uniform(-1., 1., size=(batch, 100, 100, 3))
    b,w,h,c = inpt.shape

    for act_fun in range(0,4):
      # numpynet model init
      numpynet = Activation_layer(activation=numpynet_activ[act_fun])

      # Keras Model init
      inp = Input(shape = inpt.shape[1:], batch_shape = (b,w,h,c))
      x = Activation(activation = keras_activ[act_fun])(inp)
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

      # numpynet delta init. (Multiplication with gradients)
      numpynet.delta = np.ones(shape=inpt.shape)

      # Global delta init.
      delta = np.empty(shape=inpt.shape)

      # numpynet Backward
      numpynet.backward(delta)

      # Check dimension and delta
      assert keras_delta.shape == delta.shape
      assert np.allclose(keras_delta, delta)
      # all passed

if __name__ == '_main__':
  test_activation_layer()
