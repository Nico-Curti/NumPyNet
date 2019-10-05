# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
import keras.backend as K

from NumPyNet.layers.avgpool_layer import Avgpool_layer
from keras.layers import AvgPool2D

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'AvgPool Layer testing'

def test_avgpool_layer():
  '''
  Tests:
    if the average pool layer forwards and backward are consistent with keras

  to be:
  '''
  np.random.seed(123)

  sizes   = [(1, 1), (3, 3), (30, 30)]
  strides = [(1, 1), (2, 2), (20, 20)]


  for size, stride in zip(sizes, strides):
    for pad in [False,True]:

      batch = np.random.randint(low=1, high=10)
      w, h, c = (100, 201, 3)
      inpt = np.random.uniform(0.,1., size=(batch, w, h, c))

      # Numpy_net model
      numpynet = Avgpool_layer(size=size, stride=stride, padding=pad)

      if pad:
        keras_pad = 'same'
      else :
        keras_pad = 'valid'

      # Keras model initialization.
      inp = Input(shape=(w, h, c), batch_shape=inpt.shape)
      x = AvgPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
      model = Model(inputs=[inp], outputs=x)

      # Keras Output
      forward_out_keras = model.predict(inpt)

      # numpynet forward and output
      numpynet.forward(inpt)
      forward_out_numpynet = numpynet.output

      # Test for dimension and allclose of all output
      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-8)

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
