# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Input
import keras.backend as K

from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.activation_layer import Activation_layer
from NumPyNet.network import Network


import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Route Layer testing'

# @given()
# @settings(deadline=None)
def test_route_layer():
  
  np.random.seed(123)

  batch, w, h, c = (5, 100, 200, 3)
  input = np.random.uniform(low=-10, high=10. ,size=(batch, w, h, c)) # from -10 to 10 to see both the effect of Relu and TanH activation

  # init keras model
  inp    = Input(shape=(w, h, c), batch_shape=(batch, w, h, c))
  x      = Activation(activation='relu')(inp)
  y      = Activation(activation='tanh')(x)
  Concat = Concatenate( axis=-1)([x, y]) # concatenate of x and y
  model  = Model(inputs=[inp], outputs=Concat)

  # init NumPyNet model
  net = Network(batch=batch, input_shape=(w, h, c))

  net.add(Activation_layer(activation='relu')) # layer 1
  net.add(Activation_layer(activation='tanh')) # layer 2
  net.add(Route_layer(input_layers=(1,2), by_channels=True))

  net._fitted = True # False control

  # FORWARDS
  
  fwd_out_numpynet = net.predict(X=input)
  fwd_out_keras    = model.predict(x=input, batch_size=batch)
  
  assert np.allclose(fwd_out_keras, fwd_out_numpynet) # ok
  
  # BACKWARDS, try with a cost layer.

if __name__ == '__main__':
   test_route_layer()


  
  
  
  
  
