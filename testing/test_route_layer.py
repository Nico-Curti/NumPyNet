# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Concatenate
from keras.layers import Activation
from keras.layers import Input
from keras.optimizers import SGD as K_SGD
import keras.backend as K

from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.activation_layer import Activation_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.cost_layer import cost_type
from NumPyNet.optimizer import SGD as N_SGD 
from NumPyNet.network import Network

import numpy as np

from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Route Layer testing'

#@given(w     = st.integers(min_value=10, max_value=200) 
#       h     = st.integers(max_value=10, max_value=200) 
#       c     = st.integers(min_value=10, max_value=200) 
#       batch = st.integers(min_value=1, max_value=32)) 
#@settings(deadline=None,
#          max_examples=10)
def test_route_layer():
  
  np.random.seed(123)

  batch, w, h, c = (1, 5, 5, 3)
  input = np.random.uniform(low=-10, high=10. ,size=(batch, w, h, c)) # from -10 to 10 to see both the effect of Relu and TanH activation

  # init keras model
  global model 
  inp    = Input(shape=(w, h, c), batch_shape=(batch, w, h, c))
  x      = Activation(activation='relu')(inp)
  y      = Activation(activation='tanh')(x)
  Concat = Concatenate( axis=-1)([x, y]) # concatenate of x and y
  model  = Model(inputs=[inp], outputs=Concat)

  # init NumPyNet model
  global net
  net = Network(batch=batch, input_shape=(w, h, c))

  net.add(Activation_layer(activation='relu')) # layer 1
  net.add(Activation_layer(activation='tanh')) # layer 2
  net.add(Route_layer(input_layers=(1,2), by_channels=True))
  
  net.summary()
  net._fitted = True # False control

  # FORWARDS
  
  fwd_out_numpynet = net.predict(X=input)
  fwd_out_keras    = model.predict(x=input, batch_size=batch)
  
  assert np.allclose(fwd_out_keras, fwd_out_numpynet) # ok
  
  net.fitted = False # the correct state of the network
  
  # BACKWARD
  
  # try some derivatives
  
  global delta_keras, delta_numpynet
  gradient    = K.gradients(model.output, model.inputs)
  func        = K.function(model.inputs + model.outputs ,gradient)
  delta_keras = func([input])[0]
  
  net._net[3].delta = np.ones(shape=fwd_out_numpynet.shape)
  net._backward(X=input)
  
  delta_numpynet = net._net[0].delta
  
  # i don't know how to test it for now
  
#  assert np.allclose(delta_keras, delta_numpynet) 
  

if __name__ == '__main__':
   test_route_layer()


  
  
  
  
  
