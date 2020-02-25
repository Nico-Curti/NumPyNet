# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Concatenate
from tensorflow.keras.layers import Activation
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.activation_layer import Activation_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.cost_layer import cost_type
from NumPyNet.optimizer import SGD
from NumPyNet.network import Network

import numpy as np
import pytest
from random import choice
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestRouteLayer :
  '''
  Tests
    - costructor of the Route Layer Object
    - printer of the route layer object
    - forward of the route layer object

  to be:
    - backward of the route layer object
  '''

  @given(input_layers = st.tuples(st.integers(min_value=1, max_value=10),
                                  st.integers(min_value=1, max_value=10)),
         by_channels  = st.booleans())

  def test_constructor (self, input_layers, by_channels):

    input_layers = choice([input_layers[0], input_layers, None])

    if input_layers:
      layer = Route_layer(input_layers=input_layers, by_channels=by_channels)

      assert layer.output.size == 0
      assert layer._out_shape   == None
      assert (layer.input_layers == input_layers or
              layer.input_layers == input_layers[0])

      if by_channels :
        assert layer.axis == 3
      else:
        assert layer.axis == 0

    else:
      with pytest.raises(ValueError):
        layer = Route_layer(input_layers=input_layers, by_channels=by_channels)


  @given(b = st.integers(min_value=5, max_value=15),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1, max_value=10))
  @settings(max_examples=3,
            deadline=None)
  def test_printer (self, b, w, h, c):

    net = Network(batch=b, input_shape=(w, h, c))
    net.add(Activation_layer(activation='relu')) # layer 1
    net.add(Activation_layer(activation='tanh')) # layer 2
    net.add(Route_layer(input_layers=(1,2), by_channels=True))
    net.add(Cost_layer(cost_type='mse', scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0.))
    net.compile(optimizer=SGD())

    net.summary()

  @given(b = st.integers(min_value=5, max_value=15),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_forward (self, b, w, h, c):

    input = np.random.uniform(low=-10, high=10., size=(b, w, h, c))

    # init keras model
    inp    = Input(batch_shape=(b, w, h, c))
    x      = Activation(activation='relu')(inp)
    y      = Activation(activation='tanh')(x)
    Concat = Concatenate( axis=-1)([x, y]) # concatenate of x and y
    model  = Model(inputs=[inp], outputs=Concat)
    model.compile(optimizer='sgd', loss='mse')

    # init NumPyNet model
    net = Network(batch=b, input_shape=(w, h, c))
    net.add(Activation_layer(activation='relu')) # layer 1
    net.add(Activation_layer(activation='tanh')) # layer 2
    net.add(Route_layer(input_layers=(1,2), by_channels=True))
    net.add(Cost_layer(cost_type='mse', scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0.))
    net.compile(optimizer=SGD())

    net.summary()

    assert net._fitted == False
    net._fitted = True # False control

    # FORWARDS

    fwd_out_numpynet = net.predict(X=input)
    fwd_out_keras    = model.predict(x=input, batch_size=b)

    assert np.allclose(fwd_out_keras, fwd_out_numpynet) # ok


  @given(b = st.integers(min_value=5, max_value=15),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_backward (self, b, w, h, c):

    # TODO: test backward correctly

    input = np.random.uniform(low=-10, high=10. ,size=(b, w, h, c))

    # init keras model
    inp    = Input(batch_shape=(b, w, h, c))
    x      = Activation(activation='relu')(inp)
    y      = Activation(activation='tanh')(x)
    Concat = Concatenate( axis=-1)([x, y]) # concatenate of x and y
    model  = Model(inputs=[inp], outputs=Concat)
    model.compile(optimizer='sgd', loss='mse')

    # init NumPyNet model
    net = Network(batch=b, input_shape=(w, h, c))
    net.add(Activation_layer(activation='relu')) # layer 1
    net.add(Activation_layer(activation='tanh')) # layer 2
    net.add(Route_layer(input_layers=(1,2), by_channels=True))
    net.add(Cost_layer(cost_type='mse', scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0.))
    net.compile(optimizer=SGD())

    net._fitted = True

    # FORWARDS

    fwd_out_numpynet = net.predict(X=input)
    fwd_out_keras    = model.predict(x=input, batch_size=b)

    assert np.allclose(fwd_out_keras, fwd_out_numpynet) # ok

    net._fitted = False

    # BACKWARD

    # try some derivatives
    gradient    = K.gradients(model.output, model.inputs)
    func        = K.function(model.inputs + model.outputs ,gradient)
    delta_keras = func([input])[0]

    net._net[3].delta = np.ones(shape=fwd_out_numpynet.shape)
    net._backward(X=input)

    delta_numpynet = net._net[0].delta

    assert delta_numpynet.shape == delta_keras.shape
    # assert np.allclose(delta_keras, delta_numpynet)
