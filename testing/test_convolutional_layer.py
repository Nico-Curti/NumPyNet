# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.activations import Activations
from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from tensorflow.keras.layers import Conv2D

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice
import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'Convolutional Layer testing'


class TestConvolutionalLayer :
  '''
  Tests:
    - costructor of Convolutional_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    update function
  '''


  @given(b = st.integers(min_value=1,  max_value=10),
         w = st.integers(min_value=30, max_value=100),
         h = st.integers(min_value=30, max_value=100),
         c = st.integers(min_value=1,  max_value=10),
         filters = st.integers(min_value=-5, max_value=10),
         size1   = st.integers(min_value=-5, max_value=10),
         size2   = st.integers(min_value=-5, max_value=10),
         stride  = st.integers(min_value=0, max_value=10),
         pad     = st.booleans())
  @settings(max_examples=50,
            deadline=None)
  def test_constructor (self, b, w, h, c, filters, size1, size2, stride, pad):

    numpynet_activ = [Relu, Logistic, Tanh, Linear]
    input_shape    = (b, w, h, c)

    size   = choice([(size1, size2), size1, size2])
    stride = choice([stride, None])

    if filters <= 0:
      with pytest.raises(ValueError):
        layer = Convolutional_layer(filters=filters, size=1)
      filters += 10

    if hasattr(size, '__iter__'):
      if size[0] < 0 or size[1] < 0:
        with pytest.raises(LayerError):
          layer = Convolutional_layer(filters=filters, size=size)
        size[0] += 10
        size[1] += 10

      weights_choice = [np.random.uniform(low=-1, high=1., size=(size[0], size[1], c, filters)), None]

    else :
      if size <= 0:
        with pytest.raises(LayerError):
          layer = Convolutional_layer(filters=filters, size=size)
        size += 10
      weights_choice = [np.random.uniform(low=-1, high=1., size=(size, size, c, filters)), None]

    if stride:
      if stride < 0:
        with pytest.raises(LayerError):
          layer = Convolutional_layer(filters=filters, size=size, stride=stride)
          stride += 10

    bias_choice = [np.random.uniform(low=-1, high=1., size=(filters,)), None]

    weights = choice(weights_choice)
    bias    = choice(bias_choice)

    for activ in numpynet_activ:

      layer = Convolutional_layer(filters=filters,
                                  size=size, stride=stride, pad=pad,
                                  weights=weights, bias=bias,
                                  input_shape=input_shape,
                                  activations=activ)

      if weights is not None:
        assert np.allclose(layer.weights, weights)
      else:
        assert (layer.weights is not None)

      if bias is None:
        assert np.allclose(layer.bias, np.zeros(shape=(filters,)))
      else:
        assert np.allclose(layer.bias, bias)

      assert layer.delta   == None
      assert layer.output  == None
      assert layer.weights_update == None
      assert layer.bias_update    == None
      assert layer.optimizer      == None

      assert layer.batch == b
      assert layer.w == w
      assert layer.h == h
      assert layer.c == c

      assert layer.pad        == pad
      assert layer.pad_left   == 0
      assert layer.pad_right  == 0
      assert layer.pad_top    == 0
      assert layer.pad_bottom == 0


  @given(b = st.integers(min_value=1,  max_value=10),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1,  max_value=10),
         filters = st.integers(min_value=1, max_value=10),
         size    = st.integers(min_value=1, max_value=10),
         stride  = st.integers(min_value=1, max_value=10),
         pad     = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, b, w, h, c, filters, size, stride, pad):

    activ = Linear

    layer = Convolutional_layer(filters=filters,
                                size=size, stride=stride, pad=pad,
                                input_shape=(b, w, h, c),
                                activations=activ)

    with pytest.raises(AttributeError):
      print(layer)


  @given(b = st.integers(min_value=1,  max_value=10),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1,  max_value=10),
         filters = st.integers(min_value=1, max_value=10),
         size1   = st.integers(min_value=1, max_value=10),
         size2   = st.integers(min_value=1, max_value=10),
         stride1 = st.integers(min_value=1, max_value=10),
         stride2 = st.integers(min_value=1, max_value=10),
         pad     = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c,  filters, size1, size2, stride1, stride2, pad):

    if pad:
      keras_pad = 'same'
    else :
      keras_pad = 'valid'

    keras_activs    = ['relu', 'sigmoid', 'tanh','linear']
    numpynet_activs = [Relu, Logistic, Tanh, Linear]

    size   = (size1, size2)
    stride = (stride1, stride2)

    inpt    = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    weights = np.random.uniform(low=-1., high=1., size=(size1, size2) + (c, filters))
    bias    = np.random.uniform(low=-1., high=1., size=(filters,))

    for numpynet_activ, keras_activ in zip(numpynet_activs, keras_activs):

      layer = Convolutional_layer(filters=filters, input_shape=inpt.shape,
                                  weights=weights, bias=bias,
                                  activation=numpynet_activ,
                                  size=(size1,size2), stride=stride,
                                  pad=pad)

      # Keras model
      inp    = Input(batch_shape=inpt.shape)
      Conv2d = Conv2D(filters=filters,
                      kernel_size=size, strides=stride,
                      padding=keras_pad,
                      activation=keras_activ,
                      data_format='channels_last',
                      use_bias=True, bias_initializer='zeros',
                      dilation_rate=1)(inp)
      model = Model(inputs=[inp], outputs=[Conv2d])

      model.set_weights([weights, bias])

      # FORWARD

      forward_out_keras = model.predict(inpt)

      layer.forward(inpt, copy=False)
      forward_out_numpynet = layer.output

      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet,  atol=1e-4, rtol=1e-3)


  @given(b = st.integers(min_value=1,  max_value=10),
         w = st.integers(min_value=15, max_value=100),
         h = st.integers(min_value=15, max_value=100),
         c = st.integers(min_value=1,  max_value=10),
         filters = st.integers(min_value=1, max_value=10),
         size1   = st.integers(min_value=1, max_value=10),
         size2   = st.integers(min_value=1, max_value=10),
         stride1 = st.integers(min_value=1, max_value=10),
         stride2 = st.integers(min_value=1, max_value=10),
         pad     = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, filters, size1, size2, stride1, stride2, pad):

    if pad:
      keras_pad = 'same'
    else :
      keras_pad = 'valid'

    keras_activs    = ['relu', 'sigmoid', 'tanh','linear']
    numpynet_activs = [Relu, Logistic, Tanh, Linear]

    size   = (size1, size2)
    stride = (stride1, stride2)

    inpt    = np.random.uniform(low=-1., high=1., size=(b, w, h, c))
    weights = np.random.uniform(low=-1., high=1., size=(size1, size2) + (c, filters))
    bias    = np.random.uniform(low=-1., high=1., size=(filters,))

    for numpynet_activ, keras_activ in zip(numpynet_activs, keras_activs):

      layer = Convolutional_layer(filters=filters, input_shape=inpt.shape,
                                  weights=weights, bias=bias,
                                  activation=numpynet_activ,
                                  size=(size1,size2), stride=stride,
                                  pad=pad)

      # Keras model
      inp    = Input(batch_shape=inpt.shape)
      Conv2d = Conv2D(filters=filters,
                      kernel_size=size, strides=stride,
                      padding=keras_pad,
                      activation=keras_activ,
                      data_format='channels_last',
                      use_bias=True, bias_initializer='zeros',
                      dilation_rate=1)(inp)
      model = Model(inputs=[inp], outputs=[Conv2d])

      model.set_weights([weights, bias])

      # Try backward:
      with pytest.raises(NotFittedError):
        delta = np.empty(shape=inpt.shape)
        layer.backward(inpt, delta)

      # FORWARD

      forward_out_keras = model.predict(inpt)

      layer.forward(inpt, copy=False)
      forward_out_numpynet = layer.output

      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet,  atol=1e-4, rtol=1e-3)

      # BACKWARD

      grad1 = K.gradients(model.output, [model.input])
      grad2 = K.gradients(model.output, model.trainable_weights)

      func1 = K.function(model.inputs + model.outputs, grad1 )
      func2 = K.function(model.inputs + model.trainable_weights + model.outputs, grad2)

      delta_keras = func1([inpt])[0]
      updates     = func2([inpt])

      weights_updates_keras = updates[0]
      bias_updates_keras    = updates[1]

      delta_numpynet = np.zeros(shape=inpt.shape, dtype=float)
      layer.delta    = np.ones(shape=layer.out_shape, dtype=float)
      layer.backward(delta_numpynet, copy=False)

      assert np.allclose(delta_numpynet,          delta_keras,        atol=1e-3, rtol=1e-2)
      assert np.allclose(layer.weights_update, weights_updates_keras, atol=1e-3, rtol=1e-3)
      assert np.allclose(layer.bias_update,    bias_updates_keras,    atol=1e-5, rtol=1e-3)
