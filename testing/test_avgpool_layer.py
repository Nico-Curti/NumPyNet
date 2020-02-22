# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.avgpool_layer import Avgpool_layer
from tensorflow.keras.layers import AvgPool2D

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
# __package__ = 'AvgPool Layer testing'


class TestAvgpoolLayer:
  '''
  Tests:
    - costructor of Avgpool_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras
  '''

  @given(size   = st.integers(min_value=-10, max_value=10),
         stride = st.integers(min_value=0, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=20,
            deadline=None)
  def test_constructor (self, size, stride, pad):

    if size <= 0 :

      with pytest.raises(LayerError):
        layer = Avgpool_layer(size=size, stride=stride, pad=pad)

    else:
      layer = Avgpool_layer(size=size, stride=stride, pad=pad)

      assert layer.size        == (size, size)
      assert len(layer.size)   == 2

      if stride:
        assert layer.stride == (stride, stride)
      else:
        assert layer.size == layer.stride

      assert len(layer.stride) == 2

      assert layer.delta   == None
      assert layer.output  == None

      assert layer.batch == 0
      assert layer.w == 0
      assert layer.h == 0
      assert layer.c == 0

      assert layer.pad        == pad
      assert layer.pad_left   == 0
      assert layer.pad_right  == 0
      assert layer.pad_top    == 0
      assert layer.pad_bottom == 0


  @given(size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=0, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, size, stride, pad):

    layer = Avgpool_layer(size=size, stride=stride, pad=pad)

    #TODO : to be discussed
    print(layer)

  @given(batch  = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10),
         size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=1, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, batch, w, h, c, size, stride, pad):

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

    # Numpy_net model
    numpynet = Avgpool_layer(size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'same'
    else :
      keras_pad = 'valid'

    # Keras model initialization.
    inp   = Input(batch_shape=inpt.shape)
    x     = AvgPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
    model = Model(inputs=[inp], outputs=x)

    # Keras Output
    forward_out_keras = model.predict(inpt)

    # numpynet forward and output
    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-8)


  @given(batch  = st.integers(min_value=1, max_value=15),
         w      = st.integers(min_value=15, max_value=100),
         h      = st.integers(min_value=15, max_value=100),
         c      = st.integers(min_value=1, max_value=10),
         size   = st.integers(min_value=1, max_value=10),
         stride = st.integers(min_value=1, max_value=10),
         pad    = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, batch, w, h, c, size, stride, pad):

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

    # Numpy_net model
    numpynet = Avgpool_layer(size=size, stride=stride, pad=pad)

    if pad:
      keras_pad = 'same'
    else :
      keras_pad = 'valid'

    # Keras model initialization.
    inp   = Input(batch_shape=inpt.shape)
    x     = AvgPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
    model = Model(inputs=[inp], outputs=x)

    # Keras Output
    forward_out_keras = model.predict(inpt)

    # try to backward
    with pytest.raises(NotFittedError):
      # Global delta init.
      delta = np.empty(shape=inpt.shape, dtype=float)

      # numpynet Backward
      numpynet.backward(delta)

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
