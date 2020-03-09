# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Softmax
from tensorflow.keras.layers import Reshape
from tensorflow.keras.losses import categorical_crossentropy
import tensorflow.keras.backend as K

from NumPyNet.layers.softmax_layer import Softmax_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestSoftmaxLayer :
  '''
  Tests:
    - costructor of RNN_layer object
    - print function
    - forward function against tf.keras

  to be:
    - backward function not working
  '''

  @given(g = st.floats(-2,10),
         s = st.floats(-2,10),
         t = st.floats(0,10))
  @settings(max_examples=10,
            deadline=None)
  def test_constructor (self, g, s, t):

    with pytest.raises(ValueError):
      layer = Softmax_layer(groups=g, spatial=s, temperature=t)

    g = int(g)

    if g <= 0 :
      with pytest.raises(ValueError):
        layer = Softmax_layer(groups=g, spatial=s, temperature=t)
    else:

      if t > 0:
        layer = Softmax_layer(groups=g, spatial=s, temperature=t)

        assert layer.batch == None
        assert layer.w == None
        assert layer.h == None
        assert layer.c == None
        assert layer.output == None
        assert layer.delta  == None

        assert layer.spatial == s
        assert layer.groups  == g

        assert layer.temperature == 1. / t

      else :
        with pytest.raises(ValueError):
          layer = Softmax_layer(groups=g, spatial=s, temperature=t)

  def test_printer (self):

    layer = Softmax_layer()

    with pytest.raises(TypeError):
      print(layer)

    layer.batch, layer.w, layer.h, layer.c = (1,2,3,4)

    print(layer)


  @given(b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=10, max_value=100),
         h = st.integers(min_value=10, max_value=100),
         c = st.integers(min_value=1, max_value=10))
  @settings(max_examples=20,
            deadline=None)
  def test_forward (self, b, w, h, c):

    spatials = [False, True]

    for spatial in spatials:

      if spatial:
        axis = -1
      else :
        axis = (1, 2, 3)

      batch, w, h, c = (1, 3, 3, 3)

      np.random.seed(123)
      inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

      batch, w, h, c = inpt.shape

      truth = np.random.choice([0., 1.], p=[.5,.5], size=(batch,w,h,c))
      # truth = np.ones(shape=(batch, w, h, c))

      layer = Softmax_layer(groups=1, temperature=1., spatial=spatial)

      inp = Input(batch_shape=inpt.shape)
      if isinstance(axis, tuple):
        # insert a reshape operation to be compatible with tensorflow softmax on multi axis
        reshaped = Reshape((batch, w * h * c), input_shape=inpt.shape)(inp)
        axis = -1
        x = Softmax(axis=axis)(reshaped)
        x = Reshape((batch, w, h, c), input_shape=(batch, w * h * c))(x)
      else:
        x = Softmax(axis=axis)(inp)
      model = Model(inputs=[inp], outputs=x)
      model.compile(optimizer='sgd', loss='categorical_crossentropy')

      forward_out_keras = model.predict(inpt)

      # definition of tensorflow variable
      truth_tf             = K.variable(truth.ravel())
      forward_out_keras_tf = K.variable(forward_out_keras.ravel())

      loss = categorical_crossentropy( truth_tf, forward_out_keras_tf)

      keras_loss = K.eval(loss)
      layer.forward(inpt, truth)
      numpynet_loss = layer.cost

      assert np.isclose(keras_loss, numpynet_loss, atol=1e-7)

      forward_out_numpynet = layer.output

      assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-8)

  def _backward (self):

    #TODO : not working yet

    spatials = [False, True]

    for spatial in spatials:

      if spatial:
        axis = -1
      else :
        axis = (1, 2, 3)

      batch, w, h, c = (1, 3, 3, 3)

      np.random.seed(123)
      inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

      batch, w, h, c = inpt.shape

      truth = np.random.choice([0., 1.], p=[.5,.5], size=(batch,w,h,c))
      # truth = np.ones(shape=(batch, w, h, c))

      layer = Softmax_layer(groups=1, temperature=1., spatial=spatial)

      inp = Input(batch_shape=inpt.shape)
      if isinstance(axis, tuple):
        # insert a reshape operation to be compatible with tensorflow softmax on multi axis
        reshaped = Reshape((batch, w * h * c), input_shape=inpt.shape)(inp)
        axis = -1
        x = Softmax(axis=axis)(reshaped)
        x = Reshape((batch, w, h, c), input_shape=(batch, w * h * c))(x)
      else:
        x = Softmax(axis=axis)(inp)
      model = Model(inputs=[inp], outputs=x)
      model.compile(optimizer='sgd', loss='categorical_crossentropy')

      forward_out_keras = model.predict(inpt)

      # definition of tensorflow variable
      truth_tf             = K.variable(truth.ravel())
      forward_out_keras_tf = K.variable(forward_out_keras.ravel())

      loss = categorical_crossentropy( truth_tf, forward_out_keras_tf)

      keras_loss = K.eval(loss)
      layer.forward(inpt, truth)
      numpynet_loss = layer.cost

      assert np.isclose(keras_loss, numpynet_loss, atol=1e-7)

      forward_out_numpynet = layer.output

      assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-8)

      # def get_loss_grad(model, inputs, outputs):
      #   x, y, sample_weight = model._standardize_user_data(inputs, outputs)
      #   grad_ce = K.gradients(model.total_loss, model.output)
      #   func = K.function((model._feed_inputs + model._feed_targets + model._feed_sample_weights), grad_ce)
      #   return func(x + y + sample_weight)

       # ## compute gradient of loss with respect to inputs
       # grad_loss = K.gradients(loss, [model.input])
       #
       # # create a function to be able to run this computation graph
       # func = K.function(model.inputs + [truth_tf], grad_loss)
       # keras_delta = func([np.expand_dims(inpt, axis=0), truth])
       # keras_delta = get_loss_grad(model, inpt, truth)
       #
       # numpynet_delta = numpynet.delta
       #
       # assert np.allclose(keras_delta[0], numpynet_delta) # BROKEN
