#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
import tensorflow.keras.backend as K

from NumPyNet.utils import cost_type
from NumPyNet.layers.cost_layer import Cost_layer

from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import logcosh
from tensorflow.keras.losses import hinge

from random import choice
import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


nn_losses = [cost_type.mae, cost_type.mse, cost_type.logcosh, cost_type.hinge]
tf_losses = [mean_absolute_error, mean_squared_error, logcosh, hinge]


class TestCostLayer :
  '''
  Tests:
    - the costructor of the Cost_layer object.
    - the print function of the Cost_layer
    - If the cost is computed correctly
    - if the delta is correctly computed

  To be:
    _smoothing
    _threshold
    _ratio             problems.
    noobject_scale
    masked
    _seg
    _wgan
  '''

  @given(scale = st.floats(min_value=0., max_value=10.),
         ratio = st.floats(min_value=0., max_value=10.),
         nbj_scale = st.floats(min_value=0., max_value=10.),
         threshold = st.floats(min_value=0., max_value=10.),
         smoothing = st.floats(min_value=0., max_value=10.),
         b = st.integers(min_value=1, max_value=15),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10),
         cost = st.integers(min_value=0, max_value=8)
         )
  @settings(max_examples=10,
            deadline=None)
  def test_constructor (self, b, w, h, c, scale, ratio, nbj_scale, threshold, smoothing, cost):

    input_shape = choice([None, (b, w, h, c)])

    layer = Cost_layer(cost_type=cost, input_shape=input_shape, scale=scale, ratio=ratio, noobject_scale=nbj_scale, threshold=threshold, smoothing=smoothing)

    assert layer.cost_type == cost
    assert layer.scale     == scale
    assert layer.ratio     == ratio
    assert layer.noobject_scale == nbj_scale
    assert layer.threshold == threshold
    assert layer.smoothing == smoothing

    assert layer.out_shape == input_shape
    assert layer.output == None
    assert layer.delta  == None


  @given(cost = st.integers(min_value=0, max_value=len(nn_losses)-1))
  @settings(max_examples=10,
            deadline=None)
  def test_printer (self, cost):

    layer = Cost_layer(cost_type=cost)

    with pytest.raises(TypeError):
      print(layer)

    layer.input_shape = (1, 2, 3, 4)

    print(layer)


  @given(scale = st.floats(min_value=0., max_value=10.),
         # ratio = st.floats(min_value=0., max_value=10.),
         nbj_scale = st.floats(min_value=0., max_value=10.),
         threshold = st.floats(min_value=0., max_value=10.),
         smoothing = st.floats(min_value=0., max_value=10.),
         outputs = st.integers(min_value=10, max_value=100),
         cost_idx = st.integers(min_value=0, max_value=len(nn_losses)-1)
         )
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, outputs, scale, nbj_scale, threshold, smoothing, cost_idx):

    ratio = 0.
    nn_cost = nn_losses[cost_idx]
    tf_cost = tf_losses[cost_idx]

    truth = np.random.uniform(low=0., high=10., size=(outputs,)).astype(np.float32) # I don't know why but TF in this case requires a float32
    inpt  = np.random.uniform(low=0., high=10., size=(outputs,)).astype(np.float32) # I don't know why but TF in this case requires a float32

    truth_tf = tf.Variable(truth)
    inpt_tf  = tf.Variable(inpt)

    layer = Cost_layer(input_shape=inpt.shape, cost_type=nn_cost,
                       scale=scale, ratio=ratio, noobject_scale=nbj_scale,
                       threshold=threshold, smoothing=smoothing)

    keras_loss_tf = tf_cost(truth_tf, inpt_tf)

    keras_loss = keras_loss_tf.numpy()

    layer.forward(inpt=inpt, truth=truth)

    assert layer.out_shape == inpt.shape
    assert layer.output is not None
    assert layer.delta is not None
    assert layer.cost is not None

    # recreate cost layer with default values foor testing against keras
    layer = Cost_layer(input_shape=inpt.shape, cost_type=nn_cost,
                       scale=1., ratio=0., noobject_scale=1.,
                       threshold=0., smoothing=0.)
    layer.forward(inpt=inpt, truth=truth)
    loss = layer.cost

    assert np.isclose(keras_loss, loss, atol=1e-3)


  @given(outputs = st.integers(min_value=10, max_value=100),
         cost_idx = st.integers(min_value=0, max_value=len(nn_losses)-2)) #hinge derivative is ambigous
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, outputs, cost_idx):
    # testing only default values since the backard is really simple

    nn_cost = nn_losses[cost_idx]
    tf_cost = tf_losses[cost_idx]


    truth = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32) # I don't know why but TF in this case requires a float32
    inpt  = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32) # I don't know why but TF in this case requires a float32

    truth_tf = tf.Variable(truth)
    inpt_tf  = tf.Variable(inpt)

    layer = Cost_layer(input_shape=inpt.shape, cost_type=nn_cost,
                       scale=1., ratio=0., noobject_scale=1.,
                       threshold=0., smoothing=0.)

    with tf.GradientTape() as tape :
      preds = tf_cost(truth_tf, inpt_tf)
      grads = tape.gradient(preds, inpt_tf)

      keras_loss  = preds.numpy()
      delta_keras = grads.numpy()


    layer.forward(inpt=inpt, truth=truth)
    loss = layer.cost

    assert np.isclose(keras_loss, loss, atol=1e-7)

    # BACKWARD

    numpynet_delta = layer.delta

    np.testing.assert_allclose(delta_keras, numpynet_delta, rtol=1e-4, atol=1e-8)
