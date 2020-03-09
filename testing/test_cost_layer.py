#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

from NumPyNet.layers.cost_layer import cost_type
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


  @given(scale=st.floats(min_value=0., max_value=10.),
         ratio=st.floats(min_value=0., max_value=10.),
         nbj_scale=st.floats(min_value=0., max_value=10. ),
         threshold=st.floats(min_value=0., max_value=10., ),
         smoothing=st.floats(min_value=0., max_value=10., ),
         b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10),
         )
  @settings(max_examples=10,
            deadline=None)
  def test_constructor (self, b, w, h, c, scale, ratio, nbj_scale, threshold, smoothing):

    for cost in range(0,9):

      input_shape = choice([None, (b, w, h, c)])

      layer = Cost_layer(cost_type=cost, input_shape=input_shape, scale=scale, ratio=ratio, noobject_scale=nbj_scale, threshold=threshold, smoothing=smoothing)

      assert layer.cost_type == cost
      assert layer.scale     == scale
      assert layer.ratio     == ratio
      assert layer.noobject_scale == nbj_scale
      assert layer.threshold == threshold
      assert layer.smoothing == smoothing

      assert layer._out_shape == input_shape
      assert layer.output == None
      assert layer.delta  == None


  def test_printer (self):

    layer = Cost_layer(cost_type=cost_type.mse)

    with pytest.raises(TypeError):
      print(layer)

    layer._out_shape = (1,2,3,4)

    print(layer)


  @given(scale=st.floats(min_value=0., max_value=10.),
         # ratio=st.floats(min_value=0., max_value=10.),
         nbj_scale=st.floats(min_value=0., max_value=10. ),
         threshold=st.floats(min_value=0., max_value=10., ),
         smoothing=st.floats(min_value=0., max_value=10., ),
         outputs=st.integers(min_value=10, max_value=100),
         )
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, outputs, scale, nbj_scale, threshold, smoothing):

    ratio = 0.
    losses = [mean_absolute_error, mean_squared_error, logcosh, hinge]

    truth = np.random.uniform(low=0., high=10., size=(outputs,)).astype(np.float32)
    inpt  = np.random.uniform(low=0., high=10., size=(outputs,)).astype(np.float32)

    for loss_function in losses :

      sess = tf.Session()

      keras_loss_type = loss_function

      truth_tf = tf.convert_to_tensor(truth)
      inpt_tf  = tf.convert_to_tensor(inpt)

      if   keras_loss_type is mean_squared_error:  cost = cost_type.mse
      elif keras_loss_type is mean_absolute_error: cost = cost_type.mae
      elif keras_loss_type is logcosh:             cost = cost_type.logcosh
      elif keras_loss_type is hinge:               cost = cost_type.hinge
      else:
        raise ValuError()

      layer = Cost_layer(input_shape=inpt.shape, cost_type=cost,
                         scale=scale, ratio=ratio, noobject_scale=nbj_scale,
                         threshold=threshold, smoothing=smoothing)

      keras_loss_tf = keras_loss_type(truth_tf, inpt_tf)

      tf.compat.v1.global_variables_initializer()

      keras_loss = keras_loss_tf.eval(session=sess)

      layer.forward(inpt, truth)

      assert layer.out_shape == inpt.shape
      assert layer.output is not None
      assert layer.delta is not None
      assert layer.cost is not None

      # recreate cost layer with default values foor testing against keras
      layer = Cost_layer(input_shape=inpt.shape, cost_type=cost,
                                  scale=1., ratio=0., noobject_scale=1.,
                                  threshold=0., smoothing=0.)
      layer.forward(inpt, truth)
      loss = layer.cost

      assert np.isclose(keras_loss, loss, atol=1e-3)


  @given(outputs=st.integers(min_value=10, max_value=100))
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, outputs):
    # testing only default values since the backard is really simple

    losses = [mean_absolute_error, mean_squared_error, logcosh]
              #, hinge] # derivative is ambigous

    for loss_function in losses :

      sess = tf.Session()

      losses = [mean_absolute_error, mean_squared_error, logcosh]
                #, hinge] # derivative is ambigous

      keras_loss_type = loss_function

      truth = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32)
      inpt  = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32)

      truth_tf = tf.convert_to_tensor(truth)
      inpt_tf  = tf.convert_to_tensor(inpt)

      if   keras_loss_type is mean_squared_error:  cost = cost_type.mse
      elif keras_loss_type is mean_absolute_error: cost = cost_type.mae
      elif keras_loss_type is logcosh:             cost = cost_type.logcosh
      elif keras_loss_type is hinge:               cost = cost_type.hinge
      else:
        raise ValuError()

      layer = Cost_layer(input_shape=inpt.shape, cost_type=cost,
                                  scale=1., ratio=0., noobject_scale=1.,
                                  threshold=0., smoothing=0.)

      keras_loss_tf = keras_loss_type(truth_tf, inpt_tf)

      tf.compat.v1.global_variables_initializer()

      keras_loss = keras_loss_tf.eval(session=sess)

      layer.forward(inpt, truth)
      loss = layer.cost

      assert np.isclose(keras_loss, loss, atol=1e-7)

      # BACKWARD

      # compute loss based on model's output and true labels
      if   keras_loss_type is mean_squared_error:
        loss_tf = K.mean( K.square(truth_tf - inpt_tf) )
      elif keras_loss_type is mean_absolute_error:
        loss_tf = K.mean( K.abs(truth_tf - inpt_tf) )
      elif keras_loss_type is logcosh:
        loss_tf = K.mean( K.log(tf.math.cosh(truth_tf - inpt_tf)))
      elif keras_loss_type is hinge:
        loss_tf = K.maximum(1. - truth_tf * inpt_tf, 0)
      else:
        raise ValueError()

      # compute gradient of loss with respect to inputs
      grad_loss = K.gradients(loss_tf, [inpt_tf])

      # create a function to be able to run this computation graph
      func = K.function([inpt_tf] + [truth_tf], grad_loss)
      keras_delta = func([np.expand_dims(inpt, axis=0), truth])

      numpynet_delta = layer.delta

      assert np.allclose(keras_delta, numpynet_delta)

def cost_layer():
  '''
  Tests:
        the fwd function of the cost layer.
        if the cost is the same for every cost_type (mse and mae)
        if the delta is correctly computed

  To be tested:
        _smoothing
        _threshold
        _ratio
        noobject_scale
        masked
        _seg
        _wgan
  '''
  np.random.seed(123)

  losses = [mean_absolute_error, mean_squared_error, logcosh]
            #, hinge] # derivative is ambigous

  for loss_function in losses :

    sess = tf.Session()

    losses = [mean_absolute_error, mean_squared_error, logcosh]
              #, hinge] # derivative is ambigous

    keras_loss_type = loss_function

    outputs = 100
    truth = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32)
    inpt  = np.random.uniform(low=0., high=1., size=(outputs,)).astype(np.float32)

    truth_tf = tf.convert_to_tensor(truth)
    inpt_tf  = tf.convert_to_tensor(inpt)

    if   keras_loss_type is mean_squared_error:  cost = cl.cost_type.mse
    elif keras_loss_type is mean_absolute_error: cost = cl.cost_type.mae
    elif keras_loss_type is logcosh:             cost = cl.cost_type.logcosh
    elif keras_loss_type is hinge:               cost = cl.cost_type.hinge
    else:
      raise ValuError()

    numpynet_layer = Cost_layer(input_shape=inpt.shape, cost_type=cost,
                                scale=1., ratio=0., noobject_scale=1.,
                                threshold=0., smoothing=0.)

    keras_loss_tf = keras_loss_type(truth_tf, inpt_tf)

    tf.compat.v1.global_variables_initializer()

    keras_loss = keras_loss_tf.eval(session=sess)

    numpynet_layer.forward(inpt, truth)
    numpynet_loss = numpynet_layer.cost

    assert np.isclose(keras_loss, numpynet_loss, atol=1e-7)

    # BACKWARD

    # compute loss based on model's output and true labels
    if   keras_loss_type is mean_squared_error:
      loss = K.mean( K.square(truth_tf - inpt_tf) )
    elif keras_loss_type is mean_absolute_error:
      loss = K.mean( K.abs(truth_tf - inpt_tf) )
    elif keras_loss_type is logcosh:
      loss = K.mean( K.log(tf.math.cosh(truth_tf - inpt_tf)))
    elif keras_loss_type is hinge:
      loss = K.maximum(1. - truth_tf * inpt_tf, 0)
    else:
      raise ValueError()

    # compute gradient of loss with respect to inputs
    grad_loss = K.gradients(loss, [inpt_tf])

    # create a function to be able to run this computation graph
    func = K.function([inpt_tf] + [truth_tf], grad_loss)
    keras_delta = func([np.expand_dims(inpt, axis=0), truth])

    numpynet_delta = numpynet_layer.delta

    assert np.allclose(keras_delta, numpynet_delta)

if __name__ == '__main__':

  test_cost_layer()
