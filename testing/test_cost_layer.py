#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
from tensorflow.keras.layers import Activation
import tensorflow.keras.backend as K

from NumPyNet.utils import cost_type
from NumPyNet.layers.cost_layer import Cost_layer

from tensorflow.keras.losses import mean_squared_error
from tensorflow.keras.losses import mean_absolute_error
from tensorflow.keras.losses import logcosh
from tensorflow.keras.losses import hinge

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Cost Layer testing'


def test_cost_layer():
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

    if   keras_loss_type is mean_squared_error:  cost = cost_type.mse
    elif keras_loss_type is mean_absolute_error: cost = cost_type.mae
    elif keras_loss_type is logcosh:             cost = cost_type.logcosh
    elif keras_loss_type is hinge:               cost = cost_type.hinge
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
