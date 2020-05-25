#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from tensorflow.keras.models import Model, Sequential
from tensorflow.keras.layers import GRU, Input
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.gru_layer import GRU_layer
from NumPyNet.utils import data_to_timesteps

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

from random import choice

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

class TestGRUlayer :

  def test_constructor (self):
    pass

  def test_printer (self):
    pass

  def _forward (self):
    outputs = 30
    steps   = 1
    features = 10
    batch = 16

    data = np.random.uniform(size=(batch, features))

    weights = [np.random.uniform(size=(features, outputs)),
               np.random.uniform(size=(features, outputs)),
               np.random.uniform(size=(features, outputs)),
               np.random.uniform(size=(outputs, outputs)),
               np.random.uniform(size=(outputs, outputs)),
               np.random.uniform(size=(outputs, outputs))]

    bias    = [np.zeros(shape=(outputs,)), np.zeros(shape=outputs)]

    # assign same weights to all the kernel in keras as for NumPyNet
    keras_weights1 = np.concatenate([weights[i] for i in range(3)], axis=1)
    keras_weights2 = np.concatenate([weights[i] for i in range(3, 6)], axis=1)
    keras_bias     = np.concatenate([bias[0] for i in range(4)])

    inpt_keras, _ = data_to_timesteps(data, steps)

    assert inpt_keras.shape == (batch - steps, steps, features)

    inp   = Input(shape=(steps, features))
    gru   = GRU(units=outputs, use_bias=False)(inp)
    model = Model(inputs=inp, outputs=gru)

    model.set_weights([keras_weights1, keras_weights2])

    layer = GRU_layer(outputs=outputs, steps=steps, weights=weights, bias=[0,0,0])

    layer.forward(data)

    forward_out_keras = model.predict(inpt_keras)

    forward_out_numpynet = layer.output

    np.allclose(forward_out_keras, forward_out_numpynet)

    forward_out_keras
    forward_out_numpynet


  def test_backward (self):
    pass
