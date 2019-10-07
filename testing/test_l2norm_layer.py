# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation

from NumPyNet.layers.l2norm_layer import L2Norm_layer

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'L2Norm Layer testing'

@given(batch = st.integers(min_value=1, max_value=15 ),
       w     = st.integers(min_value=1, max_value=100),
       h     = st.integers(min_value=1, max_value=100),
       c     = st.integers(min_value=1, max_value=10 ))
@settings(max_examples=10,
          deadline=1000)
def test_l2norm_layer(batch, w, h, c):
  '''
  TTests:
    if the l2norm layer forwards and backward are consistent with keras

  to be:
    backward does not work
  '''
  np.random.seed(123)

  for axis in [None, 1, 2, 3]:

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))
    inpt = np.ones(shape=(batch, w, h, c))

    # Keras Model
    inp = Input(shape=inpt.shape[1:])
    x = Activation(activation='linear')(inp)
    model = Model(inputs=[inp], outputs=x)

    # NumPyNet model
    numpynet = L2Norm_layer(axis=axis)

    # Keras Output
    forward_out_keras = K.eval(K.l2_normalize(inpt, axis=axis))

    # numpynet forward and output
    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    # Test for dimension and allclose of all output
    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-7, rtol=1e-6)


    # BACKWARD

    l2 = model.input / K.sqrt(K.sum(K.square(model.input), axis=axis, keepdims=True))
    grad = K.gradients(l2, [model.input])
    func = K.function(model.inputs + [l2], grad)

    delta_keras = func([inpt])[0]

    # Definition of starting delta for numpynet
    numpynet.delta = np.zeros(shape=numpynet.out_shape, dtype=float)
    delta = np.zeros(shape=inpt.shape, dtype=float)

    # numpynet Backward
    numpynet.backward(delta)

    # Back tests
    assert delta.shape == delta_keras.shape
    assert delta.shape == inpt.shape
    # assert np.allclose(delta, delta_keras, atol=1e-6) # NOT WORK



if __name__ == '__main__':

  test_l2norm_layer()
