# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import keras.backend as K
from keras.models import Model
from keras.layers import Input
from keras.layers import Activation

from NumPyNet.layers.l1norm_layer import L1Norm_layer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'L1Norm Layer testing'


def test_l2norm_layer():
  '''
  TTests:
    if the l1norm layer forwards and backward are consistent with keras

  to be:
    backward does not work
  '''
  np.random.seed(123)

  for axis in [None, 1, 2, 3]:

    batch = 1
    w, h, c = (11, 13, 7)

    inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

    # Keras Model
    inp = Input(shape=inpt.shape[1:])
    x = Activation(activation='linear')(inp)
    model = Model(inputs=[inp], outputs=x)

    # NumPyNet model
    numpynet = L1Norm_layer(axis=axis)

    # # Keras Output
    # forward_out_keras = K.eval(K.l1_normalize(inpt, axis=axis)) # IT DOES NOT EXIST

    # numpynet forward and output
    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    # # Test for dimension and allclose of all output
    # assert forward_out_numpynet.shape == forward_out_keras.shape
    # assert np.allclose(forward_out_numpynet, forward_out_keras, atol=1e-7, rtol=1e-6)


    # # BACKWARD

    # out = model.predict(inpt)
    # l2 = model.output / K.sum(K.abs(model.output), axis=axis, keepdims=True)
    # grad = K.gradients(l2, model.inputs)
    # func = K.function(model.inputs + [model.output], grad)

    # delta_keras = func([inpt])[0]

    # Definition of starting delta for numpynet
    numpynet.delta = np.zeros(shape=numpynet.out_shape)
    delta = np.zeros(inpt.shape)

    # numpynet Backward
    numpynet.backward(delta)

    # # Back tests
    # assert delta.shape == delta_keras.shape
    # assert delta.shape == inpt.shape
    # #assert np.allclose(delta, delta_keras, atol=1e-6)



if __name__ == '__main__':

  test_l1norm_layer()
