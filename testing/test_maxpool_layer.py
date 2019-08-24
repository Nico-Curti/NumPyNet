# !/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'MaxPool Layer testing'


from keras.models import Model
from keras.layers import Input, Activation
import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.maxpool_layer import Maxpool_layer
from keras.layers import MaxPool2D

import numpy as np

def test_maxpool_layer():
  '''
  Tests:
    if the numpy_net maxpool layer forward is consistent with Keras
    if the numpy_net maxpool layer backward is the same as Keras
    for differentsizes and strides
  to be:
  '''

  sizes   = [(1,1), (3,3), (30,30)]
  strides = [(1,1), (2,2), (20,20)]

  for size in sizes:
    for stride in strides:
      for pad in [False, True]:

        inpt = np.random.uniform(0.,1.,(5, 50, 51, 3))
        batch, w, h, c = inpt.shape

        # numpynet layer initialization, FALSE (numpynet) == VALID (Keras)
        numpynet = Maxpool_layer(size, stride, padding = pad)

        if pad:
          keras_pad = 'same'
        else :
          keras_pad = 'valid'

        # Keras model initialization.
        inp = Input(shape = (w, h, c), batch_shape=inpt.shape)
        x = MaxPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
        model = Model(inputs=[inp], outputs=x)

        # Keras Output
        forward_out_keras = model.predict(inpt)

        # numpynet forward and output
        numpynet.forward(inpt)
        forward_out_numpynet = numpynet.output

        # Test for dimension and allclose of all output
        assert forward_out_numpynet.shape == forward_out_keras.shape
        assert np.allclose(forward_out_numpynet, forward_out_keras,   atol  = 1e-8)

        # BACKWARD

        # Compute the gradient of output w.r.t input
        gradient = K.gradients(model.output, [model.input])

        # Define a function to evaluate the gradient
        func = K.function(model.inputs + [model.output], gradient)

        # Compute delta for Keras
        delta_keras = func([inpt])[0]

        # Definition of starting delta for numpynet
        delta = np.zeros(inpt.shape)
        numpynet.delta = np.ones(numpynet.out_shape())

        # numpynet Backward
        numpynet.backward(delta=delta)

        assert delta.shape == delta_keras.shape
        assert delta.shape == inpt.shape
        assert np.allclose(delta, delta_keras, atol = 1e-8)

        # ok all passed

if __name__ == '__main__':
    test_maxpool_layer()
