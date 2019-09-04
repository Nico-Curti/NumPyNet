# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from keras.layers import UpSampling2D
import keras.backend as K

from NumPyNet.layers.upsample_layer import Upsample_layer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Upsample Layer testing'

def test_upsample_layer():
  '''
  Tests:
    if the forward and the backward of Numpy_net are consistent with keras.
    if all the possible strides works with different batch_size
  to be:
  '''
  np.random.seed(123)

  ws = (5, 10, 20, 101)
  hs = (7, 13, 23, 103)
  cs = (1, 3, 5)
  batches = (1, 6, 12)

  strides = (2, 3, 4)#, -2, -3, -4) # No DownSampling2D layer available
  scales = 1.

  for batch in batches:
    for w in ws:
      for h in hs:
        for c in cs:
          for stride in strides:

            inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

            # NumPyNet model
            numpynet_layer = Upsample_layer(stride=stride, scale=scales)

            # Keras Model
            inp = Input(shape=inpt.shape[1:], batch_shape=(batch, w, h, c))
            x = UpSampling2D(size=(stride, stride), data_format='channels_last', interpolation='nearest')(inp)
            model = Model(inputs=[inp], outputs=x)

            # FORWARD

            # Keras Forward
            forward_out_keras = model.predict(inpt)

            # numpynet forwrd
            numpynet_layer.forward(inpt)
            forward_out_numpynet = numpynet_layer.output

            # Forward check (Shape and Values)
            assert forward_out_keras.shape == forward_out_numpynet.shape
            assert np.allclose(forward_out_keras, forward_out_numpynet)

            # BACKWARD

            numpynet_layer.delta = layer.output
            delta = np.empty(shape=inpt.shape, dtype=float)
            numpynet_layer.backward(delta)

            assert np.allclose(delta, inpt)


if __name__ == '_main__':

  test_upsample_layer()
