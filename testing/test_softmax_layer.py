# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
from keras.layers import Softmax
from keras.losses import categorical_crossentropy
import keras.backend as K

from NumPyNet.layers.softmax_layer import Softmax_layer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'SoftMax Layer testing'

def test_softmax_layer():

  np.random.seed(123)

  spatials = [False, True]

  for spatial in spatials:

    if spatial:
      axis = -1
    else :
      axis = (1,2,3)

    np.random.seed(123)
    inpt = np.random.uniform(low = 0., high = 1., size = (2,10,10,3))

    batch, w, h, c = inpt.shape

    truth = np.random.choice([0., 1.], p = [.5,.5], size=(batch,w,h,c))

    numpynet = Softmax_layer(groups = 1, temperature = 1., spatial = spatial)

    inp = Input(shape=(w,h,c), batch_shape = inpt.shape)
    x = Softmax(axis = axis)(inp)
    model = Model(inputs=[inp], outputs=x)

    forward_out_keras = model.predict(inpt)

    # definition of tensorflow variable
    truth_tf             = K.variable(truth.ravel())
    forward_out_keras_tf = K.variable(forward_out_keras.ravel())

    loss = categorical_crossentropy( truth_tf, forward_out_keras_tf)

    keras_loss = K.eval(loss)
    numpynet.forward(inpt, truth)
    numpynet_loss = numpynet.cost

    assert np.allclose(numpynet_loss, keras_loss)

    forward_out_numpynet = numpynet.output

    assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-8)

    # Forward passed, the backward is different though

if __name__ == '__main__':
  test_softmax_layer()
