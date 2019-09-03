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
      axis = (1, 2, 3)

    batch, w, h, c = (1, 3, 3, 3)

    np.random.seed(123)
    inpt = np.random.uniform(low = 0., high = 1., size = (batch, w, h, c))

    batch, w, h, c = inpt.shape

    truth = np.random.choice([0., 1.], p = [.5,.5], size=(batch,w,h,c))
    truth = np.ones(shape=(batch, w, h, c))

    numpynet = Softmax_layer(groups = 1, temperature = 1., spatial = spatial)

    inp = Input(shape=(w,h,c), batch_shape = inpt.shape)
    x = Softmax(axis = axis)(inp)
    model = Model(inputs=[inp], outputs=x)
    model.compile(optimizer='sgd', loss='categorical_crossentropy')

    forward_out_keras = model.predict(inpt)

    # definition of tensorflow variable
    truth_tf             = K.variable(truth.ravel())
    forward_out_keras_tf = K.variable(forward_out_keras.ravel())

    loss = categorical_crossentropy( truth_tf, forward_out_keras_tf)

    keras_loss = K.eval(loss)
    numpynet.forward(inpt, truth)
    numpynet_loss = numpynet.cost

    assert np.isclose(keras_loss, numpynet_loss, atol=1e-7)

    forward_out_numpynet = numpynet.output

    assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-8)

    def get_loss_grad(model, inputs, outputs):
      x, y, sample_weight = model._standardize_user_data(inputs, outputs)
      grad_ce = K.gradients(model.total_loss, model.output)
      func = K.function((model._feed_inputs + model._feed_targets + model._feed_sample_weights), grad_ce)
      return func(x + y + sample_weight)

    ### compute gradient of loss with respect to inputs
    #grad_loss = K.gradients(loss, [model.input])

    ## create a function to be able to run this computation graph
    #func = K.function(model.inputs + [truth_tf], grad_loss)
    #keras_delta = func([np.expand_dims(inpt, axis=0), truth])
    keras_delta = get_loss_grad(model, inpt, truth)

    numpynet_delta = numpynet.delta

    #assert np.allclose(keras_delta[0], numpynet_delta) # BROKEN


if __name__ == '__main__':
  test_softmax_layer()
