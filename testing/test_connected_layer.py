#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input
import keras.backend as K

from NumPyNet.activations import Relu
from NumPyNet.activations import Logistic
from NumPyNet.activations import Linear
from NumPyNet.activations import Tanh
from NumPyNet.layers.connected_layer import Connected_layer
from keras.layers import Dense

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Connected Layer testing'

def test_connected_layer():
  '''
  Tests:
    if the forward is coherent with keras
    if the updates (weight, bias) and delta computed by the backward are correct

  to be tested:
    update function, keras update not clear.
  '''
  np.random.seed(123)

  keras_activ = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activ = [Relu, Logistic, Tanh, Linear]

  #Usefull variables initialization
  outputs = 10
  batch, w, h, c = (5, 10, 10, 3)
  inpt = np.random.uniform(0.,1., (batch, w, h, c))

  weights = np.random.uniform(low=0., high=1., size=(w * h * c, outputs))
  bias    = np.random.uniform(low=0.,  high=1., size=(outputs))

  for activ in range(0,4):

    #Numpy_net model
    numpynet_layer = Connected_layer(inpt.shape, outputs,
                                  activation = numpynet_activ[activ],
                                  weights = weights, bias = bias)
    #Keras Model
    inp = Input(shape=(w * h * c), batch_shape=(batch, w * h * c))
    x = Dense(outputs,activation=keras_activ[activ], input_shape=(batch,inpt.size))(inp)
    model = Model(inputs=[inp], outputs=x)

    #Set weights in Keras Model.
    model.set_weights([weights, bias])

    #FORWARD

    #Keras forward output
    forward_out_keras = model.predict(inpt.reshape(batch, -1))

    #Numpy_net forward output
    numpynet_layer.forward(inpt)
    forward_out_numpynet = numpynet_layer.output

    #Forward output Test
    assert np.allclose(forward_out_numpynet[:,0,0,:], forward_out_keras, atol = 1e-8)

    #BACKWARD

    #Output derivative in respect to input
    grad      = K.gradients(model.output, [model.input])

    #Output derivative respect to trainable_weights(Weights and Biases)
    gradients = K.gradients(model.output, model.trainable_weights)

    #Definning functions to compute those gradients
    func  = K.function(model.inputs + [model.output], grad)
    func2 = K.function(model.inputs + model.trainable_weights + [model.output], gradients)

    #Evaluation of Delta, weights_updates and bias_updates for Keras
    delta_keras = func( [inpt.reshape(batch, -1)])
    updates     = func2([inpt.reshape(batch, -1)])

    #Initialization of numpy_net starting delta to ones
    numpynet_layer.delta = np.ones(shape=(batch, outputs))

    #Initialization of global delta
    delta = np.zeros(shape=(batch, w, h, c))

    #Computation of delta, weights_update and bias updates for numpy_net
    numpynet_layer.backward(inpt, delta=delta)

    #Now the global variable delta is updated

    assert np.allclose(delta_keras[0].reshape(batch, w, h, c), delta, atol = 1e-8)
    assert np.allclose(updates[0], numpynet_layer.weights_update, atol = 1e-8)
    assert np.allclose(updates[1], numpynet_layer.bias_update,    atol = 1e-8)

    #all passed


if __name__ == '__main__':
  test_connected_layer()
