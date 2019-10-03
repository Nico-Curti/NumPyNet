# !/usr/bin/env python3
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
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from keras.layers import Conv2D

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Convolutional Layer testing'

def test_convolutional_layer():
  '''
  Tests:
    if the convolutional layer forward is consistent with keras
    if the convolutional layer backward is consistent with keras

  to be:
    update function
  '''
  np.random.seed(123)

  keras_activations = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activations = [Relu, Logistic, Tanh, Linear]


  sizes   = [(3,3),(20,20),(30,30)]
  strides = [(2,2), (10,10),(20,20)]

  padding = [False, True]

  for keras_activ, numpynet_activ in zip(keras_activations,numpynet_activations):
    for size,stride in zip(sizes,strides):
      for pad in padding:

        batch = np.random.randint(low=1, high=10)
        c_out = np.random.randint(low=2, high=50)
        c_in  = np.random.randint(low=1, high=c_out)

        if pad:
          keras_pad = 'same'
        else :
          keras_pad = 'valid'

        inpt       = np.random.uniform(-1., 1., size = (batch, 100, 100, c_in))
        b, w, h, c = inpt.shape
        # Shape (size1,size2,c_in, c_out), reshape inside numpynet.forward.
        filters    = np.random.uniform(-1., 1., size = size + (c,c_out))
        bias       = np.random.uniform(-1., 1., size = (c_out,))


        # Numpy_net model
        global numpynet
        numpynet = Convolutional_layer(filters=c_out, input_shape=inpt.shape,
                                    weights=filters, bias=bias,
                                    activation=numpynet_activ,
                                    size=size, stride=stride,
                                    pad=pad)

        # Keras model
        inp  = Input(shape = inpt.shape[1:], batch_shape = inpt.shape)
        Conv2d = Conv2D(filters=c_out,
                          kernel_size=size, strides=stride,
                          padding=keras_pad,
                          activation=keras_activ,
                          data_format='channels_last',
                          use_bias=True , bias_initializer='zeros',
                          dilation_rate=1)(inp)     # dilation rate = 1 is no dilation (I think)
        model = Model(inputs=[inp], outputs=[Conv2d])

        model.set_weights([filters, bias])

        # FORWARD

        print(c_in, c_out, keras_activ, size, stride, pad, keras_pad, '\n', sep = '\n')

        global forward_out_keras, forward_out_numpynet

        forward_out_keras = model.predict(inpt)

        numpynet.forward(inpt, copy=False)
        forward_out_numpynet = numpynet.output

        assert forward_out_keras.shape == forward_out_numpynet.shape
        assert np.allclose(forward_out_keras, forward_out_numpynet,         atol=1e-04, rtol=1e-3)

        # BACKWARD
        global delta_numpynet, delta_keras, weights_updates_keras, bias_updates_keras

        grad1 = K.gradients(model.output, [model.input])
        grad2 = K.gradients(model.output, model.trainable_weights)

        func1 = K.function(model.inputs + model.outputs, grad1 )
        func2 = K.function(model.inputs + model.trainable_weights + model.outputs, grad2)

        delta_keras = func1([inpt])[0]
        updates     = func2([inpt])

        weights_updates_keras = updates[0]
        bias_updates_keras    = updates[1]

        delta_numpynet = np.zeros(shape=inpt.shape)
        numpynet.delta = np.ones(shape=numpynet.out_shape, dtype=float)
        numpynet.backward(delta_numpynet, copy=False)

        assert np.allclose(delta_numpynet,           delta_keras,           atol=1e-3, rtol=1e-3)
        assert np.allclose(numpynet.weights_updates, weights_updates_keras, atol=1e-3, rtol=1e-3) # for a lot of operations, atol is lower
        assert np.allclose(numpynet.bias_updates,    bias_updates_keras,    atol=1e-8, rtol=1e-3)

if __name__ == '__main__':
  test_convolutional_layer()
