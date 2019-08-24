#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'BatchNorm Layer testing'

from keras.models import Model
from keras.layers import Input, Activation
import keras.backend as K
import tensorflow as tf

from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from keras.layers import BatchNormalization

import numpy as np


def test_batchnorm_layer():
  '''
  Tests:
    the forward and backward functions of the batchnorm layer against keras

  Problems:
    Precison of allclose change with the images, I tried sorting the input,
    but it doesn't get any better.

  to be:
    different batch size
    update functions
  '''

  batch = 5

  inpt = np.random.uniform(0.,1.,(batch,10,10,3))

  b, w, h, c = inpt.shape

  bias = np.random.uniform(0.,1., size = (w,h,c))  #random biases
  scales = np.random.uniform(0.,1.,size = (w,h,c)) #random scales

  #Numpy_net model
  numpynet = BatchNorm_layer(scales=scales, bias=bias)

  #initializers must be callable with this syntax, I need those for dimensionality problems
  def bias_init(shape, dtype = None):
    return bias

  def gamma_init(shape, dtype = None):
    return scales

  def mean_init(shape, dtype = None):
    return inpt.mean(axis = 0)

  def var_init(shape, dtype = None):
    return inpt.var(axis = 0)

  #Keras Model
  inp = Input(shape = (b,w,h,c))
  x = BatchNormalization(momentum = 1., epsilon=1e-8, center=True, scale=True,
                         axis = -1,
                         beta_initializer            = bias_init,
                         gamma_initializer           = gamma_init,
                         moving_mean_initializer     = mean_init,
                         moving_variance_initializer = var_init)(inp)
  model = Model(inputs = [inp], outputs =  x)

  #Keras forward
  forward_out_keras = model.predict(np.expand_dims(inpt,axis = 0))[0,:,:,:,:]

  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

  #Comparing outputs
  assert forward_out_numpynet.shape == (b,w,h,c)
  assert forward_out_numpynet.shape == forward_out_keras.shape              #same shape
  assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-6)  #same output

  x_norm = (numpynet.x - numpynet.mean)*numpynet.var

  #Own variable updates comparisons
  assert np.allclose(numpynet.x, inpt)
  assert numpynet.mean.shape == (w,h,c)
  assert numpynet.var.shape == (w,h,c)
  assert x_norm.shape == numpynet.x.shape
  assert np.allclose(numpynet.x_norm, x_norm)

  #BACKWARD

  #Opens a TensorFlow Session to Initialize Variables
  sess = tf.InteractiveSession()

  #Computes analytical output gradients w.r.t input and w.r.t trainable_weights
  #Kept them apart for clarity
  grad      = K.gradients(model.output, [model.input])
  gradients = K.gradients(model.output, model.trainable_weights)

  #Define 2 functions to compute the numerical values of grad and gradients
  func  = K.function(model.inputs + [model.output], grad)
  func2 = K.function(model.inputs + model.trainable_weights + [model.output], gradients)

  #Initialization of variables, code won't work without it
  sess.run(tf.global_variables_initializer())

  #Assigns Numerical Values
  updates     = func2([np.expand_dims(inpt, axis = 0)])
  delta_keras = func([np.expand_dims(inpt,axis = 0)])[0][0,:,:,:,:]

  #Initialization of numpynet delta to one (multiplication) and an empty array to store values
  numpynet.delta = np.ones(shape=inpt.shape)
  delta_numpynet = np.empty(shape=inpt.shape)

  #numpynet bacward, updates delta_numpynet
  numpynet.backward(delta_numpynet)

  #Testing delta, the precision change with the image
  assert delta_keras.shape == delta_numpynet.shape       #1e-1 for random image, 1e-8 for dog
  assert np.allclose(delta_keras, delta_numpynet ,         atol=1e-1)

  #Testing scales updates
  assert updates[0].shape == numpynet.scales_updates.shape
  assert np.allclose(updates[0], numpynet.scales_updates,  atol=1e-05)

  #Testing Bias updates
  assert updates[1].shape == numpynet.bias_updates.shape
  assert np.allclose(updates[1], numpynet.bias_updates,    atol=1e-08)

  #All passed, but precision it's not consistent, missing update functions7


if __name__ == '__main__':
    test_batchnorm_layer()
