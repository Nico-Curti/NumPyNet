#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from tensorflow.keras.layers import BatchNormalization

import numpy as np
from hypothesis import strategies as st
from hypothesis import given, settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'BatchNorm Layer testing'

@given(b = st.integers(min_value=3, max_value=15 ),
       w = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
       h = st.integers(min_value=50, max_value=300), # numerical instability for small dimensions!
       c = st.integers(min_value=1, max_value=10 ))
@settings(max_examples=10,
          deadline=None)
def test_batchnorm_layer(b, w, h, c):
  '''
  Tests:
    the forward and backward functions of the batchnorm layer against keras

  Problems:
    Precison of allclose change with the images, I tried sorting the input,
    but it doesn't get any better.

  to be:
    update functions
  '''

  np.random.seed(42)

  inpt = np.random.uniform(low=1., high=10., size=(b, w, h, c))

  bias   = np.random.uniform(low=0., high=1., size=(w, h, c)) # random biases
  scales = np.random.uniform(low=0., high=1., size=(w, h, c)) # random scales

  inpt_tf = tf.convert_to_tensor(inpt.astype('float32'))

  # Numpy_net model
  numpynet = BatchNorm_layer(scales=scales, bias=bias)

  # initializers must be callable with this syntax, I need those for dimensionality problems
  def bias_init(shape, **kwargs):
    return np.expand_dims(bias, axis=0)

  def gamma_init(shape, dtype=None):
    return np.expand_dims(scales, axis=0)

  def mean_init(shape, dtype=None):
    return np.expand_dims(inpt.mean(axis=0), axis=0)

  def var_init(shape, dtype=None):
    return np.expand_dims(inpt.var(axis=0), axis=0)

  # Keras Model
  inp = Input(batch_shape=inpt.shape)
  x = BatchNormalization(momentum=1., epsilon=1e-8, center=True, scale=True,
                         axis=[1, 2, 3],
                         beta_initializer=bias_init,
                         gamma_initializer=gamma_init,
                         moving_mean_initializer=mean_init,
                         moving_variance_initializer=var_init)(inp)
  model = Model(inputs=[inp], outputs=x)

  # Opens a TensorFlow Session to Initialize Variables
  sess = tf.InteractiveSession()
  # Initialization of variables, code won't work without it
  sess.run([tf.global_variables_initializer(),
            tf.local_variables_initializer()])

  # Keras forward
  forward_out_keras = model.predict(inpt)

  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

  # Comparing outputs
  assert forward_out_numpynet.shape == (b, w, h, c)
  assert forward_out_numpynet.shape == forward_out_keras.shape            #same shape
  assert np.allclose(forward_out_keras, forward_out_numpynet, atol=1e-3)  #same output

  x_norm = (numpynet.x - numpynet.mean)*numpynet.var

  # Own variable updates comparisons
  assert np.allclose(numpynet.x, inpt)
  assert numpynet.mean.shape == (w, h, c)
  assert numpynet.var.shape == (w, h, c)
  assert x_norm.shape == numpynet.x.shape
  assert np.allclose(numpynet.x_norm, x_norm)

  # BACKWARD

  # Computes analytical output gradients w.r.t input and w.r.t trainable_weights
  # Kept them apart for clarity
  grad1 = K.gradients(model.output, [model.input])
  grad2 = K.gradients(model.output, model.trainable_weights)

  # Definning functions to compute those gradients
  func1 = K.function(model.inputs + [model.output], grad1)
  func2 = K.function(model.inputs + model.trainable_weights + [model.output], grad2)

  # Evaluation of Delta, weights_updates and bias_updates for Keras
  delta_keras = func1([inpt])[0]
  updates     = func2([inpt])

  # Initialization of numpynet delta to one (multiplication) and an empty array to store values
  numpynet.delta = np.ones(shape=inpt.shape, dtype=float)
  delta_numpynet = np.empty(shape=inpt.shape, dtype=float)

  # numpynet bacward, updates delta_numpynet
  numpynet.backward(delta_numpynet)
  
  # Testing delta, the precision change with the image
  assert delta_keras.shape == delta_numpynet.shape
  print(inpt.shape, abs(delta_keras - delta_numpynet).max())
  assert np.allclose(delta_keras, delta_numpynet, atol=1e-1)

  # Testing scales updates
  assert updates[0][0].shape == numpynet.scales_updates.shape
  assert np.allclose(updates[0], numpynet.scales_updates, atol=1e-03)

  # Testing Bias updates
  assert updates[1][0].shape == numpynet.bias_updates.shape
  assert np.allclose(updates[1], numpynet.bias_updates, atol=1e-06)

  # All passed, but precision it's not consistent, missing update functions


if __name__ == '__main__':

  test_batchnorm_layer()
