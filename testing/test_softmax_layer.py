# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.layers.softmax_layer import Softmax_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestSoftmaxLayer :
  '''
  Tests:
    - costructor of RNN_layer object
    - print function
    - forward function against tf.keras
    - backward function against tf.keras

  to be:
    '''

  @given(g = st.floats(-2,10),
         s = st.floats(-2,10),
         t = st.floats(0,10))
  @settings(max_examples=10,
            deadline=None)
  def test_constructor (self, g, s, t):

    with pytest.raises(ValueError):
      layer = Softmax_layer(groups=g, spatial=s, temperature=t)

    g = int(g)

    if g <= 0 :
      with pytest.raises(ValueError):
        layer = Softmax_layer(groups=g, spatial=s, temperature=t)
    else:

      if t > 0:
        layer = Softmax_layer(groups=g, spatial=s, temperature=t)

        assert layer.output == None
        assert layer.delta  == None

        assert layer.spatial == s
        assert layer.groups  == g

        assert layer.temperature == 1. / t

      else :
        with pytest.raises(ValueError):
          layer = Softmax_layer(groups=g, spatial=s, temperature=t)

  @given(b = st.integers(min_value=1, max_value=15 ),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=1, max_value=10 ))
  @settings(max_examples=50,
            deadline=None)
  def test_printer (self, b, w, h, c):

    layer = Softmax_layer(input_shape=(b, w, h, c))

    print(layer)

    layer.input_shape = (3.14, w, h, c)

    with pytest.raises(ValueError):
      print(layer)

  @example(b=5, w=1, h=1, c=100, spatial=True) # typical case
  @given(b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=10, max_value=100),
         spatial = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_forward (self, b, w, h, c, spatial):

    inpt  = np.random.uniform(low=0., high=1., size=(b, w, h, c))
    truth = np.random.choice([0., 1.], p=[.5, .5], size=(b, w, h, c))

    if spatial :
      inpt_tf  = tf.Variable(inpt.copy())
      truth_tf = tf.Variable(truth.copy())

    else :
      inpt_tf  = tf.Variable(inpt.copy().reshape(b, -1))
      truth_tf = tf.Variable(truth.copy().reshape(b, -1))

    # NumPyNet layer
    layer = Softmax_layer(input_shape=inpt.shape, groups=1, temperature=1., spatial=spatial)

    # Tensorflow layer
    model = tf.keras.layers.Softmax(axis=-1)
    loss  = tf.keras.losses.CategoricalCrossentropy(reduction=tf.keras.losses.Reduction.SUM)

    # Tensorflow softmax
    preds = model(inpt_tf)
    # Computing loss for tensorflow
    keras_loss = loss(truth_tf, preds).numpy()

    forward_out_keras = preds.numpy().reshape(b, w, h, c)

    # Softmax + crossentropy NumPyNet
    layer.forward(inpt, truth)
    forward_out_numpynet = layer.output
    numpynet_loss = layer.cost

    # testing softmax
    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, atol=1e-8)

    # testing crossentropy
    np.testing.assert_allclose(keras_loss, numpynet_loss, rtol=1e-5, atol=1e-6)


  @example(b=5, w=1, h=1, c=100, spatial=True) # typical case
  @given(b = st.integers(min_value=1, max_value=10),
         w = st.integers(min_value=1, max_value=100),
         h = st.integers(min_value=1, max_value=100),
         c = st.integers(min_value=10, max_value=100),
         spatial = st.booleans())
  @settings(max_examples=10,
            deadline=None)
  def test_backward (self, b, w, h, c, spatial):

    w, h = (1, 1) # backward working only in this case for spatial=False

    inpt  = np.random.uniform(low=0., high=1., size=(b, w, h, c))
    truth = np.random.choice([0., 1.], p=[.5, .5], size=(b, w, h, c))

    if spatial :
    	inpt_tf  = tf.Variable(inpt)
    	truth_tf = tf.Variable(truth)

    else :
    	inpt_tf = tf.Variable(inpt.copy().reshape(b,-1))
    	truth_tf = tf.Variable(truth.copy().reshape(b, -1))

    # NumPyNet layer
    layer = Softmax_layer(input_shape=inpt.shape, groups=1, temperature=1., spatial=spatial)

    # Tensorflow layer
    model = tf.keras.layers.Softmax(axis=-1)
    loss  = tf.keras.losses.CategoricalCrossentropy(from_logits=False, reduction=tf.keras.losses.Reduction.SUM)

    with tf.GradientTape() as tape :
    	preds = model(inpt_tf)
    	cost  = loss(truth_tf, preds)
    	grads = tape.gradient(cost, inpt_tf)

    	forward_out_keras = preds.numpy().reshape(b, w, h, c)
    	keras_loss        = cost.numpy()
    	delta_keras       = grads.numpy().reshape(b, w, h, c)

    layer.forward(inpt, truth)
    forward_out_numpynet = layer.output
    numpynet_loss = layer.cost

    delta = np.zeros(shape=inpt.shape)
    layer.backward(delta)

    np.testing.assert_allclose(forward_out_keras, forward_out_numpynet, atol=1e-8)
    np.testing.assert_allclose(keras_loss, numpynet_loss, rtol=1e-5, atol=1e-6)
    np.testing.assert_allclose(delta, delta_keras)
