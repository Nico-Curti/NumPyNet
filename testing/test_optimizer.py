# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.optimizer import Adam
from NumPyNet.optimizer import Adamax
from NumPyNet.optimizer import SGD
from NumPyNet.optimizer import Momentum
from NumPyNet.optimizer import NesterovMomentum
from NumPyNet.optimizer import Adadelta
from NumPyNet.optimizer import Adagrad
from NumPyNet.optimizer import RMSprop

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example


class TestOptimizer:
  '''
  Test for correct updates of most used optimizer
  With no clipping on the gradients for SGD, Momentum and nesterov, the tests are all correct.
  '''

  Optimizers = [SGD, Momentum, NesterovMomentum,
                Adadelta, Adagrad, Adam, Adamax, RMSprop]

  def test_constructor(self):
    pass

  # @example(w=62, h=25, c=3, lr = 0.11279237270355225, momentum=0.0, beta1=0., beta2=0., rho=0.1874999701976776, decay=0.)
  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         momentum = st.floats(min_value=0, max_value=1, width=32),
         beta1 = st.floats(min_value=0, max_value=1, width=32),
         beta2 = st.floats(min_value=0, max_value=1, width=32),
         rho   = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update(self, w, h, c, lr, momentum, beta1, beta2, rho, decay):

    epsilon = 1e-8

    sgd   = SGD(lr=lr, decay=decay)
    mom   = Momentum(lr=lr, momentum=momentum, decay=decay)
    nmom  = NesterovMomentum(lr=lr, momentum=momentum, decay=decay)
    adam  = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
    adamm = Adamax(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
    adag  = Adagrad(lr=lr, epsilon=epsilon, decay=decay)
    adad  = Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
    rms   = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)

    tf_sgd   = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0., nesterov=False, decay=decay)
    tf_mom   = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=False, decay=decay)
    tf_nmom  = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True, decay=decay)
    tf_adam  = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=decay)
    tf_adamm = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=decay)
    tf_adag  = tf.keras.optimizers.Adagrad(learning_rate=lr, epsilon=epsilon, initial_accumulator_value=0., decay=decay)
    tf_adad  = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=epsilon, decay=decay)
    tf_rms   = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, epsilon=epsilon, momentum=momentum, decay=decay)

    tf_optimizers = [tf_sgd, tf_mom, tf_nmom, tf_adam, tf_adamm, tf_adag, tf_adad, tf_rms]
    optimizers    = [sgd, mom, nmom, adam, adamm, adag, adad, rms]

    for i, (nn_opt, tf_opt) in enumerate(zip(optimizers, tf_optimizers)):

      first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype('float32')
      second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype('float32')
      third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype('float32')

      tf_first  = tf.Variable(first.copy())
      tf_second = tf.Variable(second.copy())
      tf_third  = tf.Variable(third.copy())

      first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype('float32')
      second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype('float32')
      third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype('float32')

      tf_first_update  = tf.Variable(first_update.copy())
      tf_second_update = tf.Variable(second_update.copy())
      tf_third_update  = tf.Variable(third_update.copy())

      params = [first, second, third]
      grads  = [first_update, second_update, third_update]

      tf_params = [tf_first, tf_second, tf_third]
      tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

      # Updates variables
      first_up, second_up, third_up = nn_opt.update(params, grads)
      tf_opt.apply_gradients(zip(tf_grads, tf_params))

      tf_updated1 = tf_params[0].numpy()
      tf_updated2 = tf_params[1].numpy()
      tf_updated3 = tf_params[2].numpy()

      np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
      np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)
