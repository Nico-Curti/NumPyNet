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

  def test_constructor(self):
    pass

  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_sgd(self, w, h, c, lr, decay):

    sgd = SGD(lr=lr, decay=decay)
    tf_sgd = tf.keras.optimizers.SGD(learning_rate=lr, momentum=0., nesterov=False, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = sgd.update(params, grads)
    tf_sgd.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         momentum = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_momentum(self, w, h, c, lr, momentum, decay):

    mom = Momentum(lr=lr, momentum=momentum, decay=decay)
    tf_mom = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=False, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = mom.update(params, grads)
    tf_mom.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         momentum = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_nesterov(self, w, h, c, lr, momentum, decay):

    nmom = NesterovMomentum(lr=lr, momentum=momentum, decay=decay)
    tf_nmom = tf.keras.optimizers.SGD(learning_rate=lr, momentum=momentum, nesterov=True, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = nmom.update(params, grads)
    tf_nmom.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         beta1 = st.floats(min_value=0, max_value=1, width=32),
         beta2 = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_adam(self, w, h, c, lr, beta1, beta2, decay):

    epsilon = 1e-8

    adam = Adam(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
    tf_adam = tf.keras.optimizers.Adam(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = adam.update(params, grads)
    tf_adam.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         beta1 = st.floats(min_value=0, max_value=1, width=32),
         beta2 = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_adamax(self, w, h, c, lr, beta1, beta2, decay):

    epsilon = 1e-8
    adamm = Adamax(lr=lr, beta1=beta1, beta2=beta2, epsilon=epsilon, decay=decay)
    tf_adamm = tf.keras.optimizers.Adamax(learning_rate=lr, beta_1=beta1, beta_2=beta2, epsilon=epsilon, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = adamm.update(params, grads)
    tf_adamm.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_adagrad(self, w, h, c, lr, decay):

    epsilon = 1e-8
    adag = Adagrad(lr=lr, epsilon=epsilon, decay=decay)
    tf_adag = tf.keras.optimizers.Adagrad(learning_rate=lr, epsilon=epsilon, initial_accumulator_value=0., decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = adag.update(params, grads)
    tf_adag.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         rho   = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_adadelta(self, w, h, c, lr, rho, decay):

    epsilon = 1e-8
    adad = Adadelta(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
    tf_adad = tf.keras.optimizers.Adadelta(learning_rate=lr, rho=rho, epsilon=epsilon, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = adad.update(params, grads)
    tf_adad.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)


  @given(w  = st.integers(min_value=15, max_value=100),
         h  = st.integers(min_value=15, max_value=100),
         c  = st.integers(min_value=1, max_value=10),
         lr = st.floats(min_value=0, max_value=1, width=32),
         momentum = st.floats(min_value=0, max_value=1, width=32),
         rho = st.floats(min_value=0, max_value=1, width=32),
         decay = st.floats(min_value=0, max_value=1, width=32))
  @settings(max_examples=10,
            deadline=None)
  def test_update_rmsprop(self, w, h, c, lr, momentum, rho, decay):

    epsilon = 1e-10
    rms = RMSprop(lr=lr, rho=rho, epsilon=epsilon, decay=decay)
    tf_rms = tf.keras.optimizers.RMSprop(learning_rate=lr, rho=rho, epsilon=epsilon, momentum=momentum, decay=decay)

    first  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    second = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)
    third  = np.random.uniform(low=-100, high=100., size=(w, h, c)).astype(float)

    tf_first  = tf.Variable(first.copy())
    tf_second = tf.Variable(second.copy())
    tf_third  = tf.Variable(third.copy())

    first_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    second_update = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)
    third_update  = np.random.uniform(low=-1000, high=1000, size=(w, h, c)).astype(float)

    tf_first_update  = tf.Variable(first_update.copy())
    tf_second_update = tf.Variable(second_update.copy())
    tf_third_update  = tf.Variable(third_update.copy())

    params = [first, second, third]
    grads  = [first_update, second_update, third_update]

    tf_params = [tf_first, tf_second, tf_third]
    tf_grads  = [tf_first_update, tf_second_update, tf_third_update]

    # Updates variables
    first_up, second_up, third_up = rms.update(params, grads)
    tf_rms.apply_gradients(zip(tf_grads, tf_params))

    tf_updated1 = tf_params[0].numpy()
    tf_updated2 = tf_params[1].numpy()
    tf_updated3 = tf_params[2].numpy()

    np.testing.assert_allclose(first_up,  tf_updated1, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(second_up, tf_updated2, atol=1e-3, rtol=1e-3)
    np.testing.assert_allclose(third_up,  tf_updated3, atol=1e-3, rtol=1e-3)
