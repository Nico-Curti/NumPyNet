# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.metrics import mean_accuracy_score
from NumPyNet.metrics import mean_square_error
from NumPyNet.metrics import mean_absolute_error
from NumPyNet.metrics import mean_logcosh
from NumPyNet.metrics import mean_hellinger
from NumPyNet.metrics import mean_iou_score

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestMetrics:

  @given(size = st.integers(min_value=10, max_value=100))
  @settings(max_examples=10, deadline=None)
  def test_mean_accuracy_score (self, size):
    y_true = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))
    y_pred = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))

    metric = tf.keras.metrics.Accuracy()

    res_py = mean_accuracy_score(y_true, y_pred)

    metric.update_state(y_true, y_pred)
    res_tf = metric.result().numpy()

    np.testing.assert_allclose(res_tf, res_py, atol=1e-8, rtol=1e-5)


  @given(size = st.integers(min_value=10, max_value=100))
  @settings(max_examples=10, deadline=None)
  def test_mean_absolute_error (self, size):
    y_true = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))
    y_pred = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))

    metric = tf.keras.metrics.MeanAbsoluteError()

    res_py = mean_absolute_error(y_true, y_pred)

    metric.update_state(y_true, y_pred)
    res_tf = metric.result().numpy()

    np.testing.assert_allclose(res_tf, res_py, atol=1e-8, rtol=1e-5)


  @given(size = st.integers(min_value=10, max_value=100))
  @settings(max_examples=10, deadline=None)
  def test_mean_squared_error (self, size):
    y_true = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))
    y_pred = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))

    metric = tf.keras.metrics.MeanSquaredError()

    res_py = mean_square_error(y_true, y_pred)

    metric.update_state(y_true, y_pred)
    res_tf = metric.result().numpy()

    np.testing.assert_allclose(res_tf, res_py, atol=1e-8, rtol=1e-5)


  @given(size = st.integers(min_value=10, max_value=100))
  @settings(max_examples=10, deadline=None)
  def test_mean_logcosh (self, size):
    y_true = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))
    y_pred = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))

    metric = tf.keras.metrics.LogCoshError()

    res_py = mean_logcosh(y_true, y_pred)

    metric.update_state(y_true, y_pred)
    res_tf = metric.result().numpy()

    np.testing.assert_allclose(res_tf, res_py, atol=1e-8, rtol=1e-5)


  @given(size = st.integers(min_value=10, max_value=100))
  @settings(max_examples=10, deadline=None)
  def test_mean_hellinger (self, size):
    y_true = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))
    y_pred = np.random.choice([0., 1.], p=[.5, .5], size=(size, ))

    res_py = mean_hellinger(y_true, y_pred)

    assert res_py >= 0.
    assert np.allclose(res_py, mean_hellinger(y_pred, y_true))


  @given(size      = st.integers(min_value=10, max_value=100),
         # num_class = st.integers(min_value=2,  max_value=100)
         )
  @settings(max_examples=10, deadline=None)
  def test_mean_iou_score (self, size):
    # working only with two classes for now
    y_true = np.random.choice([0., 1.], size=(size, ))
    y_pred = np.random.choice([0., 1.], size=(size, ))

    metric = tf.keras.metrics.MeanIoU(num_classes=2)

    res_py = mean_iou_score(y_true, y_pred)

    metric.update_state(y_true, y_pred)
    res_tf = metric.result().numpy()

    np.testing.assert_allclose(res_tf, res_py, atol=1e-8, rtol=1e-5)
