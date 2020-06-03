# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf

from NumPyNet.utils import to_categorical
from NumPyNet.utils import from_categorical

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings
from hypothesis import example

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestUtils:
  '''
  Test functions in the utils.py file

  -to_categorical
  -from_categorical
  '''

  @given(size = st.integers(min_value=10, max_value=100),
         num_labels = st.integers(min_value=1, max_value=120))
  @settings(max_examples=100, deadline=None)
  def test_to_categorical(self, size, num_labels):

    label = np.random.randint(low=0, high=num_labels, size=(size,))

    categorical_tf = tf.keras.utils.to_categorical(label, num_classes=None)
    categorical_np = to_categorical(label)

    np.testing.assert_allclose(categorical_tf, categorical_np)

  @given(size = st.integers(min_value=10, max_value=100),
         num_labels = st.integers(min_value=0, max_value=120))
  @settings(max_examples=100, deadline=None)
  def test_from_categorical(self, size, num_labels):

    label = np.random.uniform(low=0, high=num_labels, size=(size,))

    categorical_tf = tf.keras.utils.to_categorical(label, num_classes=None)
    categorical_np = to_categorical(label)

    np.testing.assert_allclose(categorical_tf, categorical_np)

    fromlabel_tf = tf.math.argmax(categorical_tf, axis=-1)
    fromlabel_np = from_categorical(categorical_np)

    np.testing.assert_allclose(fromlabel_tf, fromlabel_np)
