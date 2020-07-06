# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.metrics import mean_accuracy_score
from NumPyNet.optimizer import Adam
from NumPyNet.network import Network
from NumPyNet.exception import MetricsError

import pytest

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestNetwork:
  '''
  Tests:
    - add default metrics
  '''

  def test_add_metrics (self):

    check_function_equality = lambda f1, f2 : f1.__code__.co_code == f2.__code__.co_code

    custom_metrics_wrong = lambda y_true, y_pred, a : None
    custom_metrics_default = lambda y_true, y_pred, a=3.14 : None

    model = Network(batch=42, input_shape=(1, 1, 1))

    model.compile(optimizer=Adam(), metrics=[mean_accuracy_score])
    assert model.metrics == [mean_accuracy_score]
    assert all(check_function_equality(x1, x2) for x1, x2 in zip(model.metrics, [mean_accuracy_score]))

    model.compile(optimizer=Adam(), metrics=[custom_metrics_default])
    assert model.metrics == [custom_metrics_default]
    assert all(check_function_equality(x1, x2) for x1, x2 in zip(model.metrics, [custom_metrics_default]))

    with pytest.raises(MetricsError):
      model.compile(optimizer=Adam(), metrics=[custom_metrics_wrong])

