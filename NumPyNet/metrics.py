#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Score metrics'


def mean_iou_score (y_true, y_pred):

  unique_labels = set(y_true)
  num_unique_labels = len(unique_labels)

  I = np.empty(shape=(num_unique_labels, ), dtype=float)
  U = np.empty(shape=(num_unique_labels, ), dtype=float)

  for i, val in enumerate(unique_labels):

    pred_i = y_pred == val
    lbl_i  = y_true == val

    I[i], U[i] = np.sum(np.logical_and(lbl_i, pred_i)), np.sum(np.logical_or(lbl_i, pred_i))

  return np.mean(I / U)


