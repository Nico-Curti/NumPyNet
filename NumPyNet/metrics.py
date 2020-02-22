#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


def mean_accuracy_score (y_true, y_pred):
  '''
  Compute average accuracy score of a classification.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average accuracy between the two inputs
  '''
  return np.mean(np.equal(y_true, y_pred))


def mean_square_error (y_true, y_pred):
  '''
  Compute average square error score of a classification.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average square error between the two inputs
  '''
  diff = y_truth - y_pred
  diff *= diff
  return np.mean(diff)


def mean_absolute_error (y_true, y_pred):
  '''
  Compute average absolute error score of a classification.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average absolute error between the two inputs
  '''
  diff = np.abs(y_truth - y_pred)
  return np.mean(diff)


def mean_logcosh (y_true, y_pred):
  '''
  Compute average logcosh score of a classification.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average logcosh error between the two inputs
  '''
  diff = np.log(np.cosh(y_truth - y_pred))
  return np.mean(diff)


def mean_hellinger (y_true, y_pred):
  '''
  Compute average hellinger score of a classification.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average hellinger error between the two inputs
  '''
  diff = np.sqrt(y_true) - np.sqrt(y_pred)
  diff *= diff
  return np.mean(diff)


def mean_iou_score (y_true, y_pred):
  '''
  Compute average IoU score of a classification.
  IoU is computed as Intersection Over Union between true and predict labels.

  It's a tipical metric in segmentation problems, so we encourage to use it
  when you are dealing image processing tasks.

  Parameters
  ----------
    y_true : 2d array-like
      Ground truth (correct) labels expressed as image.

    y_pred : 2d array-like
      Predicted labels, as returned by the NN

  Returns
  -------
    score : float
      Average IoU between the two inputs
  '''

  unique_labels = set(y_true.ravel())
  num_unique_labels = len(unique_labels)

  I = np.empty(shape=(num_unique_labels, ), dtype=float)
  U = np.empty(shape=(num_unique_labels, ), dtype=float)

  for i, val in enumerate(unique_labels):

    pred_i = y_pred == val
    lbl_i  = y_true == val

    I[i] = np.sum(np.logical_and(lbl_i, pred_i))
    U[i] = np.sum(np.logical_or(lbl_i, pred_i))

  return np.mean(I / U)


