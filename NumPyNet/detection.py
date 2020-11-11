#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

from NumPyNet.box import Box

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Detection(object):

  '''
  Detection object

  Parameters
  ----------
    num_classes : int (default=None)
      Number of classes to monitor

    mask_size : int (default=None)
      Size of the possible mask values

  Notes
  -----
  .. note::
    The detection object stores the detection probability of each class and its "objectness".
    Moreover in the member "bbox" are store the detection box infos as Box object, aka (x, y, w, h)
  '''

  def __init__(self, num_classes=None, mask_size=None):

    self._prob = []*num_classes if num_classes is not None else None
    self._mask = []*mask_size   if mask_size   is not None else None

    self._objectness = None
    self._box = Box()

  @property
  def box(self):
    '''
    Return the box object as tuple
    '''
    return self._box.box

  @property
  def objectness(self):
    '''
    Return the objectness of the detection
    '''
    return self._objectness

  @property
  def prob(self):
    '''
    Return the probability of detection for each class
    '''
    return self._prob


  @staticmethod
  def top_k_predictions(output):
    '''
    Compute the indices of the sorted output

    Parameters
    ----------
      output: array_like (1D array)
        Array of predictions expressed as floats.
        Its value will be sorted in ascending order and the corresponding array of indices
        is given in output.

    Returns
    -------
      indexes: list (int32 values)
        Array of indexes which sort the output values in ascending order.
    '''
    return np.argsort(output)[::-1] # it is the fastest way to obtain a descending order


  @staticmethod
  def do_nms_obj(detections, thresh):
    '''
    Sort the detection according to the probability of each class and perform the IOU as filter for the boxes

    Parameters
    ----------
      detections: array_like (1D array)
        Array of detection objects.

      thresh: float
        Threshold to apply for IoU filtering.
        If IoU is greater than thresh the corresponding objectness and probabilities are set to null.

    Returns
    -------
      dets: array_like (1D array)
        Array of detection objects processed.
    '''

    # filter 0 objectness
    detections = filter(lambda x : x.objectness != 0, detections)

    # sort the objectness
    detections = sorted(detections, key=lambda x : x.objectness, reverse=True)

    # MISS


  @staticmethod
  def do_nms_sort(detections, thresh):
    '''
    Sort the detection according to the objectness and perform the IOU as filter for the boxes.

    Parameters
    ----------
      detections: array_like (1D array)
        Array of detection objects.

      thresh: float
        Threshold to apply for IoU filtering.
        If IoU is greater than thresh the corresponding objectness and probabilities are set to null.

    Returns
    -------
      dets: array_like (1D array)
        Array of detection objects processed.
    '''

    # filter 0 objectness
    detections = filter(lambda x : x.objectness != 0, detections)

    # sort the objectness
    detections = sorted(detections, key=lambda x : x.objectness, reverse=True)

    # MISS

  def __str__(self):
    '''
    Printer of objectness and probability
    '''
    probs = ' '.join(['{:.3f}'.format(x) for x in self.prob])
    return '{0:.3f}: {1}'.format(self.objectness, probs)


if __name__ == '__main__':

  print('insert testing here')
