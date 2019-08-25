#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

from NumPyNet.box import Box

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Detection object'


class Detection(object):

  def __init__(self, num_classes=None, mask_size=None):

    self._prob = []*num_classes if num_classes is not None else None
    self._mask = []*mask_size   if mask_size   is not None else None

    self._objectness = None
    self._box = Box()

  @property
  def box(self):
    return self._box.box

  @property
  def objectness(self):
    return self._objectness

  @property
  def prob(self):
    return self._prob


  @staticmethod
  def top_k_predictions(output):
    '''
    Compute the indexes of the sorted output
    '''
    return np.argsort(output)[::-1] # it is the fastest way to obtain a descending order


  @staticmethod
  def do_nms_obj(detections, thresh):
    '''
    '''

    # filter 0 objectness
    detections = filter(lambda x : x.objectness != 0, dets)

    # sort the objectness
    detections = sorted(detections, key=lambda x : x.objectness, reverse=True)

    # MISS


  @staticmethod
  def do_nms_sort(detections, thresh):
    '''
    '''

    # filter 0 objectness
    detections = filter(lambda x : x.objectness != 0, dets)

    # sort the objectness
    detections = sorted(detections, key=lambda x : x.objectness, reverse=True)

    # MISS

  def __str__(self):
    probs = ' '.join(['{:.3f}'.format(x) for x in self.prob])
    return '{0:.3f}: {1}'.format(self.objectness, probs)


if __name__ == '__main__':

  print('insert testing here')
