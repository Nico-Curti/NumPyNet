#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Yolo Layer'


class Yolo_layer(object):

  def __init__(self):


  def __str__(self):
    return 'yolo'

  def out_shape(self):
    '''
    '''
    return (self.batch, self.w, self.h, self.c)

  def forward(self, inpt):
    '''
    '''

    self.output = inpt.copy()

    if self.trainable:



    self.delta = np.zeros(shape=self.out_shape, dtype=float)

  def backward(self, delta):
    '''
    '''
    delta[:] += self.delta

  def num_detections(self, thresh):
    '''
    '''

if __name__ == '__main__':
