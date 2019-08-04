#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Logistic Layer'


class Logistic_layer(object):

  def __init__(self):
    '''
    Logistic Layer: performs a logistic transformation of the input and computes
      the binary cross entropy cost.

    Parameters:
    '''
    self.w, self.h, self.c  = (0, 0, 0)
    self.output, self.delta, self.loss  = (None, None, None)

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape()
    return 'logistic x entropy                       {:>4d} x{:>4d} x{:>4d} x{:>4d}' .format(
           batch, out_width, out_height, out_channels)

  def out_shape(self):
    return (self.batch, self.w, self.h, self.c)

  def forward(self, inpt, truth=None) :
    '''
    Forward function of the logistic layer, now the output should be consistent with darknet

    Parameters:
      inpt : output of the network with shape (batch, w, h, c)
      truth : arrat of same shape as input (without the batch dimension),
        if given, the function computes the binary cross entropy
    '''

    self.batch, self.w, self.h, self.c = inpt.shape
#    inpt = np.log(inpt/(1-inpt))
    self.output = 1. / (1. + np.exp(-inpt)) # as for darknet
#    self.output = inpt

    if truth is not None:
      self.loss = -truth * np.log(self.output) - (1. - truth) * np.log(1. - self.output)
      self.delta = truth - self.output
#      self.cost = np.mean(self.loss)
      self.cost = np.sum(self.loss) # as for darknet
    else :
      self.delta = np.zeros(shape = self.out_shape())

  def backward(self, delta=None):
    '''
    Backward function of the Logistic Layer

    Paramters:
      delta : array same shape as the input.
    '''
    if delta is not None:
      delta[:] += self.delta # as for darknet, probably an approx



if __name__ == '__main__':


  np.random.seed(123)
  batch, w, h, c = (5,10,10,3)

  # Binary truth, or 0 or 1
  truth = np.random.choice([0., 1.], p=[.5, .5], size=(w, h, c))

  # Random input
  inpt = np.random.uniform(low=0., high=1., size=(batch, w, h, c))

  # Model Initialization
  layer = Logistic_layer()

  # FORWARD

  layer.forward(inpt, truth)
  byron_loss = layer.cost

  print(layer)
  print('Byron loss: {:.3f}'.format(byron_loss))

  # BACKWARD

  delta_byron = np.zeros(shape=inpt.shape)
  layer.backward(delta_byron)



