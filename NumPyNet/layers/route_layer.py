#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Route layer'

class Route_layer():

  def __init__(self, input_layers, **kwargs):
    '''
    Route layer. Incomplete, I need to know how the network object wiil behave.
      For Now the idea is : it takes the seleted layers output, concatenate them, and
      then performs a linear combination with the output of the previous layer.

      This layer will be completed afterwards.

    Paramters:
      input_layers : list of previous layer for which concatenate outputs
    '''

    self.input_layers = input_layers
    self.outputs = np.array([], dtype=float)

  @property
  def out_shape(self):
    self.input_layers[-1].out_shape

  def __str__():
    pass

  def forward(self, inpt, net):

    # I will need a net object, that store all the layers in a list or something like that
    for layer_idx in self.input_layers:
      self.output = np.concatenate(self.output, net.layers[layer_idx].output, axis=1)
    self.delta = np.zeros(shape=self.output.shape, dtype=float)

  def backward(self, delta, net):

    # Do I need a backwad for the route layer?
    for layer_idx in self.input_layers:
      delta += net.layers[layer_idx].delta


if __name__ == '__main__':

  print('Insert test visualization here')