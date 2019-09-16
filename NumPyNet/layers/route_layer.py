#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Route layer'


class Route_layer():

  def __init__(self, **kwargs):
    '''
    Route layer
      For Now the idea is : it takes the seleted layers output, concatenate them, and
      then performs a linear combination with the output of the previous layer.

    Parameters:
      input_layers : list of previous layer for which concatenate outputs
    '''

    self.input_layers = kwargs.pop('layers', [])
    self.outputs = np.array([], dtype=float)
    self._out_shape = None

  def __str__(self):
    return 'route   [{}]'.format(' '.join(map(str(self._out_shape)))) # WRONG

  def __call__(self, *previous_layer):

    self.input_layers = []
    self._out_shape = []

    for prev in previous_layer:
      if prev.out_shape is None:
        class_name = self.__class__.__name__
        prev_name  = previous_layer.__class__.__name__
        raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

      self._out_shape.append(prev.out_shape)
      self.input_layers.append(prev)

    return self

  @property
  def out_shape(self):
    self._out_shape

  def forward(self, inpt, net):

    for layer_idx in self.input_layers:
      self.output = np.concatenate(self.output, net.layers[layer_idx].output, axis=1)
    self.delta = np.zeros(shape=self.output.shape, dtype=float)

  def backward(self, delta, net):

    for layer_idx in self.input_layers:
      delta += net.layers[layer_idx].delta


if __name__ == '__main__':

  layer = Route_layer()
#  print(layer)
  print('Insert test visualization here')