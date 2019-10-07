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

  def __init__(self, axis=0,**kwargs):
    '''
    Route layer
      For Now the idea is : it takes the seleted layers output and concatenate them

    Parameters:
      axis : int, default 0. Accepted values are 0 or 3. It determines along
        which dimension the concatenation is performed. for examples if two
        input with size (b1, w, h , c) and (b2, w, h, c) are concatenated along axis=0,
        then the final output shape will be (b1 + b2, w, h, c). Otherwise, if the
        shapes are (b, w, h, c1) and (b, w, h, c2) and axis=3, the final output size
        will be (b, w, h, c1 + c2)
    '''

    if axis > 1 or axis < 0 :
      raise LayerError('Incorrect value of axis: accepted values are 0 or 3')

    self.axis = axis
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

  def forward(self, inpt, network):

    for layer_idx in self.input_layers:
      self.output = np.concatenate(self.output, network[layer_idx].output, axis=self.axis)

    self.delta = np.zeros(shape=self.output.shape, dtype=float)

  def backward(self, delta, network):

    for layer_idx in self.input_layers:
      delta[:] += network[layer_idx].delta # not really sure here


if __name__ == '__main__':

  layer = Route_layer()
#  print(layer)
  print('Insert test visualization here')
