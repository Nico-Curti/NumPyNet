#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
import operator as op

from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Route_layer(BaseLayer):
  '''
  Route layer
    For Now the idea is: it takes the seleted layers output and concatenate
    them along the batch axis OR the channels axis

  YOLOv3 implementation always concatenate by channels

  By definition, this layer can't be used without a Network model.

  Parameters
  ----------
    input_layers : int or list of int.
      indexes of the layers in the network for which the outputs have to concatenated.

    by_channels : bool, (default = True).
      It determines along which dimension the concatenation is performed. For examples if two
      input with size (b1, w, h , c) and (b2, w, h, c) are concatenated with by_channels=False,
      then the final output shape will be (b1 + b2, w, h, c).
      Otherwise, if the shapes are (b, w, h, c1) and (b, w, h, c2) and axis=3, the final output size
      will be (b, w, h, c1 + c2) (YOLOv3 model). Notice that the all the other dimensions must be equal.

  Example
  -------
  TODO

  Reference
  ---------
  TODO
  '''

  def __init__(self, input_layers, by_channels=True, **kwargs):

    self.axis = 3 if by_channels else 0

    if isinstance(input_layers, int):
      self.input_layer = (input_layers, )

    elif hasattr(input_layers, '__iter__'):
      self.input_layers = tuple(input_layers)

    else:
      raise ValueError('Route Layer : parameter "input_layer" is neither iterable or an integer')

    super(Route_layer, self).__init__()

  def __str__(self):
    return 'route   {}'.format(list(self.input_layers))

  def _build(self, previous_layer):

    out_shapes = [x.out_shape for x in previous_layer]
    self.input_shape = list(out_shapes[-1])

    if self.axis:
      print(np.sum(map(op.itemgetter(self.axis), out_shapes)))
      self.input_shape[-1] = np.sum(list(map(op.itemgetter(self.axis), out_shapes)))
    else:
      self.input_shape[0] = np.sum(list(map(op.itemgetter(self.axis), out_shapes)))

    self.input_shape = tuple(self.input_shape)
    return self

  def __call__(self, previous_layer):

    for prev in previous_layer:

      if prev.out_shape is None:
        class_name = self.__class__.__name__
        prev_name = prev.__class__.__name__
        raise LayerError('Incorrect shapes found. Layer {0} cannot be connected to the previous {1} layer.'.format(class_name, prev_name))

    self._build(previous_layer)
    return self

  def forward(self, network):
    '''
    Concatenate along chosen axis the outputs of selected network layers
    In main CNN applications, like YOLOv3, the concatenation happens long channels axis

    Parameters
    ----------
      network : Network object type.
        The network model to which this layer belongs to.

    Returns
    -------
      self
    '''

    self.output = np.concatenate([network[layer_idx].output for layer_idx in self.input_layers], axis=self.axis)
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta, network):
    '''
    Sum self.delta to the correct layer delta on the network

    Parameters
    ----------
      delta : array-like
        delta array of shape (batch, w, h, c). Global delta to be backpropagated.

      network: Network object type.
        The network model to which this layer belongs to.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')

    # NumPyNet implementation
    if self.axis == 3:  # this works for concatenation by channels axis
      channels_sum = 0
      for idx in self.input_layers:
        channels = network[idx].out_shape[3]
        network[idx].delta += self.delta[..., channels_sum: channels_sum + channels]
        channels_sum += channels

    elif self.axis == 0:  # this works for concatenation by batch axis
      batch_sum = 0
      for idx in self.self.input_layers:
        batches = network[idx].out_shape[0]
        network[idx].delta += self.delta[batch_sum: batch_sum + batches, ...]
        batch_sum += batches

    return self


if __name__ == '__main__':

  layer = Route_layer((1, 2))
  print(layer)

  print(layer.out_shape)
  # TODO the idea is to create a toy model for numpynet and keras, and try some
  #      concatenation (mainly by channel, since the batch implementation doesn't really
  #      make sense to me)
