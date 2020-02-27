#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Route_layer():

  def __init__(self, input_layers, by_channels=True, **kwargs):
    '''
    Route layer
      For Now the idea is : it takes the seleted layers output and concatenate
      them along the batch axis OR the channels axis

    YOLOv3 implementation always concatenate by channels

    By definition, this layer can't be used without a Network model.

    Parameters
    ----------
      input_layers: iterable, list of integers, or single integer, index of the layers in the network for which
        inputs have to concatenated.
      by_channels   : bool, default True. It determines along
        which dimension the concatenation is performed. For examples if two
        input with size (b1, w, h , c) and (b2, w, h, c) are concatenated with by_channels=False,
        then the final output shape will be (b1 + b2, w, h, c).
        Otherwise, if the shapes are (b, w, h, c1) and (b, w, h, c2) and axis=3, the final output size
        will be (b, w, h, c1 + c2) (YOLOv3 model)
    '''

    if by_channels :
      self.axis = 3  # axis for the concatenation
    else:
      self.axis = 0

    if isinstance(input_layers, int):
      self.input_layer = (input_layers, )

    elif hasattr(input_layers, '__iter__'):
      self.input_layers = tuple(input_layers)

    else :
      raise ValueError('Route Layer : parameter "input_layer" is neither iterable or an integer')

    self.output     = np.array([], dtype=float)
    self._out_shape = None

  def __str__(self):
    return 'route   {}'.format(list(self.input_layers))

  def __call__(self, previous_layer):

    self._out_shape = [0,0,0,0]

    if self.axis:  # by channels
      for prev in previous_layer:

        if prev.out_shape is None:
          class_name = self.__class__.__name__
          prev_name  = previous_layer.__class__.__name__
          raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

        c = prev.out_shape[3]
        self._out_shape[3] += c
        self._out_shape[0:3] = prev.out_shape[0:3]

    else : # by batch
      for prev in previous_layer:

        if prev.out_shape is None:
          class_name = self.__class__.__name__
          prev_name  = previous_layer.__class__.__name__
          raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

        b = prev.out_shape[0]
        self._out_shape[0] += b
        self._out_shape[1:4] = prev.out_shape[1:4]


    self._out_shape = tuple(self._out_shape)
    return self

  @property
  def out_shape(self):
    return self._out_shape

  def forward(self, network):
    '''
    Concatenate along chosen axis the outputs of selected network layers
    In main CNN applications, like YOLOv3, the concatenation happens long channels axis

    Parameters
    ----------
      network : Network object type.

    Returns
    -------
      Route Layer object
    '''

    self.output = np.concatenate([network[layer_idx].output for layer_idx in self.input_layers], axis=self.axis)
    self.delta  = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta, network):
    '''
    Sum self.delta to the correct layer delta on the network

    Parameters
    ----------
      delta  : 4-d numpy array, network delta to be backpropagated
      network: Network object type.

    Returns
    -------
      Route layer object
    '''

    check_is_fitted(self, 'delta')

    # NumPyNet implementation
    if self.axis == 3:            # this works for concatenation by channels axis
      channels_sum = 0
      for idx in self.input_layers:
        channels = network[idx].out_shape[3]
        network[idx].delta += self.delta[:,:,:, channels_sum : channels_sum + channels]
        channels_sum += channels

    elif self.axis == 0:          # this works for concatenation by batch axis
      batch_sum = 0
      for idx in self.self.input_layers:
        batches = network[idx].out_shape[0]
        network[idx].delta += self.delta[batch_sum : batch_sum + batches,:,:,:]
        batch_sum += batches

    return self


if __name__ == '__main__':

  layer = Route_layer((1,2))
  print(layer)

  print(layer.out_shape)
  # TODO the idea is to create a toy model for numpynet and keras, and try some
  #      concatenation (mainly by channel, since the batch implementation doesn't really
  #      make sense to me)
