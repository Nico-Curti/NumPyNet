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

  def __init__(self, input_layers, by_channels=True, **kwargs):
    '''
    Route layer
      For Now the idea is : it takes the seleted layers output and concatenate
      them along the batch axis OR the channels axis

    YOLOv3 implementation always concatenate by channels

    Parameters:
      input_layers: iterable, list of integers, index of the layers in the network for which
        inputs have to concatenated
      by_channels   : bool, default True. It determines along
        which dimension the concatenation is performed. For examples if two
        input with size (b1, w, h , c) and (b2, w, h, c) are concatenated with by_channles=False,
        then the final output shape will be (b1 + b2, w, h, c).
        Otherwise, if the shapes are (b, w, h, c1) and (b, w, h, c2) and axis=3, the final output size
        will be (b, w, h, c1 + c2) (YOLOv3 model)
    '''
    if by_channels :
      self.axis = 3  # axis for the concatenation
    else:
      self.axis = 0

    self.input_layers = input_layers
    # self.input_layers  = kwargs.pop('layers', [])
    self.outputs = np.array([], dtype=float)
    self._out_shape = None

  def __str__(self):
    # return 'route   [{}]'.format(' '.join(map(str(self._out_shape)))) # WRONG
    return 'route   {}'.format([idx for idx in self.input_layers]).translate({ord(i) : None for i in '[],'})

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

  def forward(self, network):
    '''
    Concatenate along chosen axis the outputs of selected network layers
    In main CNN applications, like YOLOv3, the concatenation happens long channels axis

    Parameters:
      network : Network object type.
    '''

    self.output = np.concatenate([network[layer_idx] for layer_idx in self.input_layers], axis=self.axis)
    # self.delta  = np.zeros(shape=self.output.shape, dtype=float)  # i don't think this is necessary

  def backward(self, delta, network):
    '''
    Sum self.delta to the correct layer delta on the network

    Parameters:
      delta  : 4-d numpy array, network delta to be backpropagated
      network: Network object type.
    '''

    # # darknet:
    # delta = delta.ravel()
    # offset = 0
    # n = len(self.input_layers)
    # for i in range(n):
    #   index      = self.input_layers[i]
    #   delta      = network[index].delta.ravel()
    #   input_size = np.prod(network[index]._out_shape[1:])
    #   for j in range(_out_shape[0]):  # range(batch)
    #     for k in range(input_size):
    #       delta[j*input_size + k] += self.delta[offeset + j*out_shape + k]
    #   offeset += input_size

    # NumPyNet implementation
    if self.axis == 3:            # this works for concatenation by channels axis
      channels_sum = 0
      for idx in self.input_layers:
        channels = network[idx].out_shape[3]
        network[idx].delta += delta[:,:,:, channels_sum : channels_sum + channels]
        channels_sum += channels

    elif self.axis == 0:          # this works for concatenation by batch axis
      batch_sum = 0
      for idx in self.self.input_layers:
        batches = network[idx].out_shape[0]
        network[idx].delta += delta[batch_sum : batch_sum + batches,:,:,:]
        batch_sum += batches


if __name__ == '__main__':

  layer = Route_layer((1,2))
  print(layer)

  print(layer.out_shape)
  # TODO the idea is to create a toy model for numpynet and keras, and try some
  #      concatenation (mainly by channel, since the batch implementation doesn't really
  #      make sense to me)
