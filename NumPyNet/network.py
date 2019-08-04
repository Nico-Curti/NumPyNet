#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np
import pickle

from .layers.input_layer import Input_layer
from .parser import net_config

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Network model'


class Network(object):

  def __init__(self, input_shape, batch):
    '''
    '''
    try:

      self.w, self.h, self.c = input_shape

    except:
      raise ValueError('Network model : incorrect input_shape. Expected a 3D array (width, height, channel). Given {}'.format(input_shape))

    self.net = [ Input_layer((batch, *input_shape)) ]
    self.batch = batch
    self.train = train

  def add_layer(self, layer):
    '''
    '''
    # TODO add input_shape as first input argument of each layer object!
    if self.net[-1].out_shape() != layer.input_shape:
      raise ValueError('Incorrect shape found')

    self.net.append(layer)

  def __iter__(self):
    self.layer_index = 1 # the first layer is the input one
    return self.net[self.layer_index]

  def __next__(self):
    if self.layer_index < self.num_layers:
      self.layer_index += 1
      return self.net[self.layer_index]

    else:
      raise StopIteration


  def load(self, cfg_filename, weights=None)

    model = net_config(cfg_filename)
    # MISS loading model

    if weights is not None:
      self.load_weights(weights)

  def load_weights(self, weights_filename):
    with open(weights_filename, 'rb') as fp:

      for layer in self:
        if hasattr(layer, 'load_weights'):
          layer.load_weights(fp)

  def save_weights(self, filename):
    with open(weights_filename, 'wb') as fp:

      for layer in self:
        if hasattr(layer, 'save_weights'):
          layer.save_weights(fp)

  def load_model(self, model_filename):
    with open(model_filename, 'rb') as fp:
      tmp_dict = pickle.load(fp)

    self.__dict__.clear()
    self.__dict__.update(tmp_dict)


  def save_model(self, model_filename):
    with open(model_filename, 'wb') as fp:
      pickle.dump(self.__dict__, fp, 2)


  @property
  def shape(self):
    return self.net[0].out_shape()[1:]

  @property
  def num_layers(self):
    return len(self.net)




