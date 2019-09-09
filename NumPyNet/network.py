#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import os
import re
import pickle

from NumPyNet.layers.activation_layer import Activation_layer
from NumPyNet.layers.avgpool_layer import Avgpool_layer
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.logistic_layer import Logistic_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.shortcut_layer import Shortcut_layer
from NumPyNet.layers.shuffler_layer import Shuffler_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.yolo_layer import Yolo_layer

from NumPyNet.parser import net_config
from NumPyNet.exception import DataVariableError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Network model'


class Network(object):

  LAYERS = {'activation'    :  Activation_layer,
            'avgpool'       :  Avgpool_layer,
            'batchnorm'     :  BatchNorm_layer,
            'connected'     :  Connected_layer,
            'convolutional' :  Convolutional_layer,
            'cost'          :  Cost_layer,
            'dropout'       :  Dropout_layer,
            'input'         :  Input_layer,
            'logistic'      :  Logistic_layer,
            'maxpool'       :  Maxpool_layer,
            'route'         :  Route_layer,
            'shortcut'      :  Shortcut_layer,
            'shuffler'      :  Shuffler_layer,
            'softmax'       :  Softmax_layer,
            'yolo'          :  Yolo_layer,
            }

  def __init__(self, batch, input_shape=None, train=None):
    '''
    '''
    self.batch = batch
    self.train = train

    if input_shape is not None:

      try:

        self.w, self.h, self.c = input_shape

      except:
        raise ValueError('Network model : incorrect input_shape. Expected a 3D array (width, height, channel). Given {}'.format(input_shape))

      self.net = [ Input_layer(input_shape=(self.batch, self.w, self.h, self.c)) ]


  def add_layer(self, layer):
    '''
    '''
    # TODO add input_shape as first input argument of each layer object!
    #if self.net[-1].out_shape() != layer.input_shape:
    #  raise ValueError('Incorrect shape found')
    type_layer = str(type(layer))

    if type_layer not in self.LAYERS.keys():
      raise ValueError('Incorrect Layer type found.')

    self.net.append(layer)

    return self

  def __iter__(self):
    self.layer_index = 0
    return self

  def __next__(self):
    if self.layer_index < self.num_layers:
      self.layer_index += 1
      return self.net[self.layer_index]

    else:
      raise StopIteration


  def load(self, cfg_filename, weights=None):
    '''
    Load network model from config file in INI fmt
    '''

    model = net_config(cfg_filename)

    self.batch = model.get('net1', 'batch', 1)
    self.w = model.get('net1', 'width', 416)
    self.h = model.get('net1', 'height', 416)
    self.c = model.get('net1', 'channels', 3)
    # TODO: add other network parameters

    input_shape = (self.batch, self.w, self.h, self.c)
    self.net = [ Input_layer(input_shape=input_shape) ]

    for layer in model:
      layer_t = re.split(r'\d+', layer)[0]
      params = dict(model.get_params(layer))

      layer_params = {}
      for k, v in params.items():
        try:
          val = eval(v)
        except NameError:
          val = v
        except:
          raise DataVariableError('Type variable not recognized! Possible variables are only [int, float, string, vector<float>].')

        layer_params[k] = val

      if layer_t == 'shortcut':
        _from = model.get(layer, 'from', 0)
        self.net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)([self.net[-1], self.net[_from]]) )

      elif layer_t == 'route':
        _layers = model.get(layer, 'layers', [])
        self.net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)(self.net[_layers]) )

      else:
        self.net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)(self.net[-1]) )

      print(self.net[-1], flush=True, end='\n')

    return self


    if weights is not None:
      self.load_weights(weights)

  def load_weights(self, weights_filename):
    '''
    Load weight from filename in binary fmt
    '''
    with open(weights_filename, 'rb') as fp:

      major, minor, revision = np.fromfile(fp, dtype=np.int, count=3)
      full_weights = np.fromfile(fp, dtype=np.float, count=-1)

    pos = 0
    for layer in self:
      if hasattr(layer, 'load_weights'):
        pos = layer.load_weights(full_weights, pos)

    return self

  def save_weights(self, filename):
    '''
    Dump current network weights
    '''
    full_weights = []

    for layer in self:
      if hasattr(layer, 'save_weights'):
        full_weights += layer.save_weights()

    full_weights = np.asarray(full_weights, dtype=np.float)
    version = np.array([1, 0, 0], dtype=np.int)

    with open(filename, 'wb') as fp:
      version.tofile(fp, sep='')
      full_weights.tofile(fp, sep='') # for binary format

    return self

  def load_model(self, model_filename):
    '''
    Load network model object as pickle
    '''
    with open(model_filename, 'rb') as fp:
      tmp_dict = pickle.load(fp)

    self.__dict__.clear()
    self.__dict__.update(tmp_dict)

    return self


  def save_model(self, model_filename):
    '''
    Dump the current network model as pickle
    '''
    with open(model_filename, 'wb') as fp:
      pickle.dump(self.__dict__, fp, 2)

    return self


  @property
  def out_shape(self):
    '''
    Output shape
    '''
    return self.net[0].out_shape()[1:]

  @property
  def input_shape(self):
    '''
    Output shape
    '''
    return (self.w, self.h, self.c)

  @property
  def num_layers(self):
    return len(self.net)


if __name__ == '__main__':

  import os

  config_filename = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'yolov3.cfg')
  weight_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'yolov3.weights.byron')
  mask_w_filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'yolov3.weights.mask')

  net = Network(batch=32)
  net.load(config_filename)

  print(net.input_shape)



