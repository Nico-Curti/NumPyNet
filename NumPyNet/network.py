#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import os
import re
import sys
import pickle
from time import time as now

from NumPyNet.layers.activation_layer import Activation_layer
from NumPyNet.layers.avgpool_layer import Avgpool_layer
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.l1norm_layer import L1Norm_layer
from NumPyNet.layers.l2norm_layer import L2Norm_layer
from NumPyNet.layers.logistic_layer import Logistic_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.shortcut_layer import Shortcut_layer
from NumPyNet.layers.shuffler_layer import Shuffler_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.upsample_layer import Upsample_layer
from NumPyNet.layers.yolo_layer import Yolo_layer

from NumPyNet.parser import net_config
from NumPyNet.exception import DataVariableError
from NumPyNet.exception import LayerError
from NumPyNet.exception import NetworkError

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
            'l1norm'        :  L1Norm_layer,
            'l2norm'        :  L2Norm_layer,
            'logistic'      :  Logistic_layer,
            'maxpool'       :  Maxpool_layer,
            'route'         :  Route_layer,
            'shortcut'      :  Shortcut_layer,
            'shuffler'      :  Shuffler_layer,
            'softmax'       :  Softmax_layer,
            'upsample'      :  Upsample_layer,
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

      self._net = [ Input_layer(input_shape=(self.batch, self.w, self.h, self.c)) ]

    else:
      self._net = []

    self._fitted = False


  def add(self, layer):
    '''
    Add a new layer to the network model.
    Layers are progressively appended to the tail of the model.
    '''
    try:
      type_layer = layer.__class__.__name__.lower().split('_layer')[0]

    except:
      raise LayerError('Incorrect Layer type found. Given {}'.format(type_layer.__class__.__name__))

    if type_layer not in self.LAYERS.keys():
      raise LayerError('Incorrect Layer type found.')

    if type_layer == 'input':
      self._net.append(layer)

    else:
      self._net.append(layer(self._net[-1]))

    return self

  def __iter__(self):
    self.layer_index = 0
    return self

  def __next__(self):
    if self.layer_index < self.num_layers-1:
      self.layer_index += 1
      return self._net[self.layer_index]

    else:
      raise StopIteration


  def summary(self):
    '''
    Print the network model summary
    '''
    print('layer       filters  size              input                output')
    for i, layer in enumerate(self._net):
      print('{:>4d} {}'.format(i, self._net[i]), flush=True, end='\n')


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
    self._net = [ Input_layer(input_shape=input_shape) ]

    print('layer     filters    size              input                output')

    for i, layer in enumerate(model):
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
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)([self._net[-1], self._net[_from]]) )

      elif layer_t == 'route':
        _layers = model.get(layer, 'layers', [])
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)(self._net[_layers]) )

      else:
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **layer_params)(self._net[-1]) )

      print('{:>4d} {}'.format(i, self._net[-1]), flush=True, end='\n')

      #if model.get(layer, 'batch_normalize', 0): # wrong because it add a new layer and so the shortcut is broken
      #  self._net.append( BatchNorm_layer()(self._net[-1]) )
      #  print('{:>4d} {}'.format(i, self._net[-1]), flush=True, end='\n')

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

    self._fitted = True

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

    self._fitted = True

    return self


  def save_model(self, model_filename):
    '''
    Dump the current network model as pickle
    '''
    with open(model_filename, 'wb') as fp:
      pickle.dump(self.__dict__, fp, 2)

    return self


  def fit(self, X, y, max_iter=100, shuffle=True):
    '''
    '''

    num_data = len(X)
    batches  = num_data // self.batch

    for _ in range(max_iter):

      start = now()

      sys.stdout.write('Iter {:d}/{:d}\n'.format(_ + 1, max_iter))
      sys.stdout.flush()

      for i in range(batches):

        current_batch = i * self.batch
        input = X[current_batch : current_batch + self.batch]
        truth = y[current_batch : current_batch + self.batch]

        out = self._forward(input, truth)
        self._backward(input)

        loss = self._get_loss()

        done = int(50 * (i + 1) / batches)
        sys.stdout.write('\r%d/%d |%s%s| (%1.1f iter/sec) loss=%3.3f' % ( i + 1, batches,
                                                                         r'█' * done,
                                                                          '-' * (50 - done),
                                                                          (now() - start) / batches,
                                                                          loss
                                                                         ))
        sys.stdout.flush()

      sys.stdout.write('\n')

    self._fitted = True


  def fit_generator(self, Xy_generator, max_iter=100):
    '''
    Fit function using a train generator (ref. DataGenerator in data.py)
    '''

    Xy_generator.start()

    for i in range(max_iter):

      grabbed = False

      while not grabbed:

        data, label, grabbed = Xy_generator.load_data()


      self.fit(data, label, max_iter=1, shuffle=False) # data already shuffled

    Xy_generator.stop()

    self._fitted = True


  def predict(self, X):
    '''
    Predict the given input
    '''
    if not self._fitted:
      raise NetworkError('This Network model instance is not fitted yet. Please use the "fit" function before the predict')

    output = self._forward(X)
    return output


  def _forward(self, X, truth=None):
    '''
    Forward function.
    Apply the forward method on all layers
    '''

    y = X.copy()

    for layer in self:

      if hasattr(layer, 'truth') and truth is not None:
        layer.forward(inpt=y, truth=truth)

      else :
        layer.forward(inpt=y)

      y = layer.output

    return y

  def _backward(self, X):
    '''
    BackPropagate the error
    '''

    for i in reversed(range(1, self.num_layers)):

      input = self._net[i - 1].output
      delta = self._net[i - 1].delta

      backward_args = self._net[i].backward.__code__.co_varnames

      if 'inpt' in backward_args:
        self._net[i].backward(inpt=input, delta=delta)

      else:
        self._net[i].backward(delta=delta)

      if hasattr(self._net[i], 'update'):

        self._net[i].update()

    # last iteration
#    delta = None
#    input = X.copy()

    self._net[0].backward(delta)


  def _get_loss(self):
    '''
    Extract the loss value as the last cost in the network model
    '''

    for i in reversed(range(1, self.num_layers)):

      if hasattr(self._net[i], 'cost'):
        return self._net[i].cost

    return None


  @property
  def out_shape(self):
    '''
    Output shape
    '''
    return self._net[0].out_shape()[1:]

  @property
  def input_shape(self):
    '''
    Output shape
    '''
    return (self.w, self.h, self.c)

  @property
  def num_layers(self):
    return len(self._net)


if __name__ == '__main__':

  import os

  batch = 32
  w, h, c = (512, 512, 3)

  config_filename = os.path.join(os.path.dirname(__file__), '..', 'cfg', 'yolov3.cfg')

  net = Network(batch=batch)
  net.load(config_filename)
  print(net.input_shape)

  #net.add(Input_layer(input_shape=(batch, w, h, c)))
  #net.add(Convolutional_layer(input_shape=(batch, w, h, c), filters=64, size=3, stride=1))
  #net.add(Convolutional_layer(input_shape=(batch, w, h, c), filters=16, size=3, stride=1))

  net.summary()
