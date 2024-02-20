#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import re
import pickle
from copy import copy
import sys

import inspect
import platform
import numpy as np
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
from NumPyNet.layers.lstm_layer import LSTM_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.rnn_layer import RNN_layer
from NumPyNet.layers.route_layer import Route_layer
from NumPyNet.layers.shortcut_layer import Shortcut_layer
from NumPyNet.layers.shuffler_layer import Shuffler_layer
from NumPyNet.layers.simple_rnn_layer import SimpleRNN_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.upsample_layer import Upsample_layer
from NumPyNet.layers.yolo_layer import Yolo_layer

from NumPyNet.optimizer import Optimizer

from NumPyNet.parser import net_config
from NumPyNet.exception import DataVariableError
from NumPyNet.exception import LayerError
from NumPyNet.exception import MetricsError
from NumPyNet.exception import NetworkError

from NumPyNet.utils import _redirect_stdout

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

CRLF = '\r\x1B[K' if platform.system() != 'Windows' else '\r'

class Network(object):

  '''
  Neural Network object

  Parameters
  ----------
    batch : int
      Batch size

    input_shape : tuple
      Input dimensions

    train : bool (default=True)
      Turn on/off the parameters tuning

  Notes
  -----
  .. warning::
    Up to now the trainable variable is useless since the layer
    doesn't take it into account!
  '''

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
            'lstm'          :  LSTM_layer,
            'maxpool'       :  Maxpool_layer,
            'rnn'           :  RNN_layer,
            'route'         :  Route_layer,
            'shortcut'      :  Shortcut_layer,
            'shuffler'      :  Shuffler_layer,
            'simplernn'     :  SimpleRNN_layer,
            'softmax'       :  Softmax_layer,
            'upsample'      :  Upsample_layer,
            'yolo'          :  Yolo_layer,
            }

  def __init__(self, batch, input_shape=None, train=True):

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

    self.metrics = None
    self._fitted = False


  def add(self, layer):
    '''
    Add a new layer to the network model.
    Layers are progressively appended to the tail of the model.

    Parameters
    ----------
      layer : Layer object
        Layer object to append to the current architecture

    Returns
    -------
      self

    Notes
    -----
    .. note::
      If the architecture is empty a default InputLayer is used
      to start the model.

    .. warning::
      The input layer type must be one of the types stored into the
      LAYERS dict, otherwise a LayerError is raised.
    '''
    try:
      type_layer = layer.__class__.__name__.lower().split('_layer')[0]

    except:
      raise LayerError('Incorrect Layer type found. Given {}'.format(type_layer.__class__.__name__))

    if type_layer not in self.LAYERS.keys():
      raise LayerError('Incorrect Layer type found.')

    if type_layer == 'input':
      self._net.append(layer)

    elif type_layer == 'route':
      prev_layers = []
      for idx in layer.input_layers:
        prev_layers.append(self._net[idx]) # i need layers' info to init route

      self._net.append(layer(prev_layers))

    else:
      self._net.append(layer(self._net[-1]))

    return self

  def __iter__(self):
    '''
    Start the iteration along the architecture
    '''
    self.layer_index = 0
    return self

  def __next__(self):
    '''
    Get the next layer
    '''
    if self.layer_index < self.num_layers:
      self.layer_index += 1
      return self._net[self.layer_index - 1]

    else:
      raise StopIteration

  def next(self):
    '''
    Get the next layer

    Notes
    -----
    This should fix python2* problems with __iter__ and __next__
    '''
    return self.__next__()


  def summary(self):
    '''
    Print the network model summary

    Returns
    -------
      None
    '''
    print('layer       filters  size              input                output')
    for i, layer in enumerate(self._net):
      print('{:>4d} {}'.format(i, layer), end='\n') # flush=True


  def load(self, cfg_filename, weights=None):
    '''
    Load network model from config file in INI fmt

    Parameters
    ----------
      cfg_filename : str
        Filename or path of the neural network configuration file in INI format

      weights : str (default=None)
        Filename of the weights

    Returns
    -------
      self
    '''

    model = net_config(cfg_filename)

    self.batch = model.get('net0', 'batch', 1)
    self.w = model.get('net0', 'width', 416)
    self.h = model.get('net0', 'height', 416)
    self.c = model.get('net0', 'channels', 3)
    # TODO: add other network parameters

    input_shape = (self.batch, self.w, self.h, self.c)
    self._net = [ Input_layer(input_shape=input_shape) ]

    print('layer     filters    size              input                output')

    for i, layer in enumerate(model):
      layer_t = re.split(r'\d+', layer)[0]
      params = model.get_params(layer)

      if layer_t == 'shortcut':
        _from = model.get(layer, 'from', 0)
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **params)([self._net[-1], self._net[_from]]) )

      elif layer_t == 'route':
        _layers = model.get(layer, 'layers', [])
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **params)(self._net[_layers]) )

      else:
        self._net.append( self.LAYERS[layer_t](input_shape=input_shape, **params)(self._net[-1]) )

      input_shape = self._net[-1].out_shape

      print('{:>4d} {}'.format(i, self._net[-1]), end='\n') # flush=True
      sys.stdout.flush() # compatibility with pythonn 2.7

      # if model.get(layer, 'batch_normalize', 0): # wrong because it add a new layer and so the shortcut is broken
      #   self._net.append( BatchNorm_layer()(self._net[-1]) )
      #   print('{:>4d} {}'.format(i, self._net[-1]), flush=True, end='\n')

    if weights is not None:
      self.load_weights(weights)

    return self

  def load_weights(self, weights_filename):
    '''
    Load weight from filename in binary fmt

    Parameters
    ----------
      weights_filename : str
        Filename of the input weights file

    Returns
    -------
      self

    Notes
    -----
    .. note::
      The weights are read and set to each layer which has
      the load_weights member function.
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

    Parameters
    ----------
      filename : str
        Filename of the output weights file

    Returns
    -------
      self

    Notes
    -----
    .. note::
      The weights are extracted from each layer which has
      the save_weights member function.
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

    Parameters
    ----------
      model_filename : str
        Filename or path of the model (binary) file

    Returns
    -------
      self

    Notes
    -----
    .. note::
      The model loading is performed using pickle.
      If the model was previously dumped with the save_model
      function everything should be ok.
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

    Parameters
    ----------
      model_filename : str
        Filename or path for the model dumping

    Returns
    -------
      self

    Notes
    -----
    .. note::
      The model is dumped using pickle binary format.
    '''
    with open(model_filename, 'wb') as fp:
      pickle.dump(self.__dict__, fp, 2)

    return self

  def compile(self, optimizer=Optimizer, metrics=None):
    '''
    Compile the neural network model setting the optimizer
    to each layer and the evaluation metrics

    Parameters
    ----------
      optimizer : Optimizer
        Optimizer object to use during the training

      metrics : list (default=None)
        List of metrics functions to use for the model evaluation.

    Notes
    -----
    .. note::
      The optimizer is copied into each layer object
      which requires a parameters optimization.
    '''

    for layer in self:

      if hasattr(layer, 'optimizer'):
        layer.optimizer = copy(optimizer)

      if isinstance(layer, RNN_layer):
        layer.input_layer.optimizer = copy(optimizer)
        layer.self_layer.optimizer = copy(optimizer)
        layer.output_layer.optimizer = copy(optimizer)

      if isinstance(layer, LSTM_layer):
        layer.uf.optimizer = copy(optimizer)
        layer.ui.optimizer = copy(optimizer)
        layer.ug.optimizer = copy(optimizer)
        layer.uo.optimizer = copy(optimizer)
        layer.wf.optimizer = copy(optimizer)
        layer.wi.optimizer = copy(optimizer)
        layer.wg.optimizer = copy(optimizer)
        layer.wo.optimizer = copy(optimizer)

    if metrics is not None:
      self._check_metrics(metrics)


  def _check_metrics(self, metrics):
    '''
    Check the signature of the given metric functions.
    The right signature must have only two required arguments (y_true, y_pred)
    plus other possible arguments with default values.
    The checked function are added to the list of network metric functions.

    Parameters
    ----------
      metrics : list
        List of metrics as functions

    Returns
    -------
      check : bool
        True if everything is ok
    '''

     # getfullargspec works only in python3.*
    argspec = inspect.getfullargspec if int(sys.version[0]) >= 3 else inspect.getargspec

    for func in metrics:
      if not callable(func):
        raise MetricsError('Metrics {} is not a callable object'.format(func.__name__))

      infos = argspec(func)
      num_defaults = len(infos.defaults) if infos.defaults else 0

      if len(infos.args) - num_defaults != 2:
        raise MetricsError('Metrics {0} is not a valid metrics function. '
                           'The required signature is only func (y_true, y_pred, **kwargs). '
                           'Try to use a partial to overcome this kind of issue.')

    self.metrics = metrics

    return True


  def _evaluate_metrics(self, y_true, y_pred):
    '''
    Evaluate the training metrics

    Parameters
    ----------
      y_true : 2d array-like
        Ground truth (correct) labels expressed as image.

      y_pred : 2d array-like
        Predicted labels, as returned by the NN

    Returns
    -------
      None

    Notes
    -----
    .. note::
      The resulting metrics are just printed in stdout.
    '''

    results = {func.__name__ : func(y_true, y_pred) for func in self.metrics}
    print(' '.join(' {}: {:1.3f}'.format(k, v) for k, v in results.items()))


  def fit(self, X, y, max_iter=100, shuffle=True, verbose=True):
    '''
    Fit/training function

    Parameters
    ----------
      X : array-like
        Input data

      y : array-like
        Ground truth or labels

      max_iter : int (default=100)
        Maximum number of iterations/epochs to perform

      shuffle : bool (default=True)
        Turn on/off the random shuffling of the data

      verbose : bool (default=True)
        Turn on/off the verbosity given by the training progress bar

    Returns
    -------
      self
    '''

    num_data = X.shape[0]
    begin = now()
    self._fitted = True

    indices = np.arange(0, num_data).astype('int64')
    num_batches = num_data // self.batch

    with _redirect_stdout(verbose):
      for _ in range(max_iter):

        start = now()

        print('Epoch {:d}/{:d}'.format(_ + 1, max_iter)) # flush=True)

        sys.stdout.flush() # compatibility with python 2.7

        loss = 0.
        seen = 0

        if shuffle:
          np.random.shuffle(indices)

        batches = np.lib.stride_tricks.as_strided(indices, shape=(num_batches, self.batch), strides=(self.batch * 8, 8))

        for i, idx in enumerate(batches):

          _input = X[idx, ...]
          _truth = y[idx, ...]

          _ = self._forward(X=_input, truth=_truth, trainable=True)
          self._backward(X=_input, trainable=True)

          loss += self._get_loss()
          seen += len(idx)

          done = int(50 * (i + 1) / len(batches))
          print('{}{:>3d}/{:<3d} |{}{}| ({:1.1f} sec/iter) loss: {:3.3f}'.format( CRLF,
                                                                                  seen,
                                                                                  num_data,
                                                                                 r'█' * done,
                                                                                  '-' * (50 - done),
                                                                                  now() - start,
                                                                                  loss / seen
                                                                                ), end='')  # flush=True
          sys.stdout.flush()  # compatibility with pythonn 2.7
          start = now()

        if self.metrics is not None:

          y_pred = self.predict(X, truth=None, verbose=False)
          self._evaluate_metrics(y, y_pred)

        print('\n', end='')  # flush=True)
        sys.stdout.flush()  # compatibility with pythonn 2.7

      end = now()
      print('Training on {:d} epochs took {:1.1f} sec'.format(max_iter, end - begin))

    return self


  def fit_generator(self, Xy_generator, max_iter=100):
    '''
    Fit function using a train generator

    Parameters
    ----------
      Xy_generator : DataGenerator
        Data generator object

      max_iter : int (default=100)
        Maximum number of iterations/epochs to perform

    Returns
    -------
      self

    References
    ----------
    DataGenerator object in data.py
    '''

    Xy_generator.start()

    for _ in range(max_iter):

      grabbed = False

      while not grabbed:

        data, label, grabbed = Xy_generator.load_data()


      self.fit(data, label, max_iter=1, shuffle=False)  # data already shuffled

    Xy_generator.stop()

    self._fitted = True

    return self


  def predict(self, X, truth=None, verbose=True):
    '''
    Predict the given input

    Parameters
    ----------
      X : array-like
        Input data

      truth : array-like (default=None)
        Ground truth or labels

      verbose : bool (default=True)
        Turn on/off the verbosity given by the training progress bar

    Returns
    -------
      output : array-like
        Output of the model as numpy array
    '''
    if not self._fitted:
      raise NetworkError('This Network model instance is not fitted yet. Please use the "fit" function before the predict')

    num_data = len(X)
    _truth = None

    if num_data > 1:
      batches = np.array_split(range(num_data), indices_or_sections=num_data // self.batch)
    else:
      batches = [np.array([0])]

    begin = now()
    start = begin

    loss = 0.
    seen = 0

    output = []

    with _redirect_stdout(verbose):
      for i, idx in enumerate(batches):

        _input = X[idx, ...]
        if truth is not None:
          _truth = truth[idx, ...]

        predict = self._forward(X=_input, truth=_truth, trainable=False)
        output.append(predict)

        loss += self._get_loss()
        seen += len(idx)

        done = int(50 * (i + 1) / len(batches))
        print('{}{:>3d}/{:<3d} |{}{}| ({:1.1f} sec/iter) loss: {:3.3f}'.format( CRLF,
                                                                                seen,
                                                                                num_data,
                                                                               r'█' * done,
                                                                                '-' * (50 - done),
                                                                                now() - start,
                                                                                loss / seen
                                                                              ), end='')  # flush=True,
        sys.stdout.flush()  # compatibility with python 2.7
        start = now()

      print('\n', end='')  # flush=True)
      sys.stdout.flush()  # compatibility with python 2.7


      end = now()
      print('Prediction on {:d} samples took {:1.1f} sec'.format(num_data, end - begin))

    return np.concatenate(output)

  def evaluate(self, X, truth, verbose=False):
    '''
    Return output and loss of the model

    Parameters
    ----------
      X : array-like
        Input data

      truth : array-like
        Ground truth or labels

      verbose : bool (default=False)
        Turn on/off the verbosity given by the training progress bar

    Returns
    -------
      loss : float
        The current loss of the model

      output : array-like
        Output of the model as numpy array
    '''
    output = self.predict(X, truth=truth, verbose=verbose)
    loss = self._get_loss() / len(X)

    return (loss, output)


  def _forward(self, X, truth=None, trainable=True):
    '''
    Forward function.
    Apply the forward method on all layers

    Parameters
    ----------
      X : array-like
        Input data

      truth : array-like (default=None)
        Ground truth or labels

      trainable : bool (default=True)
        Switch if train or not the model

    Returns
    -------
      y : array-like
        Output of the model

    Notes
    -----
    .. warning::
      Up to now the trainable variable is useless since the layer
      doesn't take it into account!
    '''
    # TODO: add trainable to forward and backward of each layer signature

    y = X[:]

    for layer in self:

      forward_args = layer.forward.__code__.co_varnames

      if 'truth' in forward_args and truth is not None:
        layer.forward(inpt=y[:], truth=truth)

      elif 'network' in forward_args:
        layer.forward(network=self)

      else:
        layer.forward(inpt=y[:])

      y = layer.output[:]

    return y

  def _backward(self, X, trainable=True):
    '''
    BackPropagate the error

    Parameters
    ----------
      X : array-like
        Input data

      trainable : bool (default=True)
        Switch if train or not the model

    Returns
    -------
      self

    Notes
    -----
    .. warning::
      Up to now the trainable variable is useless since the layer
      doesn't take it into account!
    '''

    for i in reversed(range(1, self.num_layers)):

      input = self._net[i - 1].output[:]
      delta = self._net[i - 1].delta[:]

      backward_args = self._net[i].backward.__code__.co_varnames

      if 'inpt' in backward_args:
        self._net[i].backward(inpt=input[:], delta=delta[:])

      elif 'network' in backward_args:
        self._net[i].backward(delta=delta[:], network=self)

      else:
        self._net[i].backward(delta=delta[:])

      if hasattr(self._net[i], 'update'):
        self._net[i].update()

    self._net[0].backward(delta=delta[:])

    return self


  def _get_loss(self):
    '''
    Extract the loss value as the last cost in the network model

    Returns
    -------
      loss : float
        The value of the loss/cost stored in the latest layer
        of the neural network ables to compute a cost function.

    Notes
    -----
    .. note::
      If no layers with the cost evaluation capability is found
      the return value is None.
    '''

    for i in reversed(range(1, self.num_layers)):

      if hasattr(self._net[i], 'cost'):
        return self._net[i].cost

    return None


  @property
  def out_shape(self):
    '''
    Get the output shape
    '''
    return self._net[0].out_shape[1:]

  @property
  def input_shape(self):
    '''
    Get the input shape
    '''
    return (self.w, self.h, self.c)

  @property
  def num_layers(self):
    '''
    Get the number of layers in the model
    '''
    return len(self._net)

  def __getitem__(self, pos):
    '''
    Get the layer element

    Parameters
    ----------
      pos : int
        Index of the layer in the neural network structure
    '''

    if pos < 0 or pos >= self.num_layers:
      raise ValueError('Network model : layer out of range. The model has {:d} layers'.format(self.num_layers))

    return self._net[pos]


if __name__ == '__main__':

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
