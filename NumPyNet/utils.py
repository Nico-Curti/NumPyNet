#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import sys
import numpy as np
from enum import Enum
from io import StringIO
from inspect import isclass
from contextlib import contextmanager

from NumPyNet import activations
from NumPyNet.exception import NotFittedError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

# Enum of cost_function, declarations inside class
class cost_type(int, Enum):
  mse = 0 # mean square error
  masked = 1
  mae = 2 # mean absolute error
  seg = 3
  smooth = 4
  wgan = 5
  hellinger = 6
  hinge = 7
  logcosh = 8

def _check_activation (layer, activation_func):
  '''
  Check if the activation function is valid.

  Parameters
  ----------
    layer : object
      Layer object (ex. Activation_layer)

    activation_func : string or Activations object
      activation function to check. If the Activations object is not created yet
      the 'eval' is done on the object.

  Returns
  -------
    Byron activation function index

  Notes
  -----
  You can use this function to verify if the given activation function is valid.
  The function can be passed either as a string either as object or simply as class object.

  Examples
  --------

  >>> layer = Activation_layer(input_shape=(1,2,3))
  >>> print(_check_activation(layer, 'Linear'))
      6
  >>> print(_check_activation(layer, Activations.Relu))
      6
  >>> print(_check_activation(layer, Activations.Linear()))
      6
  '''

  if isinstance(activation_func, str):
    allowed_activation_func = [f.lower() for f in dir(activations) if isclass(getattr(activations, f)) and f != 'Activations']

    if activation_func.lower() not in allowed_activation_func:
      class_name = layer.__class__.__name__
      raise ValueError('{0}: incorrect value of Activation Function given'.format(class_name))
    else:
      activation_func = activation_func.lower()
      activation_func = ''.join([activation_func[0].upper(), activation_func[1:]])

    activation_func = ''.join(['activations.', activation_func, '()'])
    activation = eval(activation_func)

  elif issubclass(type(activation_func), activations.Activations):
    activation = activation_func

  elif issubclass(activation_func, activations.Activations):
    activation = activation_func

  else:
    class_name = layer.__class__.__name__
    raise ValueError('{0}: incorrect value of Activation Function given'.format(class_name))

  return activation


def _check_cost (layer, cost):
  '''
  Check if the cost function is valid.

  Parameters
  ----------
    layer : object
      Layer object (ex. Cost_layer)

    cost : string or Cost object
      cost function to check. The cost object can be use by the cost enum

  Returns
  -------
    NumPyNet cost function index

  Notes
  -----
  You can use this function to verify if the given cost function is valid.
  The function can be passed either as a string either as object.

  Examples
  --------

  >>> layer = Cost_layer(input_shape=(1,2,3))
  >>> print(_check_cost(layer, 'mae'))
      2
  >>> print(_check_cost(layer, cost.mae))
      2
  '''
  if isinstance(cost, str):
    allowed_cost = [c for c in dir(cost_type) if not c.startswith('__')]

    if cost.lower() not in allowed_cost:
      class_name = layer.__class__.__name__
      raise ValueError('{0}: incorrect value of Cost Function given'.format(class_name))

    else:
      cost = eval('cost_type.{0}.value'.format(cost))

  elif isinstance(cost, cost_type):
    cost = cost.value

  elif isinstance(cost, int) and cost <= max(cost_type):
    cost = cost_type(cost)

  else:
    class_name = layer.__class__.__name__
    raise ValueError('{0}: incorrect value of Cost Function given'.format(class_name))

  return cost


def check_is_fitted (obj, variable='delta'):
  '''
  Check if for the current layer is available the backward function.

  Parameters
  ----------
    obj : layer type
      The object used as self

    variable : str
      The variable name which allows the backward status if it is not None

  Note
  ----
    The backward function can be used ONLY after the forward procedure.
    This function allows to check if the forward function has been already applied.
  '''

  fitted_var = getattr(obj, variable)

  if fitted_var is None:
    raise NotFittedError('This layer instance is not fitted yet. Call "forward" with appropriate arguments before using its backward function.')

  else:
    return True


def print_statistics (arr):
  '''
  Compute the common statistics of the input array

  Parameters
  ----------
    arr : array-like

  Returns
  -------
    mse : Mean Squared Error, i.e sqrt(mean(x*x))
    mean: Mean of the array
    variance: Variance of the array

  Notes
  -----
  The value are printed and returned
  '''

  mean = np.mean(arr)
  variance = np.var(arr)
  mse = np.sqrt(np.mean(arr * arr))

  print('MSE: {:>3.3f}, Mean: {:>3.3f}, Variance: {:>3.3f}'.format(mse, mean, variance))

  return (mse, mean, variance)


def to_categorical (arr):
  '''
  Converts a vector of labels into one-hot encoding format

  Parameters
  ----------
    arr : array-like 1D
      array of integer labels (without holes)

  Returns
  -------
  2D matrix in one-hot encoding format
  '''

  n = len(arr)
  pos = np.expand_dims(arr, axis=1).astype(int)
  num_label = np.max(pos) + 1

  categorical = np.zeros(shape=(n, num_label), dtype=float)
  np.put_along_axis(categorical, indices=pos, values=1, axis=1)

  return categorical


def from_categorical (categoricals):
  '''
  Convert a one-hot encoding format into a vector of labels

  Parameters
  ----------
    categoricals : array-like 2D
      one-hot encoding format of a label set

  Returns
  -------
  Corresponding labels in 1D array
  '''

  return np.argmax(categoricals, axis=-1)

def data_to_timesteps (data, steps, shift=1):
  '''
  Prepare data for a Recurrent model, dividing a series of data with shape (Ndata, features)
   into timesteps, with shapes (Ndata - steps + 1, steps, features)
   If 'data' has more than two dimension, it'll be reshaped.
   Pay attention to the final number of 'batch'

  Parameters
  ----------
  data : two or 4 dimensional numpy array, with shapes (Ndata, features) or (Ndata, w, h, c).
  steps : integer, number of timesteps considered for the Recurrent layer
  shift : integer, default is 1. TODO

  Returns
  -------
   X, a view on the data array of input, for Recurrent layers
  '''

  X = data.reshape(data.shape[0], -1)

  Npoints, features = X.shape
  stride0, stride1  = X.strides

  shape   = (Npoints - steps*shift, steps, features)
  strides = (shift*stride0, stride0, stride1)

  X = np.lib.stride_tricks.as_strided(data, shape=shape, strides=strides)
  y = data[steps:]

  return X, y

@contextmanager
def _redirect_stdout (verbose):
  '''
  Redirect output stdout from cython wrap to devnull or not.
  This function does not work for cython stdout!
  If you want something for cython wraps you can refer to the
  implementation in the rFBP package (https://github.com/Nico-Curti/rFBP)

  Parameters
  ----------
    verbose : bool
      Enable or disable stdout
  '''

  if verbose:
    try:
      yield
    finally:
      return

  old_target = sys.stdout
  try:
    with open(os.devnull, "w") as new_target:
      sys.stdout = new_target
      yield new_target
  finally:
    sys.stdout = old_target
