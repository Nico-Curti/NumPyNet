#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.exception import LayerError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class BaseLayer (object):

  '''
  Abstract Base Layer object

  Parameters
  ----------
    input_shape : tuple (default=None)
      Input layer dimension
  '''

  def __init__(self, input_shape=None):
    self.input_shape = input_shape
    self.output, self.delta = (None, None)

  def __str__(self):
    '''
    Print the layer
    '''
    raise NotImplementedError

  def __repr__(self):
    '''
    Object representation
    '''
    class_name = self.__class__.__qualname__
    params = self.__init__.__code__.co_varnames
    params = set(params) - {'self'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)

  def _build(self, *args, **kwargs):
    '''
    Build layer parameters
    '''
    pass

  def _check_dims(self, shape, arr, func):
    '''
    Check shape array
    '''
    if shape[1:] != arr.shape[1:]:
      class_name = self.__class__.__name__
      raise ValueError('{0} {1}. Incorrect input shape. Expected {2} and given {3}'.format(func, class_name, shape[1:], arr.shape[1:]))

  def __call__(self, previous_layer):
    '''
    Overload operator ()

    Parameters
    ----------
      previous_layer : Layer obj
        Layer object to join
    '''

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = previous_layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {0} cannot be connected to the previous {1} layer.'.format(class_name, prev_name))

    self.input_shape = previous_layer.out_shape

    self._build()

    return self

  @property
  def out_shape(self):
    '''
    Get the output shape

    Returns
    -------
      out_shape : tuple
        Tuple as (batch, out_w, out_h, out_c)
    '''
    return self.input_shape

  def forward(self, input, *args, **kwargs):
    '''
    Abstract Forward function

    Parameters
    ----------
      input : array-like
        Input array of data to process

      *args : list
        Optional arguments

      **kwargs : dict
        Optional arguments
    '''
    raise NotImplementedError

  def backward(self, delta, input=None, *args, **kwargs):
    '''
    Abstract Backward function

    Parameters
    ----------
      delta : array-like
        Gradient array to back-propagate

      input : array-like (default=None)
        Input array of data to process

      *args : list
        Optional arguments

      **kwargs : dict
        Optional arguments
    '''
    raise NotImplementedError
