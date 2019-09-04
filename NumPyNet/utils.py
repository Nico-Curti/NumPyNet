#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet import activations
from inspect import isclass

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Utilities'

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
