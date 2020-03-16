#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class RNN_layer(object):

  def __init__ (self, outputs, steps, activation=Activations, input_shape=None, weights=None, bias=None, return_sequence=False):
    '''
    Recurrent Neural Network layer. Build a Recurrent fully connected architecture.

    Parameters
    ----------

    outputs : integer, number of outputs of the layer.
    steps   : integer, number of timesteps of the recurrency.
    activation1 : activation function object. First activation function, applied to
      every hidden state
    input_shape : tuple of int, default None. Used for a single layer model.
      "None" is used when the layer is part of a network.
    weights : list of numpy array, default None. List containing two numpy array of weights.
      If None, the init is random.
    bias : list of numpy array, default None. List containing two numpy array of bias.
      If None, the init is random.
    '''

    if isinstance(outputs, int) and outputs > 0:
      self.outputs = outputs
    else :
      raise ValueError('Parameter "outputs" must be an integer and > 0')

    if isinstance(steps, int) and steps > 0:
      self.steps = steps
    else :
      raise ValueError('Parameter "steps" must be an integer and > 0')

    b, w, h, c = input_shape

    self.activation = _check_activation(layer=self, activation_func=activation)
    self.return_sequence = return_sequence

    self.weights = None
    self.recurrent_weights = None
    self.bias = None

    self.weights_update = np.zeros(shape=(w*h*c, self.outputs)) # Why
    self.recurrent_weights_update = np.zeros(shape=(self.outputs, self.outputs))
    self.bias_update = np.zeros(shape=(self.outputs,))

    self.output = None
    self.delta = None
    self.states = None

    self.optimizer = None

  def __str__ (self):
    return 'very good rnn'

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    b, w, h, c = previous_layer.out_shape

    self._out_shape = (b, 1, 1, self.outputs)

    if b < self.steps:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect steps found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.weights = np.random.uniform(low=-1, high=1, size=(w*h*c, self.outputs))
    self.recurrent_weights = np.random.uniform(low=-1, high=1, size=(self.outputs, self.outputs))
    self.bias = np.random.uniform(-1, 1, size=(self.outputs,))

    self.weights_update = np.zeros(self.weights3.shape) # Why
    self.recurrent_weights_update = np.zeros(self.recurrent_weights.shape)
    self.bias_update = np.zeros(self.bias.shape)

    self.output, self.delta = (None, None)
    self.states = None

    return self

  @property
  def out_shape(self):
    return self._out_shape

  def set_weights(self, weights):

    if len(weights) != 3:
      raise ValueError('RNN layer : parameters weights must have 3 set of weights, but has {}'.format(len(weights)))

    self.weights = weights[0]
    self.recurrent_weights = weights[1]
    self.bias = weights[2]

  def _as_Strided(self, inpt, shift=1):
    '''
    Generate a view on the inpt data with shape (batch, steps, features), then
      swap the first two axis to ease the computation.

    Parameters
    ----------
      inpt : numpy array, input data of the layer. The input should be (batch, features),
        so two dimensional.
      shift : integer, default one. shift of the window.

    Returns
    -------
      A view on the input array
    '''

    if len(inpt.shape) != 2:
      raise ValueError('RNN layer : shape of the input for _as_Strided must be two dimensionals but is {}'.format(inpt.shape))

    batch, features  = inpt.shape
    stride0, stride1 = inpt.strides

    shape   = (batch - self.steps*shift, self.steps, features)
    strides = (shift*stride0, stride0, stride1)

    view = np.lib.stride_tricks.as_strided(inpt, shape=shape, strides=strides)

    return np.swapaxes(view, 0, 1)


  def forward (self, inpt, copy=False):
    '''
    Forward of the RNN layer
    '''

    X = self._as_Strided(inpt.reshape(-1, np.prod(inpt.shape[1:])))
    self.output = np.zeros(X.shape[:2] + (self.outputs,))
    self.states = np.zeros(shape=(self.output.shape))

    for i, _input in enumerate(X) :

      if i :
        prev_output = self.states[i-1]
      else :
        prev_output = np.zeros_like(self.states[i])

      op = 'ij, jk -> ik'
      h = np.einsum(op, _input, self.weights, optimize=True) + self.bias
      r = np.einsum(op, prev_output, self.recurrent_weights, optimize=True)

      self.states[i] = self.activation.activate(h + r, copy=copy)

    if not self.return_sequence:
      self.output = self.states[-1, ...]

    else :
      self.output = np.swapaxes(self.states, 0, 1)

    self.output = self.output.reshape(X.shape[1], 1, 1, -1)
    self.delta  = np.zeros_like(self.output)

    return self


  def backward(self, inpt, delta, copy=False):
    '''
    backward of the RNN layer
    '''

    X = self._as_Strided(inpt.reshape(-1, np.prod(inpt.shape[1:])))

    if self.return_sequence:
      self.delta = self.delta.reshape(self.steps, -1, self.outputs)
    else:
      self.delta.reshape(-1, self.outputs)

    delta = delta.reshape(X.shape)

    _delta = np.zeros(shape=(X.shape[1], self.outputs))

    for i, _input in reversed(list(enumerate(X))):

      if self.return_sequence:
        _delta += self.delta[i]
        _delta *= self.activation.gradient(self.states[i], copy=copy)

      elif i == X.shape[0]-1:
        _delta += self.delta
        _delta *= self.activation.gradient(self.states[i], copy=copy)

      else :
        _delta *= self.activation.gradient(self.states[i], copy=copy)

      if i :
        _prev_output = self.states[i-1]
      else :
        _prev_output = np.zeros_like(self.states[i])

      self.bias_update += _delta.sum(axis=0)

      op = 'ij, ik -> kj'
      self.weights_update += np.einsum(op, _delta, _input, optimize=True)
      self.recurrent_weights_update += np.einsum(op, _delta, _prev_output, optimize=True)

      _delta   = np.einsum('ij, jk -> ik', _delta, self.recurrent_weights, optimize=True) # passed back in timesteps
      delta_view[i] = np.einsum('ij, kj -> ik', _delta, self.weights, optimize=True) # delta to be backprop.

    return self


  def update (self):
    '''
    update function for the rnn layer. optimizer must be assigned
      externally as an optimizer object.
    '''
    check_is_fitted(self, 'delta')

    self.bias, self.weights, self.recurrent_weights = \
                        self.optimizer.update(params    = [self.bias, self.weights, self.recurrent_weights],
                                              gradients = [self.bias_update, self.weights_update, self.recurrent_weights_update]
                                              )
    return self

if __name__ == '__main__':
  pass
