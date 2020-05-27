#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted

from NumPyNet.layers.base import BaseLayer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class SimpleRNN_layer(BaseLayer):

  def __init__ (self, outputs, steps, activation=Activations, input_shape=None, weights=None, recurrent_weights=None, bias=None, return_sequence=False):
    '''
    Recurrent Neural Network layer. Build a Recurrent fully connected architecture.

    Parameters
    ----------
    outputs : integer, number of outputs of the layer.
    steps   : integer, number of timesteps of the recurrency.
    activation : activation function object. activation function, applied to every hidden state
    input_shape : tuple of int, default None. Used for a single layer model.
      "None" is used when the layer is part of a network.
    weights : numpy array, default None. numpy array of weights
      of shapes (inputs, outputs). If None, the init is random.
    recurrent_weights : numpy array, default None. numpy array of weights
      of shapes (outputs, outputs). If None, the init is random.
    bias : numpy array, default None. shape (outputs, )
      If None, the init is zeros.
    '''

    if isinstance(outputs, int) and outputs > 0:
      self.outputs = outputs
    else :
      raise ValueError('Parameter "outputs" must be an integer and > 0')

    if isinstance(steps, int) and steps > 0:
      self.steps = steps
    else :
      raise ValueError('Parameter "steps" must be an integer and > 0')

    self.activation = _check_activation(layer=self, activation_func=activation)
    self.return_sequence = return_sequence

    self.weights = weights
    self.recurrent_weights = recurrent_weights
    self.bias = bias

    self.weights_update = None
    self.recurrent_weights_update = None
    self.bias_update = None
    self.states      = None
    self.optimizer   = None

    if input_shape is not None:
      super(SimpleRNN_layer, self).__init__(input_shape=input_shape)
      self._build()

  def _build(self):

    b, w, h, c = self.input_shape

    if self.weights is None:
      scale = np.sqrt(2 / (w * h * c * self.outputs))
      self.weights = np.random.normal(loc=scale, scale=1., size=(w * h * c, self.outputs))

    if self.recurrent_weights is None:
      scale = np.sqrt(2 / (self.outputs * self.outputs))
      self.recurrent_weights = np.random.normal(loc=scale, scale=1., size=(self.outputs, self.outputs))

    if self.bias is None:
      self.bias = np.zeros(shape=(self.outputs, ), dtype=float)

    if self.return_sequence:
      self._out_shape = (b - self.steps, 1, 1, self.outputs * self.steps)
    else :
      self._out_shape = (b - self.steps, 1, 1, self.outputs)

    # init to zeros because of += in backward
    self.weights_update = np.zeros_like(self.weights)
    self.recurrent_weights_update = np.zeros_like(self.recurrent_weights)
    self.bias_update = np.zeros_like(self.bias)


  def __str__ (self):
    batch, w, h, c = self.input_shape
    out_b, out_w, out_h, out_c = self.out_shape
    return 'srnn                   {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {4:>4d} x{5:>4d} x{6:>4d} x{7:>4d}'.format(
           batch, w, h, c, out_b, out_w, out_h, out_c)

  def __call__(self, previous_layer):

    super(SimpleRNN_layer, self).__call__(previous_layer)
    self._build()

    return self

  @property
  def out_shape(self):
    return self._out_shape

  def load_weights(self, chunck_weights, pos=0):
    '''
    Load weights from full array of model weights

    Parameters
    ----------
      chunck_weights : numpy array of model weights
      pos : current position of the array

    Returns
    ----------
      pos
    '''

    w, h, c = self.input_shape[1:]
    self.bias = chunck_weights[pos : pos + self.outputs]
    pos += self.outputs

    self.weights = chunck_weights[pos : pos + self.weights.size]
    self.weights = self.weights.reshape(w*h*c, self.outputs)
    pos += self.weights.size

    self.recurrent_weights = chunck_weights[pos : pos + self.recurrent_weights.size]
    self.recurrent_weights = self.recurrent_weights.reshape(self.outputs, self.outputs)
    pos += self.recurrent_weights.size

    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.bias.ravel(), self.weights.ravel(), self.recurrent_weights.ravel()], axis=0).tolist()

  def _asStride(self, inpt, shift=1):
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
      A view on the input array of shape (steps, batch, features)
    '''

    if len(inpt.shape) != 2:
      raise ValueError('RNN layer : shape of the input for _asStrid must be two dimensionals but is {}'.format(inpt.shape))

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

    inpt = inpt.astype('float64')
    self.X = self._asStride(inpt.reshape(-1, np.prod(inpt.shape[1:])))
    self.states = np.zeros(shape=(self.steps, self.X.shape[1], self.outputs))

    for i, _input in enumerate(self.X) :

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

    self.output = self.output.reshape(self.X.shape[1], 1, 1, -1)
    self.delta  = np.zeros_like(self.output)

    return self


  def backward(self, inpt, delta, copy=False):
    '''
    backward of the RNN layer
    '''

    # if self.return_sequence:
    #   self.delta = self.delta.reshape(self.steps, -1, self.outputs)
    # else:
    #   self.delta.reshape(-1, self.outputs)

    delta_view = np.swapaxes(delta, 0, 1)

    self.delta *= self.activation.gradient(self.output, copy=copy)

    _delta_state = self.delta.reshape(-1, self.outputs)

    for i, _input in reversed(list(enumerate(self.X))):

      # if self.return_sequence:
      #   _delta_state += self.delta[i]
      #   _delta_state *= self.activation.gradient(self.states[i], copy=copy)

      # elif i == X.shape[0]-1:
      #   _delta_state += self.delta
      #   _delta_state *= self.activation.gradient(self.states[i], copy=copy)
      #
      # else :
      #   _delta_state *= self.activation.gradient(self.states[i], copy=copy)

      if i :
        _prev_output = self.states[i-1]
      else :
        _prev_output = np.zeros_like(self.states[i])

      self.bias_update += _delta_state.sum(axis=0)

      op = 'ij, ik -> kj'
      self.weights_update += np.einsum(op, _delta_state, _input, optimize=True)
      self.recurrent_weights_update += np.einsum(op, _delta_state, _prev_output, optimize=True)

      delta_view[i, ...] += np.einsum('ij, kj -> ik', _delta_state, self.weights, optimize=True)    # delta to be backprop.
      _delta_state = np.einsum('ij, kj -> ik', _delta_state, self.recurrent_weights, optimize=True) # passed back in timesteps
      _delta_state *= self.activation.gradient(_prev_output, copy=copy)


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
