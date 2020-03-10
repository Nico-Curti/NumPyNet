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

  def __init__ (self, outputs, steps, activation1=Activations, activation2=Activations, input_shape=None, weights=None, bias=None):
    '''
    Recurrent Neural Network layer. Build a Recurrent fully connected architecture.

    Parameters
    ----------

    outputs : integer, number of outputs of the layer.
    steps   : integer, number of timesteps of the recurrency.
    activation1 : activation function object. First activation function, applied to
      every hidden state
    activation2 : activation function object. Second activation function, applied to
      the output of every timestep.
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

    self.activation1 = _check_activation(layer=self, activation_func=activation1)
    self.activation2 = _check_activation(layer=self, activation_func=activation2)

    self.weights1 = None
    self.weights2 = None
    self.weights3 = None

    self.bias  = None

    self.weights_update1 = None
    self.weights_update2 = None
    self.weights_update3 = None
    self.bias_update1 = None
    self.bias_update2 = None

    self.output = None
    self.delta = None

    self.optimizer = None

  def __str__ (self):
    return 'very good rnn '

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

    self.weights1 = np.random.uniform(-1, 1, size=(w*h*c, self.outputs))         # Wxh
    self.weights2 = np.random.uniform(-1, 1, size=(self.outputs, self.outputs))  # Whh
    self.weights3 = np.random.uniform(-1, 1, size=(self.outputs, self.outputs))  # Why

    self.bias1  = np.random.uniform(-1, 1, size=(self.outputs,))
    self.bias2  = np.random.uniform(-1, 1, size=(self.outputs,))

    self.weights_update3 = np.zeros(self.weights3.shape) # Why
    self.weights_update2 = np.zeros(self.weights2.shape) # Whh
    self.weights_update1 = np.zeros(self.weights1.shape) # Wxh
    self.bias_update1 = np.zeros(self.bias1.shape)
    self.bias_update2 = np.zeros(self.bias2.shape)

    self.batch = b // self.steps
    self.input_shape = (self.batch, w, h, c)
    indices = np.arange(0, b)
    self.batches = np.lib.stride_tricks.as_strided(indices, shape=(self.steps, self.batch), strides=(self.batch * 8, 8)).copy()
    self.output, self.delta = (None, None)

    return self

  @property
  def out_shape(self):
    return self._out_shape

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

    shape   = (batch - steps*shift + 1, self.steps, features)
    strides = (shift*stride0, stride0, stride1)

    view = np.lib.stride_tricks.as_strided(inpt, shape=shape, strides=strides)

    return np.swapaxes(view, 0, 1)


  def forward (self, inpt, copy=False):
    '''
    Forward of the RNN layer
    '''

    X = inpt.reshape(inpt.shape[0], -1)

    _state = np.zeros(shape=(self.batch, self.outputs))
    self.states = np.empty(shape=(X.shape[0], self.outputs))
    self.output = np.empty(shape=(X.shape[0], self.outputs))   # (steps, outputs)

    for idx in self.batches:

      # final shape (steps, inputs)*(inputs, hidden) -> (steps, hidden)
      out1 = np.einsum('ij, jk -> ik', X[idx, ...], self.weights1, optimize=True)

      # final shape (steps, hidden)*(hidden, hidden) -> (steps, hidden)
      out2 = np.einsum('ij, jk -> ik', _state, self.weights2, optimize=True)

      # update of state at time T -> (steps, hidden)
      _state = self.activation1.activate(out1 + out2 + self.bias1, copy=copy)

      self.states[idx, ...] = _state.copy()

      # final output of the timestep T -> (steps, hidden)
      outT = np.einsum('ij, jk -> ik', _state, self.weights3) + self.bias2
      self.output[idx, ...] = self.activation2.activate(outT, copy=copy)

    self.output = self.output.reshape(-1, 1, 1, self.outputs)
    self.delta  = np.zeros(shape=self.output.shape)

    return self


  def backward(self, inpt, delta, copy=False):
    '''
    backward of the RNN layer
    '''

    X = inpt.reshape(inpt.shape[0], -1)

    delta_r = delta.reshape(X.shape)

    # gradient of the second activation function
    self.delta *= self.activation2.gradient(self.output, copy=copy)
    self.delta  = self.delta.reshape(-1, self.outputs)

    _delta_state = 0.

    for i, idx in reversed(list(enumerate(self.batches))):

      if i > 0:
        _prev_state = self.states[self.batches[i-1]].copy()
      else :
        _prev_state = np.zeros(shape=self.states[idx,...].shape)

      self.bias_update2    += self.delta[idx, ...].sum(axis=0)
      self.weights_update3 += np.einsum('ij, ik -> jk', self.delta[idx, ...], self.states[idx, ...], optimize=True)

      dh = np.einsum('ij, kj -> ki', self.weights3, self.delta[idx, ...], optimize=True) + _delta_state
      dh *= self.activation1.gradient(dh, copy=copy)

      self.bias_update1    += dh.sum(axis=0)
      self.weights_update1 += np.einsum('ij, ik -> kj', dh, X[idx, ...], optimize=True)
      self.weights_update2 += np.einsum('ij, ik -> jk', dh, _prev_state, optimize=True)

      _delta_state = np.einsum('ij, ki -> kj', self.weights2, dh, optimize=True)

      # TODO: don't know if this is correct
      delta_r[idx, ...] = np.einsum('ij, kj -> ik', dh, self.weights1, optimize=True)


  def update (self):
    '''
    update function for the rnn layer. optimizer must be assigned
      externally as an optimizer object.
    '''
    check_is_fitted(self, 'delta')

    self.bias1, self.bias2, self.weights1, self.weights2, self.weights3 = \
                        self.optimizer.update(params   = [self.bias1, self.bias2, self.weights1, self.weights2, self.weights3],
                                              gradients= [self.bias_update1, self.bias_update2,
                                                          self.weights_update1, self.weights_update2, self.weights_update3]
                                              )
    return self
