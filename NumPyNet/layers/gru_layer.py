#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Logistic
from NumPyNet.activations import Tanh
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class GRU_layer (object):

  def __init__(self, outputs, steps, input_shape=None, weights=None, bias=None):

    if isinstance(outputs, int) and outputs > 0:
      self.outputs = outputs
    else:
      raise ValueError('GRU layer : Parameter "outputs" must be an integer and > 0')

    if isinstance(steps, int) and steps > 0:
      self.steps = steps
    else:
      raise ValueError('GRU layer : Parameter "steps" must be an integer and > 0')

    self.input_shape = input_shape

    self.Wz = weights[0]  # shape (inputs, outputs)
    self.Wr = weights[1]
    self.Wh = weights[2]

    self.Uz = weights[3]  # shape (outputs, outputs)
    self.Ur = weights[4]
    self.Uh = weights[5]

    self.bz = bias[0]  # shape (outputs, )
    self.br = bias[1]
    self.bh = bias[2]

  def __call__(self, prev_layer):
    raise NotImplementedError

  @property
  def out_shape(self):
    pass

  def _as_Strided(self, arr, shift=1):
    '''
    Stride the input array to return mini-sequences
    '''

    X = arr.reshape(arr.shape[0], -1)

    Npoints, features = X.shape
    stride0, stride1 = X.strides

    shape = (Npoints - self.steps * shift, self.steps, features)
    strides = (shift * stride0, stride0, stride1)

    X = np.lib.stride_tricks.as_strided(arr, shape=shape, strides=strides)

    return np.swapaxes(X, 0, 1)

  def forward(self, inpt):

    inpt = inpt.astype('float64')
    _input = self._as_Strided(inpt)
    state = np.zeros(shape=(_input.shape[1], self.outputs))

    self.output = np.zeros_like(state)

    for i, X in enumerate(_input):

      op = 'ij, jk -> ik'
      xz = np.einsum(op, X, self.Wz) + self.bz
      xr = np.einsum(op, X, self.Wr) + self.br
      xh = np.einsum(op, X, self.Wh) + self.bh

      hz = np.einsum(op, state, self.Uz)
      hr = np.einsum(op, state, self.Ur)

      zt = Logistic.activate(xz + hz)
      rt = Logistic.activate(xr + hr)

      hh = np.einsum(op, state * rt, self.Uh)

      state = zt * state + (1 - zt) * Tanh.activate(xh + hh)

    # implementation "no sequence"
    self.output = state

    self.delta = np.zeros_like(self.output)

  def backward(self, delta):
    pass

  def update(self):
    pass


if __name__ == '__main__':
  pass
