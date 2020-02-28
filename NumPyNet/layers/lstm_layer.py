#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.layers import Connected_layer
from NumPyNet.activations import Logistic
from NumPyNet.activations import Tanh
from NumPyNet.utils import check_is_fitted

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class LSTM_layer(object):

  def __init__(self, outputs, steps, input_shape=None, weights=None, bias=None, **kwargs):
    '''
    LSTM layer

    Parameters
    ----------
      outputs     : integer, number of outputs of the layers
      input_shape : tuple, default None. Shape of the input in the format (batch, w, h, c),
                    None is used when the layer is part of a Network model.
      weights     : array of shape (w * h * c, outputs), default is None. Weights of the dense layer.
                    If None, weights init is random.
      bias        : array of shape (outputs, ), default None. Bias of the connected layer.
                    If None, bias init is random
    '''

    if isinstance(outputs, int) and outputs > 0:
      self.outputs = outputs
    else :
      raise ValueError('Parameter "outputs" must be an integer and > 0')

    if isinstance(steps, int) and steps > 0:
      self.steps = steps
    else :
      raise ValueError('Parameter "steps" must be an integer and > 0')

    if weights is None:
      weights = (None, None, None)
    else:
      if np.shape(weights)[0] != 3:
        raise ValueError('Wrong number of init "weights". There are 3 connected layers into the RNN cell.')

    if bias is None:
      bias = (None, None, None)
    else:
      if np.shape(bias)[0] != 3:
        raise ValueError('Wrong number of init "biases". There are 3 connected layers into the RNN cell.')

    # if input shape is passed, init of weights, else done in  __call__
    if input_shape is not None:
      self.input_shape = input_shape

      self.batch = self.input_shape[0] // self.steps
      self.batches = np.array_split(range(self.input_shape[0]), indices_or_sections=self.batch)

      _, w, h, c = self.input_shape
      initial_shape = (self.batch, w, h, c)

    else:
      self.input_shape = None

      # TODO: remember to overwrite this layer when the input_shape is known!
      self.batch = None
      self.batches = None
      initial_shape = None

    self.uf = Connected_layer(outputs, activation='Linear', input_shape=initial_shape, weights=weights[0], bias=bias[0])
    self.ui = Connected_layer(outputs, activation='Linear', input_shape=initial_shape, weights=weights[0], bias=bias[0])
    self.ug = Connected_layer(outputs, activation='Linear', input_shape=initial_shape, weights=weights[0], bias=bias[0])
    self.uo = Connected_layer(outputs, activation='Linear', input_shape=initial_shape, weights=weights[0], bias=bias[0])

    self.wf = Connected_layer(outputs, activation='Linear', input_shape=(self.batch, 1, 1, self.outputs), weights=weights[1], bias=bias[1])
    self.wi = Connected_layer(outputs, activation='Linear', input_shape=(self.batch, 1, 1, self.outputs), weights=weights[1], bias=bias[1])
    self.wg = Connected_layer(outputs, activation='Linear', input_shape=(self.batch, 1, 1, self.outputs), weights=weights[1], bias=bias[1])
    self.wo = Connected_layer(outputs, activation='Linear', input_shape=(self.batch, 1, 1, self.outputs), weights=weights[1], bias=bias[1])

    if input_shape is not None:
      self.uf.input_shape  = (self.input_shape[0], w, h, c)
      self.ui.input_shape  = (self.input_shape[0], w, h, c)
      self.ug.input_shape  = (self.input_shape[0], w, h, c)
      self.uo.input_shape  = (self.input_shape[0], w, h, c)

      self.wf.input_shape   = (self.input_shape[0], w, h, self.outputs)
      self.wi.input_shape   = (self.input_shape[0], w, h, self.outputs)
      self.wg.input_shape   = (self.input_shape[0], w, h, self.outputs)
      self.wo.input_shape   = (self.input_shape[0], w, h, self.outputs)

    self.output = np.empty(shape=self.uf.out_shape, dtype=float)
    self.cell = np.empty(shape=self.uf.out_shape, dtype=float)
    self.delta = None
    self.optimizer = None

  def __str__(self):
    return '\n\t'.join(('LSTM Layer: {:d} inputs, {:d} outputs'.format(self.inputs, self.outputs),
                        '{}'.format(self.uf),
                        '{}'.format(self.ui),
                        '{}'.format(self.ug),
                        '{}'.format(self.uo),
                        '{}'.format(self.wf),
                        '{}'.format(self.wi),
                        '{}'.format(self.wg),
                        '{}'.format(self.wo)
                        ))

  def __call__(self, previous_layer):
    raise NotImplementedError('Not yet supported')
    # return self

  @property
  def inputs(self):
    return np.prod(self.input_shape[1:])

  @property
  def out_shape(self):
    return self.wo.out_shape

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

    pos = self.uf.load_weights(chunck_weights, pos=pos)
    pos = self.ui.load_weights(chunck_weights, pos=pos)
    pos = self.ug.load_weights(chunck_weights, pos=pos)
    pos = self.uo.load_weights(chunck_weights, pos=pos)
    pos = self.wf.load_weights(chunck_weights, pos=pos)
    pos = self.wi.load_weights(chunck_weights, pos=pos)
    pos = self.wg.load_weights(chunck_weights, pos=pos)
    pos = self.wo.load_weights(chunck_weights, pos=pos)
    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.uf.bias.ravel(), self.uf.weights.ravel(),
                           self.ui.bias.ravel(), self.ui.weights.ravel(),
                           self.ug.bias.ravel(), self.ug.weights.ravel(),
                           self.uo.bias.ravel(), self.uo.weights.ravel(),
                           self.wf.bias.ravel(), self.wf.weights.ravel(),
                           self.wi.bias.ravel(), self.wi.weights.ravel(),
                           self.wg.bias.ravel(), self.wg.weights.ravel(),
                           self.wo.bias.ravel(), self.wo.weights.ravel()], axis=0).tolist()

  def _internal_forward(self, layer, inpt, indices, copy=False):

    _input = inpt[indices, ...]
    _input = _input.reshape(_input.shape[0], -1)

    z = np.einsum('ij, jk -> ik', _input, layer.weights, optimize=True) + layer.bias
    layer.output[indices, ...] = layer.activation(z, copy=copy).reshape(-1, 1, 1, layer.outputs)


  def forward(self, inpt, copy=False):
    '''
    Forward function of the LSTM layer. It computes the matrix product
      between inpt and weights, add bias and activate the result with the
      chosen activation function.

    Parameters
    ----------
      inpt : numpy array with shape (batch, w, h, c). Input batch of images of the layer
      shortcut : boolean, default False. Enable/Disable internal shortcut connection.

    Returns
    ----------
    LSTM_layer object
    '''

    self.uf.output = np.empty(shape=self.uf.out_shape, dtype=float)
    self.ui.output = np.empty(shape=self.ui.out_shape, dtype=float)
    self.ug.output = np.empty(shape=self.ug.out_shape, dtype=float)
    self.uo.output = np.empty(shape=self.uo.out_shape, dtype=float)
    self.wf.output = np.empty(shape=self.wf.out_shape, dtype=float)
    self.wi.output = np.empty(shape=self.wi.out_shape, dtype=float)
    self.wg.output = np.empty(shape=self.wg.out_shape, dtype=float)
    self.wo.output = np.empty(shape=self.wo.out_shape, dtype=float)

    h = np.zeros(shape=self.out_shape, dtype=float)

    for idx in self.batches:

      h_slice = h[idx, ...]

      self._internal_forward(layer=self.wf, inpt=h, indices=idx, copy=copy)
      self._internal_forward(layer=self.wi, inpt=h, indices=idx, copy=copy)
      self._internal_forward(layer=self.wg, inpt=h, indices=idx, copy=copy)
      self._internal_forward(layer=self.wo, inpt=h, indices=idx, copy=copy)

      self._internal_forward(layer=self.uf, inpt=inpt, indices=idx, copy=copy)
      self._internal_forward(layer=self.ui, inpt=inpt, indices=idx, copy=copy)
      self._internal_forward(layer=self.ug, inpt=inpt, indices=idx, copy=copy)
      self._internal_forward(layer=self.uo, inpt=inpt, indices=idx, copy=copy)

      f = Logistic.activate(self.wf.output[idx, ...] + self.uf.output[idx, ...])
      i = Logistic.activate(self.wi.output[idx, ...] + self.ui.output[idx, ...])
      g = Tanh.activate(self.wg.output[idx, ...] + self.ug.output[idx, ...])
      o = Logistic.activate(self.wo.output[idx, ...] + self.uo.output[idx, ...])

      c  = i * g
      c *= f
      h_slice = Tanh.activate(c) * o
      self.cell[idx, ...] = c
      self.output[idx, ...] = h_slice

    self.uf.delta = np.zeros(shape=self.uf.out_shape, dtype=float)
    self.ui.delta = np.zeros(shape=self.ui.out_shape, dtype=float)
    self.ug.delta = np.zeros(shape=self.ug.out_shape, dtype=float)
    self.uo.delta = np.zeros(shape=self.uo.out_shape, dtype=float)
    self.wf.delta = np.zeros(shape=self.wf.out_shape, dtype=float)
    self.wi.delta = np.zeros(shape=self.wi.out_shape, dtype=float)
    self.wg.delta = np.zeros(shape=self.wg.out_shape, dtype=float)
    self.wo.delta = np.zeros(shape=self.wo.out_shape, dtype=float)

    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def _internal_backward(self, layer, inpt, indices, delta=None, copy=False):

    _input = inpt[indices, ...]
    _input = _input.reshape(_input.shape[0], -1)

    _delta = layer.delta[indices, ...]

    _delta *= layer.gradient(layer.output[indices, ...], copy=copy)
    _delta_r = _delta.reshape(-1, layer.outputs)

    layer.bias_update = _delta_r.sum(axis=0)   # shape : (outputs,)

    layer.weights_update = np.dot(_input.transpose(), _delta_r)

    if delta is not None:

      delta_view = delta[indices, ...]
      delta_shaped = delta_view.reshape(len(indices), -1)

      # shapes : (batch , w * h * c) = (batch , w * h * c) + (batch, outputs) @ (outputs, w * h * c)

      # delta_shaped[:] += self.delta @ self.weights.transpose()')  # I can modify delta using its view
      delta_shaped[:] += np.einsum('ij, kj -> ik', _delta_r, layer.weights, optimize=True)

      return delta_shaped.reshape(delta_view.shape)

  def backward(self, inpt, delta=None, copy=False):
    '''
    Backward function of the LSTM layer, updates the global delta of the
      network to be Backpropagated, he weights upadtes and the biases updates

    Parameters
    ----------
      inpt     : original input of the layer
      delta    : global delta, to be backpropagated.

    Returns
    ----------
    LSTM_layer object
    '''

    check_is_fitted(self, 'delta')

    dh = np.zeros(shape=self.out_shape, dtype=float)
    prev_cell = None
    prev_state = None

    for _i, idx in reversed(list(enumerate(self.batches))):

      prev_cell  = self.cell[idx, ...] if _i != 0 else prev_cell
      c          = self.cell[idx, ...]
      prev_state = self.output[idx, ...] if _i != 0 else prev_state

      h = self.output[idx, ...]
      f = Logistic.activate(self.wf.output[idx, ...] + self.uf.output[idx, ...])
      i = Logistic.activate(self.wi.output[idx, ...] + self.ui.output[idx, ...])
      g = Tanh.activate(self.wg.output[idx, ...] + self.ug.output[idx, ...])
      o = Logistic.activate(self.wo.output[idx, ...] + self.uo.output[idx, ...])

      temp1  = Tanh.activate(c)
      temp2  = self.delta[idx, ...] * o * Tanh.gradient(temp1)
      temp1 *= self.delta[idx, ...] * Logistic.gradient(o)
      self.wo.delta[idx, ...] = temp1
      self.uo.delta[idx, ...] = temp1

      temp1 = temp2 * i * Tanh.gradient(g)
      self.wg.delta[idx, ...] = temp1
      self.ug.delta[idx, ...] = temp1

      temp1 = temp2 * g * Logistic.gradient(i)
      self.wi.delta[idx, ...] = temp1
      self.ui.delta[idx, ...] = temp1

      temp1 = temp2 * prev_cell * Logistic.gradient(f)
      self.wf.delta[idx, ...] = temp1
      self.uf.delta[idx, ...] = temp1

      dc = temp2 * f

      dh[idx, ...]    = self._internal_backward(self.wo, inpt=prev_state, indices=range(prev_state.shape[0]), delta=dh[idx, ...], copy=copy)
      delta[idx, ...] = self._internal_backward(self.uo, inpt=inpt, indices=idx, delta=delta, copy=copy)

      dh[idx, ...]    = self._internal_backward(self.wg, inpt=prev_state, indices=range(prev_state.shape[0]), delta=dh[idx, ...], copy=copy)
      delta[idx, ...] = self._internal_backward(self.ug, inpt=inpt, indices=idx, delta=delta, copy=copy)

      dh[idx, ...]    = self._internal_backward(self.wi, inpt=prev_state, indices=range(prev_state.shape[0]), delta=dh[idx, ...], copy=copy)
      delta[idx, ...] = self._internal_backward(self.ui, inpt=inpt, indices=idx, delta=delta, copy=copy)

      dh[idx, ...]    = self._internal_backward(self.wf, inpt=prev_state, indices=range(prev_state.shape[0]), delta=dh[idx, ...], copy=copy)
      delta[idx, ...] = self._internal_backward(self.uf, inpt=inpt, indices=idx, delta=delta, copy=copy)

    return self

  def update(self):
    '''
    Update function for the LSTM_layer object. optimizer must be assigned
      externally as an optimizer object.
    '''

    check_is_fitted(self, 'delta')

    self.uf.update()
    self.ui.update()
    self.ug.update()
    self.uo.update()
    self.wf.update()
    self.wi.update()
    self.wg.update()
    self.wo.update()

    return self

if __name__ == '__main__':

  import pylab as plt
  from NumPyNet.utils import to_categorical

  data = {'good': True,
          'bad': False,
          'happy': True,
          'sad': False,
          'not good': False,
          'not bad': True,
          'not happy': False,
          'not sad': True,
          'very good': True,
          'very bad': False,
          'very happy': True,
          'very sad': False,
          'i am happy': True,
          'this is good': True,
          'i am bad': False,
          'this is bad': False,
          'i am sad': False,
          'this is sad': False,
          'i am not happy': False,
          'this is not good': False,
          'i am not bad': True,
          'this is not sad': True,
          'i am very happy': True,
          'this is very good': True,
          'i am very bad': False,
          'this is very sad': False,
          'this is very happy': True,
          'i am good not bad': True,
          'this is good not bad': True,
          'i am bad not good': False,
          'i am good and happy': True,
          'this is not good and not happy': False,
          'i am not at all good': False,
          'i am not at all bad': True,
          'i am not at all happy': False,
          'this is not at all sad': True,
          'this is not at all happy': False,
          'i am good right now': True,
          'i am bad right now': False,
          'this is bad right now': False,
          'i am sad right now': False,
          'i was good earlier': True,
          'i was happy earlier': True,
          'i was bad earlier': False,
          'i was sad earlier': False,
          'i am very bad right now': False,
          'this is very good right now': True,
          'this is very sad right now': False,
          'this was bad earlier': False,
          'this was very good earlier': True,
          'this was very bad earlier': False,
          'this was very happy earlier': True,
          'this was very sad earlier': False,
          'i was good and not bad earlier': True,
          'i was not good and not happy earlier': False,
#          'i am not at all bad or sad right now': True,
#          'i am not at all good or happy right now': False,
          'this was not happy and not good earlier': False,
        }

  # TODO: Up to now it works only if the number of samples is perfectly divisible by batch!!!

  words = set((w for text in data.keys() for w in text.split(' ')))
  print('{:d} unique words found'.format(len(words)))

  coding = {k : v for k, v in enumerate(words)}

  one_hot_encoding = to_categorical(list(data.keys()))
  batch, size = one_hot_encoding.shape
  one_hot_encoding = one_hot_encoding.reshape(batch, 1, 1, size)

  outputs = 8

  # Model initialization
  layer = LSTM_layer(outputs, steps=3, input_shape=one_hot_encoding.shape)
  print(layer)

  layer.forward(one_hot_encoding)
  forward_out = layer.output.copy()

  layer.wf.delta = np.ones(shape=(layer.wf.out_shape), dtype=float)
  layer.wi.delta = np.ones(shape=(layer.wi.out_shape), dtype=float)
  layer.wg.delta = np.ones(shape=(layer.wg.out_shape), dtype=float)
  layer.wo.delta = np.ones(shape=(layer.wo.out_shape), dtype=float)
  layer.uf.delta = np.ones(shape=(layer.uf.out_shape), dtype=float)
  layer.ug.delta = np.ones(shape=(layer.ug.out_shape), dtype=float)
  layer.ui.delta = np.ones(shape=(layer.ui.out_shape), dtype=float)
  layer.uo.delta = np.ones(shape=(layer.uo.out_shape), dtype=float)
  layer.delta    = np.ones(shape=(layer.out_shape), dtype=float)
  delta = np.zeros(shape=(batch, 1, 1, size), dtype=float)
  layer.backward(one_hot_encoding, delta=delta, copy=True)


  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('LSTM Layer')

  ax1.imshow(one_hot_encoding[:, 0, 0, :])
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.matshow(forward_out[:, 0, 0, :].T, cmap='bwr')
  ax2.set_title('Forward', y=3)
  ax2.axes.get_xaxis().set_visible(False)         # no x axis tick

  ax3.imshow(delta[:, 0, 0, :])
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
