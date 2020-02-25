#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.layers import Connected_layer
from NumPyNet.activations import Activations
from NumPyNet.utils import check_is_fitted

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class RNN_layer(object):

  def __init__(self, outputs, steps, activation=Activations, input_shape=None, weights=None, bias=None, **kwargs):
    '''
    RNN layer

    Parameters
    ----------
      outputs     : integer, number of outputs of the layers
      steps       : integer, number of mini-batch/steps to perform.
      activation  : activation function of the layer
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

      self.batch = None
      self.batches = None
      initial_shape = None

    self.input_layer  = Connected_layer(self.outputs, activation, input_shape=initial_shape, weights=weights[0], bias=bias[0])
    self.self_layer   = Connected_layer(self.outputs, activation, input_shape=(self.batch, 1, 1, self.outputs), weights=weights[1], bias=bias[1])
    self.output_layer = Connected_layer(self.outputs, activation, input_shape=(self.batch, 1, 1, self.outputs), weights=weights[2], bias=bias[2])

    if input_shape is not None:
      self.input_layer.input_shape  = (self.input_shape[0], w, h, c)
      self.self_layer.input_shape   = (self.input_shape[0], w, h, self.outputs)
      self.output_layer.input_shape = (self.input_shape[0], w, h, self.outputs)

    self.state     = None

  def __str__(self):
    return '\n\t'.join(('RNN Layer: {:d} inputs, {:d} outputs'.format(self.inputs, self.outputs),
                        '{}'.format(self.input_layer),
                        '{}'.format(self.self_layer),
                        '{}'.format(self.output_layer)
                        ))

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.input_shape = previous_layer.out_shape

    self.batch = self.input_shape[0] // self.steps
    self.batches = np.array_split(range(self.input_shape[0]), indices_or_sections=self.batch)

    _, w, h, c = self.input_shape
    initial_shape = (self.batch, w, h, c)
    activation = self.self_layer.activation.__qualname__
    activation = activation.split('.')[0]

    self.input_layer = Connected_layer(self.outputs, activation=activation, input_shape=initial_shape)

    self.input_layer.input_shape  = (self.input_shape[0], w, h, c)
    self.self_layer.input_shape   = (self.input_shape[0], w, h, self.outputs)
    self.output_layer.input_shape = (self.input_shape[0], w, h, self.outputs)

    return self

  @property
  def inputs(self):
    return np.prod(self.input_shape[1:])

  @property
  def out_shape(self):
    return self.output_layer.out_shape

  @property
  def output(self):
    return self.output_layer.output

  @property
  def delta(self):
    return self.output_layer.delta

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

    pos = self.input_layer.load_weights(chunck_weights, pos=pos)
    pos = self.self_layer.load_weights(chunck_weights, pos=pos)
    pos = self.output_layer.load_weights(chunck_weights, pos=pos)
    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.input_layer.bias.ravel(), self.input_layer.weights.ravel(),
                           self.self_layer.bias.ravel(), self.self_layer.weights.ravel(),
                           self.output_layer.bias.ravel(), self.output_layer.weights.ravel()], axis=0).tolist()

  def _internal_forward(self, layer, inpt, indices, copy=False):

    _input = inpt[indices, ...]
    _input = _input.reshape(_input.shape[0], -1)

    z = np.einsum('ij, jk -> ik', _input, layer.weights, optimize=True) + layer.bias
    layer.output[indices, ...] = layer.activation(z, copy=copy).reshape(-1, 1, 1, layer.outputs)


  def forward(self, inpt, shortcut=False, copy=False):
    '''
    Forward function of the RNN layer. It computes the matrix product
      between inpt and weights, add bias and activate the result with the
      chosen activation function.

    Parameters
    ----------
      inpt : numpy array with shape (batch, w, h, c). Input batch of images of the layer
      shortcut : boolean, default False. Enable/Disable internal shortcut connection.

    Returns
    ----------
    RNN_layer object
    '''

    self.state = np.zeros(shape=self.out_shape, dtype=float)
    self.input_layer.output = np.empty(shape=self.input_layer.out_shape, dtype=float)
    self.self_layer.output = np.empty(shape=self.self_layer.out_shape, dtype=float)
    self.output_layer.output = np.empty(shape=self.output_layer.out_shape, dtype=float)

    for idx in self.batches:

      self._internal_forward(layer=self.input_layer, inpt=inpt, indices=idx, copy=copy)
      self._internal_forward(layer=self.self_layer, inpt=self.state, indices=idx, copy=copy)

      if shortcut:
        self.state[idx, ...] += self.input_layer.output[idx, ...] + self.self_layer.output[idx, ...]
      else:
        self.state[idx, ...]  = self.input_layer.output[idx, ...] + self.self_layer.output[idx, ...]

      self._internal_forward(layer=self.output_layer, inpt=self.state, indices=idx, copy=copy)

    self.input_layer.delta = np.zeros(shape=self.input_layer.out_shape, dtype=float)
    self.self_layer.delta = np.zeros(shape=self.self_layer.out_shape, dtype=float)
    self.output_layer.delta = np.zeros(shape=self.output_layer.out_shape, dtype=float)

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

  def backward(self, inpt, delta=None, shortcut=False, copy=False):
    '''
    Backward function of the RNN layer, updates the global delta of the
      network to be Backpropagated, he weights upadtes and the biases updates

    Parameters
    ----------
      inpt     : original input of the layer
      delta    : global delta, to be backpropagated.
      shortcut : boolean, default False. Enable/Disable internal shortcut connection

    Returns
    ----------
    RNN_layer object
    '''

    check_is_fitted(self, 'delta')

    for i, idx in reversed(list(enumerate(self.batches))):

      self.self_layer.delta[idx, ...] = self._internal_backward(self.output_layer, inpt=self.state, indices=idx, delta=self.self_layer.delta, copy=copy)

      if i == 0:
        self._internal_backward(layer=self.self_layer, inpt=self.state, indices=idx, delta=None, copy=copy)
      else:
        self.self_layer.delta[idx, ...] = self._internal_backward(layer=self.self_layer, inpt=self.state, indices=idx, delta=self.self_layer.delta, copy=copy)

      self.input_layer.delta[idx, ...] = self.self_layer.delta[idx, ...]
      if shortcut and i > 0:
        self.input_layer.delta[idx, ...] += self.self_layer.delta[batches[idx - 1], ...]

      if delta is not None:
        delta[idx, ...] = self._internal_backward(layer=self.input_layer, inpt=inpt, indices=idx, delta=delta, copy=copy)
      else:
        self._internal_backward(layer=self.input_layer, inpt=inpt, indices=idx, delta=None, copy=copy)

    return self

  def update(self):
    '''
    Update function for the RNN_layer object. optimizer must be assigned
      externally as an optimizer object.
    '''

    check_is_fitted(self, 'delta')

    self.input_layer.update()
    self.self_layer.update()
    self.output_layer.update()

    return self

if __name__ == '__main__':

  import pylab as plt
  from NumPyNet.utils import to_categorical
  from NumPyNet import activations

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
          'i am not at all bad or sad right now': True,
          'i am not at all good or happy right now': False,
          'this was not happy and not good earlier': False,
        }

  words = set((w for text in data.keys() for w in text.split(' ')))
  print('{:d} unique words found'.format(len(words)))

  coding = {k : v for k, v in enumerate(words)}

  one_hot_encoding = to_categorical(list(data.keys()))
  batch, size = one_hot_encoding.shape
  one_hot_encoding = one_hot_encoding.reshape(batch, 1, 1, size)

  outputs = 10
  layer_activation = activations.Relu()

  # Model initialization
  layer = RNN_layer(outputs, steps=3, input_shape=one_hot_encoding.shape,
                    activation=layer_activation)
  print(layer)

  layer.forward(one_hot_encoding)
  forward_out = layer.output.copy()

  layer.input_layer.delta = np.ones(shape=(layer.input_layer.out_shape), dtype=float)
  layer.self_layer.delta = np.ones(shape=(layer.self_layer.out_shape), dtype=float)
  layer.output_layer.delta = np.ones(shape=(layer.output_layer.out_shape), dtype=float)
  delta = np.zeros(shape=(batch, 1, 1, size), dtype=float)
  layer.backward(one_hot_encoding, delta=delta, copy=True)


  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('RNN Layer\nactivation : {}'.format(layer_activation.name))

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
