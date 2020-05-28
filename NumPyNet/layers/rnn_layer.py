#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.layers import Connected_layer
from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted
from NumPyNet.exception import LayerError

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

    self.activation = _check_activation(self, activation)

    # if input shape is passed, init of weights, else done in  __call__
    if input_shape is not None:
      b, w, h, c = input_shape

      if b < self.steps:
        class_name = self.__class__.__name__
        prev_name  = layer.__class__.__name__
        raise LayerError('Incorrect steps found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

      self.batch = b // self.steps
      self.input_shape = (self.batch, w, h, c)
      indices = np.arange(0, b)
      self.batches = np.lib.stride_tricks.as_strided(indices, shape=(self.steps, self.batch), strides=(self.batch * 8, 8)).copy()

      self.input_layer  = Connected_layer(self.outputs, self.activation, input_shape=(self.batches[-1][-1] + 1, w, h, c), weights=weights[0], bias=bias[0])
      self.self_layer   = Connected_layer(self.outputs, self.activation, weights=weights[1], bias=bias[1])(self.input_layer)
      self.output_layer = Connected_layer(self.outputs, self.activation, weights=weights[2], bias=bias[2])(self.self_layer)

      self.state = np.zeros(shape=(self.batch, w, h, self.outputs), dtype=float)

    else:
      self.batch = None
      self.input_shape = None
      self.batches = None

      self.input_layer = None
      self.self_layer = None
      self.output_layer = None
      self.state = None

    self.prev_state = None
    self.output, self.delta = (None, None)

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

    b, w, h, c = previous_layer.out_shape

    if b < self.steps:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect steps found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.batch = b // self.steps
    self.input_shape = (self.batch, w, h, c)
    indices = np.arange(0, b)

    self.batches = np.lib.stride_tricks.as_strided(indices, shape=(self.steps, self.batch), strides=(self.batch * 8, 8)).copy()

    self.input_layer  = Connected_layer(self.outputs, self.activation, input_shape=(self.batches[-1][-1] + 1, w, h, c))
    self.self_layer   = Connected_layer(self.outputs, self.activation)(self.input_layer)
    self.output_layer = Connected_layer(self.outputs, self.activation)(self.self_layer)

    self.state = np.zeros(shape=(self.batch, w, h, self.outputs), dtype=float)

    self.output, self.delta = (None, None)

    return self

  @property
  def inputs(self):
    return np.prod(self.input_shape[1:]) * self.steps

  @property
  def out_shape(self):
    return self.output_layer.out_shape

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

  def forward(self, inpt, copy=True, trainable=True):
    '''
    Forward function of the RNN layer. It computes the matrix product
      between inpt and weights, add bias and activate the result with the
      chosen activation function.

    Parameters
    ----------
      inpt : numpy array with shape (batch, w, h, c). Input batch of images of the layer

    Returns
    ----------
    RNN_layer object
    '''

    if trainable:
      self.prev_state = self.state.copy()
      self.state = np.zeros_like(self.state)

    self.input_layer.output = np.zeros(shape=self.input_layer.out_shape, dtype=float)
    self.self_layer.output = np.zeros(shape=self.self_layer.out_shape, dtype=float)
    self.output_layer.output = np.zeros(shape=self.output_layer.out_shape, dtype=float)

    for idx in self.batches:

      _input = inpt[idx, ...].reshape(len(idx), -1)
      _state = self.state.reshape(self.state.shape[0], -1)

      z = np.einsum('ij, jk -> ik', _input, self.input_layer.weights, optimize=True) + self.input_layer.bias
      self.input_layer.output[idx, ...] = self.input_layer.activation(z, copy=copy).reshape(-1, 1, 1, self.input_layer.outputs)

      z = np.einsum('ij, jk -> ik', _state, self.self_layer.weights, optimize=True) + self.self_layer.bias
      self.self_layer.output[idx, ...] = self.self_layer.activation(z, copy=copy).reshape(-1, 1, 1, self.self_layer.outputs)

      self.state = self.input_layer.output[idx, ...] + self.self_layer.output[idx, ...]
      _state = self.state.reshape(self.state.shape[0], -1)

      z = np.einsum('ij, jk -> ik', _state, self.output_layer.weights, optimize=True) + self.output_layer.bias
      self.output_layer.output[idx, ...] = self.output_layer.activation(z, copy=copy).reshape(-1, 1, 1, self.output_layer.outputs)


    self.input_layer.delta = np.zeros(shape=self.input_layer.out_shape, dtype=float)
    self.self_layer.delta = np.zeros(shape=self.self_layer.out_shape, dtype=float)
    self.output_layer.delta = np.zeros(shape=self.output_layer.out_shape, dtype=float)

    self.output = self.output_layer.output
    self.delta = self.output_layer.delta

    return self


  def backward(self, inpt, delta=None, copy=True):
    '''
    Backward function of the RNN layer, updates the global delta of the
      network to be Backpropagated, he weights upadtes and the biases updates

    Parameters
    ----------
      inpt     : original input of the layer
      delta    : global delta, to be backpropagated.

    Returns
    ----------
    RNN_layer object
    '''

    check_is_fitted(self, 'delta')

    last_input = self.input_layer.output[self.batches[-1]]
    last_self  = self.self_layer.output[self.batches[-1]]

    for i, idx in reversed(list(enumerate(self.batches))):

      self.state = self.input_layer.output[idx, ...] + self.self_layer.output[idx, ...]

      _state = self.state.reshape(self.state.shape[0], -1)
      _input = inpt[idx, ...].reshape(len(idx), -1)

      # output_layer backward

      _delta = self.output_layer.delta[idx, ...]
      _delta[:] *= self.output_layer.gradient(self.output_layer.output[idx, ...], copy=copy)
      _delta_r = _delta.reshape(-1, self.output_layer.outputs)

      self.output_layer.bias_update = _delta_r.sum(axis=0)
      self.output_layer.weights_update = np.einsum('ji, jk -> ik', _state, _delta_r, optimize=True)

      delta_view = self.self_layer.delta[idx, ...]
      delta_shaped = delta_view.reshape(len(idx), -1)
      delta_shaped[:] += np.einsum('ij, kj -> ik', _delta_r, self.output_layer.weights, optimize=True)
      self.self_layer.delta[idx, ...] = delta_shaped.reshape(delta_view.shape)

      # end 1st backward

      if i != 0:
        idx2 = self.batches[i - 1]
        self.state = self.input_layer.output[idx2, ...] + self.self_layer.output[idx2, ...]
      else:
        self.state = self.prev_state.copy()

      self.input_layer.delta[idx, ...] = self.self_layer.delta[idx, ...]

      _state = self.state.reshape(self.state.shape[0], -1)

      # self_layer backward

      _delta = self.self_layer.delta[idx, ...]
      _delta[:] *= self.self_layer.gradient(self.self_layer.output[idx, ...], copy=copy)
      _delta_r = _delta.reshape(-1, self.self_layer.outputs)

      self.self_layer.bias_update = _delta_r.sum(axis=0)
      self.self_layer.weights_update = np.einsum('ji, jk -> ik', _state, _delta_r, optimize=True)

      if i > 0:
        idx2 = self.batches[i - 1]

        delta_view = self.self_layer.delta[idx2, ...]
        delta_shaped = delta_view.reshape(len(idx2), -1)
        delta_shaped[:] += np.einsum('ij, kj -> ik', _delta_r, self.self_layer.weights, optimize=True)
        self.self_layer.delta[idx2, ...] = delta_shaped.reshape(delta_view.shape)

      # end 2nd backward

      # input_layer backward

      _delta = self.input_layer.delta[idx, ...]
      _delta[:] *= self.input_layer.gradient(self.input_layer.output[idx, ...], copy=copy)
      _delta_r = _delta.reshape(-1, self.input_layer.outputs)

      self.input_layer.bias_update = _delta_r.sum(axis=0)
      self.input_layer.weights_update = np.einsum('ji, jk -> ik', _input, _delta_r, optimize=True)

      if delta is not None:
        delta_view = delta[idx, ...]
        delta_shaped = delta_view.reshape(len(idx), -1)
        delta_shaped[:] += np.einsum('ij, kj -> ik', _delta_r, self.input_layer.weights, optimize=True)
        delta[idx, ...] = delta_shaped.reshape(delta_view.shape)

      # end 3rd backward

    self.state = last_input[idx, ...] + last_self[idx, ...]

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
  layer = RNN_layer(outputs, steps=4, input_shape=one_hot_encoding.shape,
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
  ax2.axes.get_xaxis().set_visible(False)

  ax3.imshow(delta[:, 0, 0, :])
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
