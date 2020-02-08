#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation

import numpy as np
from NumPyNet.exception import LayerError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Connected Layer'


class Connected_layer(object):

  def __init__(self, input_shape, outputs, activation=Activations, weights=None, bias=None, **kwargs):
    '''
    Connected layer:

    Parameters :
      input_shape : tuple, shape of the input in the format (batch, w, h, c)
      outputs     : int, number of output of the layers
      activation  : activation function of the layer
      weights     : array of shape (w * h * c, outputs), weights of the dense layer
      bias        : array of shape (outputs, ), bias of the dense layer
    '''
    self._out_shape = input_shape
    self.inputs = np.prod(input_shape[1:])
    self.outputs = outputs

    activation = _check_activation(self, activation)

    self.activation = activation.activate
    self.gradient = activation.gradient

    if weights is not None:
      self.weights = np.asarray(weights)
    else:
      # initialize weights with shape (w*h*c, outputs)
      scale = np.sqrt(2. / self.inputs)
      self.weights = np.random.uniform(low=-scale, high=scale, size=(self.inputs, self.outputs))

    if bias is not None:
      self.bias = np.asarray(bias)
    else:
      self.bias = np.zeros(shape=(self.outputs,), dtype=float)

    self.output, self.delta = (None, None)
    self.weights_update = None
    self.bias_update    = None
    self.optimizer      = None

  def __str__(self):
    b, w, h, c = self._out_shape
    return 'connected              {:4d} x{:4d} x{:4d} x{:4d}   ->  {:4d} x{:4d}'.format(
            b, w, h, c, b, self.outputs)

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self._out_shape = previous_layer.out_shape
    return self

  @property
  def out_shape(self):
    return (self._out_shape[0], 1, 1, self.outputs)

  def load_weights(self, chunck_weights, pos=0):
    '''
    Load weights from full array of model weights

    Parameters:
      chunck_weights : numpy array of model weights
      pos : current position of the array
    '''
    self.bias = chunck_weights[pos : pos + self.outputs]
    pos += self.outputs

    self.weights = chunck_weights[pos : pos + self.weights.size]
    self.weights = self.weights.reshape(self.inputs, self.outputs)
    pos += self.weights.size

    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.bias.ravel(), self.weights.ravel()], axis=0).tolist()


  def forward(self, inpt, copy=False):
    '''
    Forward function of the connected layer. It computes the matrix product
      between inpt and weights, add bias and activate the result with the
      chosen activation function.

    Parameters:
      inpt : numpy array with shape (batch, w, h, c) input batch of images of the layer
      copy : boolean, states if the activation function have to return a copy of the
            input or not.
    '''

    inpt = inpt.reshape(inpt.shape[0], -1)                  # shape (batch, w*h*c)

    # z = (inpt @ self.weights) + self.bias                # shape (batch, outputs)
    z = np.einsum('ij, jk -> ik', inpt, self.weights, optimize=True) + self.bias
    # z = np.dot(inpt, self.weights) + self.bias

    # shape (batch, outputs), activated
    self.output = self.activation(z, copy=copy).reshape(-1, 1, 1, self.outputs)
    self.delta  = np.zeros(shape=self.out_shape, dtype=float)

  def backward(self, inpt, delta=None, copy=False):
    '''
    Backward function of the connected layer, updates the global delta of the
      network to be Backpropagated, he weights upadtes and the biases updates

    Parameters:
      inpt  : original input of the layer
      delta : global delta, to be backpropagated.
      copy  : boolean, states if the activation function have to return a copy of the
            input or not.
    '''

    # reshape to (batch , w * h * c)
    inpt = inpt.reshape(self._out_shape[0], -1)
    # out  = self.output.reshape(-1, self.outputs)

    self.delta *= self.gradient(self.output, copy=copy)
    self.delta = self.delta.reshape(-1, self.outputs)

    self.bias_update = self.delta.sum(axis=0)   # shape : (outputs,)

    # self.weights_update += inpt.transpose() @ self.delta') # shape : (w * h * c, outputs)
    self.weights_update = np.dot(inpt.transpose(), self.delta)

    if delta is not None:
      delta_shaped = delta.reshape(self._out_shape[0], -1)  # it's a reshaped VIEW

      # shapes : (batch , w * h * c) = (batch , w * h * c) + (batch, outputs) @ (outputs, w * h * c)

      # delta_shaped[:] += self.delta @ self.weights.transpose()')  # I can modify delta using its view
      delta_shaped[:] += np.dot(self.delta, self.weights.transpose())

  def update(self):
    '''
    update function for the convolution layer

    Parameters:
      optimizer : Optimizer object
    '''
    self.bias, self.weights = self.optimizer.update(params=[self.bias, self.weights],
                                                    gradients=[self.bias_update, self.weights_update]
                                                   )


if __name__ == '__main__':

  import pylab as plt
  from PIL import Image

  import os

  from NumPyNet import activations

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)

  # from (w, h, c) to shape (1, w, h, c)
  inpt = np.expand_dims(inpt, axis=0) # just to add the 'batch' dimension

  # Number of outputs
  outputs = 10
  layer_activation = activations.Relu()
  batch, w, h, c = inpt.shape

  # Random initialization of weights with shape (w * h * c) and bias with shape (outputs,)
  np.random.seed(123) # only if one want always the same set of weights
  weights = np.random.uniform(low=-1., high=1., size=(np.prod(inpt.shape[1:]), outputs))
  bias    = np.random.uniform(low=-1., high=1., size=(outputs,))

  # Model initialization
  layer = Connected_layer(inpt.shape, outputs,
                          activation=layer_activation, weights=weights, bias=bias)
  print(layer)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output.copy()

  # BACKWARD

  layer.delta = np.ones(shape=(layer.out_shape), dtype=float)
  delta = np.zeros(shape=(batch, w, h, c), dtype=float)
  layer.backward(inpt, delta=delta, copy=True)

  # print('Output: {}'.format(', '.join( ['{:.3f}'.format(x) for x in forward_out[0]] ) ) )

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('Connected Layer\nactivation : {}'.format(layer_activation.name))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.matshow(forward_out[:,0,0,:], cmap='bwr')
  ax2.set_title('Forward', y = 4)
  ax2.axes.get_yaxis().set_visible(False)         # no y axis tick
  ax2.axes.get_xaxis().set_ticks(range(outputs))  # set x axis tick for every output

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()
