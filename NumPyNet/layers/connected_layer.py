#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations

import sys
import numpy as np


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Connected Layer'


class Connected_layer(object):

  def __init__(self, inputs, outputs, activation=Activations, weights=None, bias=None):
    '''
    Connected layer:

    Parameters :
      inputs     : tuple, shape of the input in the format (batch, w, h, c)
      outputs    : int, number of output of the layers
      activation : activation function of the layer
      weights    : array of shape (w * h * c, outputs), weights of the dense layer
      bias       : array of shape (outputs, ), bias of the dense layer
    '''
    self.batch, self.w, self.h, self.c = inputs
    self.inputs = inputs[1:]
    self.outputs = outputs

    self.activation = activation.activate
    self.gradient = activation.gradient

    if weights is not None:
      self.weights = weights
    else:
      # initialize weights with shape (w*h*c, outputs)
      self.weights = np.random.uniform(low=0., high=1., size=(np.prod(self.inputs), outputs))

    if bias is not None:
      self.bias = bias
    else:
      self.bias = np.ones(shape=(outputs,))

    self.output, self.delta = (None, None)
    self.weights_update = np.zeros(shape=self.weights.shape)
    self.bias_update = np.zeros(shape=(outputs,))

  def __str__(self):
    return 'connected            {:4d} x{:4d} x{:4d}  ->  {:4d}'.format(
            self.w, self.h, self.c, self.outputs)

  def out_shape(self):
    return (self.batch, self.outputs)

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

    inpt = inpt.reshape(-1, self.w * self.h * self.c)    # shape (batch, w*h*c)

    # z = (inpt @ self.weights) + self.bias')      # shape (batch, outputs)
    z = np.dot(inpt, self.weights) + self.bias

    self.output = self.activation(z, copy=copy)     # shape (batch, outputs), activated

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
    inpt = inpt.reshape(self.batch, -1)

    self.delta *= self.gradient(self.output, copy=copy)

    self.bias_update += self.delta.sum(axis=0)   # shape : (outputs,)

    # self.weights_update += inpt.transpose() @ self.delta') # shape : (w * h * c, outputs)
    self.weights_update += np.dot(inpt.transpose(), self.delta)

    if delta is not None:
      delta_shaped = delta.reshape(self.batch, -1)  # it's a reshaped VIEW

      # shapes : (batch , w * h * c) = (batch , w * h * c) + (batch, outputs) @ (outputs, w * h * c)

      # delta_shaped[:] += self.delta @ self.weights.transpose()')  # I can modify delta using its view
      delta_shaped[:] += np.dot(self.delta, self.weights.transpose())

  def update(self, momentum=0., decay=0., lr=1e-2, lr_scale=1.):
    '''
    update function for the connected layer

    Parameters:
      momentum : float, default = 0., scale factor of weight update
      decay    : float, default = 0., determines the decay of weights_update
      lr       : float, default = 1e-02, learning rate of the layer
      lr_scale : float, default = 1., learning rate scale of the layer
    '''
    # Update rule copied from darknet, missing batch_normalize
    lr *= lr_scale
    lr /= self.batch

    # Bias update
    self.bias += lr * self.bias_update

    # Weights update
    self.weights_update += (-decay) * batch * self.weights
    self.weights        += lr * self.weights_update
    self.weights_update *= momentum


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
  weights = np.random.uniform(low=-1, high=1., size=(np.prod(inpt.shape[1:]), outputs))
  bias    = np.random.uniform(low=-1,  high=1., size=(outputs,))

  # Model initialization
  layer = Connected_layer(inpt.shape, outputs,
                          activation=layer_activation, weights=weights, bias=bias)

  # FORWARD

  layer.forward(inpt)
  forward_out_byron = layer.output.copy()

  # BACKWARD

  layer.delta = np.ones(shape=(batch, outputs))
  delta = np.zeros(shape=(batch, w, h, c))
  layer.backward(inpt, delta=delta, copy=True)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('Connected Layer\nactivation : {}'.format(layer_activation.name))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.matshow(forward_out_byron, cmap='bwr')
  ax2.set_title('Forward', y = 4)
  ax2.axes.get_yaxis().set_visible(False)         # no y axis tick
  ax2.axes.get_xaxis().set_ticks(range(outputs))  # set x axis tick for every output

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()
