#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Shortcut Layer'


class Shortcut_layer(object):

  def __init__(self, layer1_shape, layer2_shape, activation=Activations, alpha=1., beta=1.):

    '''
    Shortcut layer: activation of the linear combination of the output of two layers

                layer1 * alpha + layer2 * beta = output

    Now working only with same shapes input

    Parameters :
      layer1_shape : tuple, shape of the first layer in the format (batch, w, h, c)
      layer2_shape : tuple, shape of the second layer in the format (batch, w, h, c)
      activation   : activation function of the layer
      alpha        : float, default = 1., first weight of the combination
      beta         : float, default = 1., second weight of the combination

    '''

    self.activation = activation.activate
    self.gradient = activation.gradient

    self.alpha, self.beta = alpha, beta

    self.output, self.delta, self.out = (None, None, None)

    self.layer1_shape = layer1_shape
    self.layer2_shape = layer2_shape

  def __str__(self):
    b1, w1, h1, c1 = self.layer1_shape
    b2, w2, h2, c2 = self.layer2_shape
    return 'Shortcut                 {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d} x{:>4d}'.format(b1, w2, h2, c2, w1, h1, c1)

  def out_shape(self):
    return self.layer2_shape

  def forward(self, inpt, prev_output):
    '''
    Forward function of the Shortcut layer: activation of the linear combination between input

    Parameters:
      inpt        : array of shape (batch, w, h, c), first input of the layer
      prev_output : array of shape (batch, w, h, c), second input of the layer
    '''

    self.output = inpt.copy()

    self.output[:] = self.alpha * self.output[:] + self.beta * prev_output[:]
    # MISS combination
    self.output = self.activation(self.output)

  def backward(self, delta, prev_delta):
    '''
    Backward function of the Shortcut layer

    Parameters:
      delta      : array of shape (batch, w, h, c), first delta to be backpropagated
      delta_prev : array of shape (batch, w, h, c), second delta to be backporpagated

    '''

    # derivatives of the activation funtion w.r.t. to input
    self.out = self.gradient(self.output)
    self.delta *= self.out

    delta      += self.delta * self.alpha
    # MISS combination
    prev_delta += self.delta * self.beta


if __name__ == '__main__':

  import pylab as plt

  from NumPyNet import activations

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  # Set seed to have same input
  np.random.seed(123)

  layer_activ = activations.Relu()

  batch = 2

  alpha = 0.75
  beta  = 0.5

  # Random input
  inpt1      = np.random.uniform(low=-1., high=1., size=(batch, 100, 100, 3))
  inpt2      = np.random.uniform(low=-1., high=1., size=inpt1.shape)
  b, w, h, c = inpt1.shape


  # model initialization
  layer = Shortcut_layer(inpt1.shape, inpt2.shape,
                         activation=layer_activ,
                         alpha=alpha, beta=beta)

  # FORWARD

  layer.forward(inpt1, inpt2)
  forward_out = layer.output.copy()

  # BACKWARD

  delta      = np.zeros(shape=inpt1.shape)
  delta_prev = np.zeros(shape=inpt2.shape)

  layer.delta = np.ones(shape=layer.out_shape())
  layer.backward(delta, delta_prev)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('Shortcut Layer\nalpha : {}, beta : {}, activation : {} '.format(alpha, beta, layer_activ.name))

  ax1.imshow(float_2_img(inpt1[0]))
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title('Forward')
  ax2.axis('off')

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()
