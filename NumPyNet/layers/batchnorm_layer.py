#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.utils import check_is_fitted
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class BatchNorm_layer(BaseLayer):

  epsil = 1e-8

  def __init__(self, scales=None, bias=None, input_shape=None, **kwargs):

    '''
    BatchNormalization Layer: It performs a Normalization over the Batch axis
      of the Input. Both scales and bias are trainable weights

      equation : output = scale * input_normalized + bias

    Parameters:
      scales : scale to be multiplied to the normalized input, of shape (w, h, c)
      bias   : bias to be added to the multiplication of scale and normalized input of shape (w, h, c)
      input_shape : tuple of 4 integers: input shape of the layer.
    '''

    self.scales = scales
    self.bias = bias

    #Updates
    self.scales_updates, self.bias_updates = (None, None)
    self.optimizer = None

    super(BatchNorm_layer, self).__init__(input_shape=input_shape)

  def __str__(self):
    return 'batchnorm                    {0:4d} x{1:4d} x{2:4d} image'.format(*self.out_shape[1:])

  def load_weights(self, chunck_weights, pos=0):
    '''
    Load weights from full array of model weights

    Parameters:
      chunck_weights : numpy array of model weights
      pos : current position of the array
    '''
    outputs = np.prod(self.out_shape)

    self.bias = chunck_weights[pos : pos + outputs]
    self.bias = self.bias.reshape(self.out_shape)
    pos += outputs

    self.scales = chunck_weights[pos : pos + outputs]
    self.scales = self.scales.reshape(self.out_shape)
    pos += outputs

    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.bias.ravel(), self.scales.ravel()], axis=0).tolist()

  def forward(self, inpt):
    '''
    Forward function of the BatchNormalization layer. It computes the output of
    the layer, the formula is :

                    output = scale * input_norm + bias

    Where input_norm is:

                    input_norm = (input - mean) / sqrt(var + epsil)

    where mean and var are the mean and the variance of the input batch of
    images computed over the first axis (batch)

    Parameters:
      inpt  : numpy array, batch of input images in the format (batch, w, h, c)
    '''

    self._check_dims(shape=self.input_shape, arr=inpt, func='Forward')

    # Copy input, compute mean and inverse variance with respect the batch axis
    self.x    = inpt.copy()
    self.mean = self.x.mean(axis=0)                             # shape = (w, h, c)
    self.var  = 1. / np.sqrt((self.x.var(axis=0)) + self.epsil) # shape = (w, h, c)
    # epsil is used to avoid divisions by zero

    # Compute the normalized input
    self.x_norm = (self.x - self.mean) * self.var # shape (batch, w, h, c)
    self.output = self.x_norm.copy() # made a copy to store x_norm, used in Backward

    # Init scales and bias if they are not initialized (ones and zeros)
    if self.scales is None:
      self.scales = np.ones(shape=self.out_shape[1:])

    if self.bias is None:
      self.bias = np.zeros(shape=self.out_shape[1:])

    # Output = scale * x_norm + bias
    self.output = self.output * self.scales + self.bias

    # output_shape = (batch, w, h, c)
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta=None):
    '''
    BackPropagation function of the BatchNormalization layer. Every formula is a derivative
    computed by chain rules: dbeta = derivative of output w.r.t. bias, dgamma = derivative of
    output w.r.t. scales etc...

    Parameters:
      delta : the global error to be backpropagated, its shape should be the same
        as the input of the forward function (batch, w, h ,c)
    '''

    check_is_fitted(self, 'delta')
    self._check_dims(shape=self.input_shape, arr=delta, func='Forward')

    invN = 1. / np.prod(self.mean.shape)

    # Those are the explicit computation of every derivative involved in BackPropagation
    # of the batchNorm layer, where dbeta = dout / dbeta, dgamma = dout / dgamma etc...

    self.bias_updates = self.delta.sum(axis=0)                   # dbeta
    self.scales_updates = (self.delta * self.x_norm).sum(axis=0) # dgamma

    self.delta *= self.scales                                    # self.delta = dx_norm from now on

    self.mean_delta = (self.delta * (-self.var)).mean(axis=0)    # dmu

    self.var_delta = ((self.delta * (self.x - self.mean)).sum(axis=0) *
                     (-.5 * self.var * self.var * self.var))     # dvar

    # Here, delta is the derivative of the output w.r.t. input
    self.delta = (self.delta * self.var +
                  self.var_delta * 2 * (self.x - self.mean) * invN +
                  self.mean_delta * invN)

    if delta is not None:
      delta[:] += self.delta

    return self

  def update(self):
    '''
    update function for the convolution layer

    Parameters:
      optimizer : Optimizer object
    '''

    check_is_fitted(self, 'delta')

    self.bias, self.scales = self.optimizer.update(params=[self.bias, self.scales],
                                                   gradients=[self.bias_updates, self.scales_updates]
                                                  )

    return self


if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  # I need to load at least to images, or made a copy of it
  filename = os.path.join(os.path.dirname('__file__'), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  w, h, c = inpt.shape

  batch_size = 5

  np.random.seed(123) # set seed to have fixed bias and scales

  # create a pseudo-input with batch_size images with a random offset from the original image
  rng  = np.random.uniform(low=0., high=100., size=(batch_size, w, h, c))
  inpt = np.concatenate([np.expand_dims(inpt, axis=0) + r for r in rng], axis=0) # create a set of image

  # img_to_float of input, to work with numbers btween 0. and 1.
  inpt = np.asarray([img_2_float(x) for x in inpt ])

  b, w, h, c = inpt.shape # needed for initializations of bias and scales

  bias   = np.random.uniform(0., 1., size=(w, h, c)) # random biases
  scales = np.random.uniform(0., 1., size=(w, h, c)) # random scales

  bias = np.zeros(shape=(w, h, c), dtype=float)
  scales = np.ones(shape=(w, h, c), dtype=float)

  # Model Initialization
  layer = BatchNorm_layer(input_shape=inpt.shape, scales=scales, bias=bias)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output
  print(layer)

  # BACKWARD

  layer.delta = np.random.uniform(low=0., high=100., size=layer.out_shape)
  delta = np.ones(shape=inpt.shape, dtype=float) # delta same shape as the Input
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2) = plt.subplots(nrows=2, ncols=2, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('BatchNormalization Layer')

  ax1[0].imshow(float_2_img(inpt[0]))
  ax1[0].set_title('Original image')
  ax1[0].axis('off')

  ax1[1].imshow(float_2_img(layer.mean))
  ax1[1].set_title("Mean Image")
  ax1[1].axis("off")

  ax2[0].imshow(float_2_img(forward_out[0]))
  ax2[0].set_title('Forward')
  ax2[0].axis('off')

  ax2[1].imshow(float_2_img(delta[0]))
  ax2[1].set_title('Backward')
  ax2[1].axis('off')

  fig.tight_layout()
  plt.show()
