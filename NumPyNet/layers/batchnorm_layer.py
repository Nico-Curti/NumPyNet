#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'BatchNorm Layer'


class BatchNorm_layer(object):

  def __init__(self, scales=None, bias=None):

    '''
    BatchNormalization Layer: It performs a Normalization over the Batch axis
      of the Input. Both scales and bias are trainable weights

      equation : output = scale * input_normalized + bias

    Parameters:
      scales : scale to be multiplied to the normalized input, of shape (w, h, c)
      bias   : bias to be added to the multiplication of scale and normalized input of shape (w, h, c)
    '''

    self.scales = scales
    self.bias = bias

    self.batch, self.w, self.h, self.c = (0, 0, 0, 0)
    self.output, self.delta = (None, None)

    #Updates
    self.scales_updates, self.bias_updates = (None, None)


  def __str__(self):
    return 'Batch Normalization Layer: {:4d} x {:4d} x {:4d} image'.format(
           self.w, self.h, self.c)

  def out_shape(self):
    return (self.batch, self.w, self.h, self.c)

  def forward(self, inpt, epsil=1e-8):
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
      epsil : float, used to avoi division by zero when computing 1. / var
    '''

    self.batch, self.w, self.h, self.c = inpt.shape

    # Copy input, compute mean and inverse variance with respect the batch axis
    self.x    = inpt
    self.mean = self.x.mean(axis=0)  # Shape = (w, h, c)
    self.var  = 1. / np.sqrt(self.x.var(axis=0) + epsil) # shape = (w, h, c)
    # epsil is used to avoid divisions by zero

    # Compute the normalized input
    self.x_norm = (self.x - self.mean) * self.var # shape (batch, w, h, c)
    self.output = self.x_norm.copy() # made a copy to store x_norm, used in Backward

    # Output = scale * x_norm + bias
    if self.scales is not None:
      self.output *= self.scales  # Multiplication for scales

    if self.bias is not None:
      self.output += self.bias # Add bias

    # output_shape = (batch, w, h, c)


  def backward(self, delta=None):
    '''
    BackPropagation function of the BatchNormalization layer. Every formula is a derivative
    computed by chain rules: dbeta = derivative of output w.r.t. bias, dgamma = derivative of
    output w.r.t. scales etc...

    Parameters:
      delta : the global error to be backpropagated, its shape should be the same
        as the input of the forward function (batch, w, h ,c)
    '''


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
      delta[:] = self.delta

    def update(self, momentum=0., decay=0., lr=1e-2, lr_scale=1.):
      '''
      Update function of the BatchNormalization layer

      Parameters:
        momentum : float, default = 0, momentum of the optimizer
        decay    : float, default = 0, decay parameter
        lr       : float, default = 1e-2, learning rate parameter
        lr_scale : float, default = 1 , learning rate scale

      '''

      lr *= lr_scale
      lr /= self.batch

      # bias updates
      self.bias += self.bias_updates * lr

      # Scales updates
      self.scales_updates += (-decay) * self.batch * self.weights
      self.scales         += lr * self.scales_updates
      self.scale_updates  *= momentum



if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  # I need to load at least to images, or made a copy of it
  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  w, h, c = inpt.shape

  batch_size = 5

  np.random.seed(123) # set seed to have fixed bias and scales

  # create a pseudo-input with batch_size images with a random offset from the original image
  rng = np.random.uniform(low=0., high=100., size=(batch_size, w, h, c))
  inpt = np.concatenate([np.expand_dims(inpt, axis=0) + r for r in rng], axis=0) # create a set of image

  # img_to_float of input, to work with numbers btween 0. and 1.
  inpt = np.asarray([img_2_float(x) for x in inpt ])

  b, w, h, c = inpt.shape # needed for initializations of bias and scales

  # bias   = np.random.uniform(0., 1., size=(w, h, c)) # random biases
  # scales = np.random.uniform(0., 1., size=(w, h, c)) # random scales

  bias = np.zeros(shape=(w, h, c), dtype=float)
  scales = np.ones(shape=(w, h, c), dtype=float)

  # Model Initialization
  layer = BatchNorm_layer(scales=scales, bias=bias)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output
  print(layer)

  # BACKWARD

  layer.delta = np.random.uniform(low=0., high=100., size=layer.out_shape())
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

  plt.show()
