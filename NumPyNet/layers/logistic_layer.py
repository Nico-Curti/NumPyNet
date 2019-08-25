#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Logistic Layer'


class Logistic_layer(object):

  def __init__(self):
    '''
    Logistic Layer: performs a logistic transformation of the input and computes
      the binary cross entropy cost.

    Parameters:
    '''
    self.w, self.h, self.c  = (0, 0, 0)
    self.output, self.delta, self.loss  = (None, None, None)

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape()
    return 'logistic x entropy                       {:>4d} x{:>4d} x{:>4d} x{:>4d}' .format(
           batch, out_width, out_height, out_channels)

  def out_shape(self):
    return (self.batch, self.w, self.h, self.c)

  def forward(self, inpt, truth=None) :
    '''
    Forward function of the logistic layer, now the output should be consistent with darknet

    Parameters:
      inpt : output of the network with shape (batch, w, h, c)
      truth : arrat of same shape as input (without the batch dimension),
        if given, the function computes the binary cross entropy
    '''

    self.batch, self.w, self.h, self.c = inpt.shape
    # inpt = np.log(inpt/(1-inpt))
    self.output = 1. / (1. + np.exp(-inpt)) # as for darknet
    # self.output = inpt

    if truth is not None:
      out = np.clip(self.output, 1e-8, 1. - 1e-8)
      self.loss = -truth * np.log(out) - (1. - truth) * np.log(1. - out)
      out_upd = out * (1. - out)
      out_upd[out_upd <= 1e-8] = 1e-8
      self.delta = (truth - out) * out_upd
      # self.cost = np.mean(self.loss)
      self.cost = np.sum(self.loss) # as for darknet
    else :
      self.delta = np.zeros(shape=self.out_shape())

  def backward(self, delta=None):
    '''
    Backward function of the Logistic Layer

    Parameters:
      delta : array same shape as the input.
    '''
    if delta is not None:
      delta[:] += self.delta # as for darknet, probably an approx



if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)
  inpt = inpt * 2. - 1.

  inpt = np.expand_dims(inpt, axis=0)

  np.random.seed(123)
  batch, w, h, c = inpt.shape

  # truth definition, it's random so don't expect much
  truth = np.random.choice([0., 1.], p=[.5, .5], size=(batch, w, h, c))

  # Model Initialization
  layer = Logistic_layer()

  # FORWARD

  layer.forward(inpt, truth)
  forward_out = layer.output
  layer_loss = layer.cost

  print(layer)
  print('Loss: {:.3f}'.format(layer_loss))

  # BACKWARD

  delta = np.zeros(shape=inpt.shape, dtype=float)
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Logistic Layer:\nloss({0:.3f})'.format(layer_loss))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.axis('off')
  ax1.set_title('Original Image')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.axis('off')
  ax2.set_title('Forward Image')

  ax3.imshow(float_2_img(delta[0]))
  ax3.axis('off')
  ax3.set_title('Delta Image')

  plt.show()


