#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Input Layer'

class Input_layer(object):

  def __init__(self, input_shape):
    '''
    Input layer
    '''
    try:

      self.batch, self.w, self.h, self.c = input_shape

    except:
      raise ValueError('Input layer error. Incorrect input_shape. Expected a 4D array (batch, width, height, channel). Given {}'.format(input_shape))

    self.output = np.empty(shape=input_shape, dtype=float)
    self.delta = np.empty(shape=input_shape, dtype=float)

  def __str__(self):
    return 'input:                  {0:>4d} x{1:>4d} x{2:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d}'.format(self.w, self.h, self.c)

  def out_shape(self):
    return (self.batch, self.w, self.h, self.c)

  def forward(self, inpt):
    '''
    Simply store the input array

    Parameters:
      inpt: the input array
    '''
    if self.out_shape() != inpt.shape:
      raise ValueError('Forward Input layer. Incorrect input shape. Expected {} and given {}'.format(self.out_shape(), inpt.shape))

    self.output[:] = inpt

  def backward(self, delta):
    '''
    Simply pass the gradient

    Parameter:
      delta : global error to be backpropagated
    '''
    if self.out_shape() != delta.shape:
      raise ValueError('Forward Input layer. Incorrect delta shape. Expected {} and given {}'.format(self.out_shape(), delta.shape))

    self.delta[:] = delta


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

  layer = Input_layer(input_shape=inpt.shape)

  # FORWARD

  layer.forward(inpt)
  forward_out_byron = layer.output

  # BACKWARD

  delta = np.empty(shape=inpt.shape)
  layer.backward(delta, copy=True)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Input Layer')

  ax1.imshow(float_2_img(inpt))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(layer.output))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(delta)
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()