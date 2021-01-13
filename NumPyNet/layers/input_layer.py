#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.utils import check_is_fitted
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Input_layer(BaseLayer):
  '''
  Input layer, this layer can be used at the beginning of a Network to define all
  the model's input-output dimensions

  Parameters
  ----------
  input_shape : tuple
    Shape of the input in the format (batch, w, h, c).

  Example
  -------
  >>> import os
  >>>
  >>> import pylab as plt
  >>> from PIL import Image
  >>>
  >>> img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  >>> float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)
  >>>
  >>> filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  >>> inpt = np.asarray(Image.open(filename), dtype=float)
  >>> inpt.setflags(write=1)
  >>> inpt = img_2_float(inpt)
  >>> inpt = np.expand_dims(inpt, axis=0)
  >>>
  >>> layer = Input_layer(input_shape=inpt.shape)
  >>>
  >>> # FORWARD
  >>>
  >>> layer.forward(inpt)
  >>> forward_out_byron = layer.output
  >>>
  >>> # BACKWARD
  >>>
  >>> delta = np.zeros(shape=inpt.shape, dtype=float)
  >>> layer.backward(delta)
  >>>
  >>> # Visualizations
  >>>
  >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  >>> fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  >>>
  >>> fig.suptitle('Input Layer')
  >>>
  >>> ax1.imshow(float_2_img(inpt[0]))
  >>> ax1.set_title('Original image')
  >>> ax1.axis('off')
  >>>
  >>> ax2.imshow(float_2_img(layer.output[0]))
  >>> ax2.set_title("Forward")
  >>> ax2.axis("off")
  >>>
  >>> ax3.imshow(float_2_img(delta[0]))
  >>> ax3.set_title('Backward')
  >>> ax3.axis('off')
  >>>
  >>> fig.tight_layout()
  >>> plt.show()

  References
  ----------
  TODO
  '''

  def __init__(self, input_shape, **kwargs):

    if sum(np.shape(input_shape)) != 4:
      raise ValueError('Input layer error. Incorrect input_shape. Expected a 4D array (batch, width, height, channel). Given {}'.format(input_shape))

    super(Input_layer, self).__init__(input_shape=input_shape)

  def __str__(self):
    batch, w, h, c = self.input_shape
    return 'input                  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(batch, w, h, c)

  def forward(self, inpt):
    '''
    Forward function of the Input Layer: simply store the input array.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c)

    Return
    ------
      self
    '''

    self._check_dims(shape=self.out_shape, arr=inpt, func='Forward')

    self.output = inpt
    self.delta  = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta):
    '''
    Simply pass the gradient.

    Parameters
    ----------
      delta : array-like
        delta array of shape (batch, w, h, c). Global delta to be backpropagated.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')
    self._check_dims(shape=self.out_shape, arr=delta, func='Backward')

    delta[:] = self.delta

    return self


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
  inpt = np.expand_dims(inpt, axis=0)

  layer = Input_layer(input_shape=inpt.shape)

  # FORWARD

  layer.forward(inpt)
  forward_out_byron = layer.output

  # BACKWARD

  delta = np.zeros(shape=inpt.shape, dtype=float)
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Input Layer')

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(layer.output[0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
