#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted

import numpy as np
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Activation_layer(BaseLayer):

  def __init__(self, input_shape=None, activation=Activations, **kwargs):
    '''
    Activation layer

    Parameters
    ----------
      input_shape : tuple of 4 integers: input shape of the layer.
      activation : activation function of the layer.
    '''

    activation = _check_activation(self, activation)

    self.activation = activation.activate
    self.gradient = activation.gradient

    super(Activation_layer, self).__init__(input_shape=input_shape)

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape
    return 'activ                  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(
           batch, out_width, out_height, out_channels)

  def forward(self, inpt, copy=True):
    '''
    Forward of the activation layer, apply the selected activation function to
    the input.

    Parameters
    ----------
      inpt: numpy array. Input images to be activated.
      copy: bolean, default True. If True make a copy of the input before
            applying the activation.

    Returns
    ----------
      activation layer object.
    '''

    self._check_dims(shape=self.out_shape, arr=inpt, func='Forward')

    self.output = self.activation(inpt, copy=copy)
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta, copy=False):
    '''
    Compute the backward of the activation layer

    Parameter
    ----------
      delta : global error to be backpropagated

    Returns
    ----------
      activation layer object.
    '''

    check_is_fitted(self, 'delta')
    self._check_dims(shape=self.out_shape, arr=delta, func='Backward')

    self.delta *= self.gradient(self.output, copy=copy)
    delta[:] = self.delta

    return self


if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  from NumPyNet import activations

  activation_func = activations.Hardtan()

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)
  # Relu activation constrain
  inpt = inpt * 2 - 1

  # add batch = 1
  inpt = np.expand_dims(inpt, axis=0)

  layer = Activation_layer(input_shape=inpt.shape, activation=activation_func)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output
  print(layer)

  # BACKWARD

  layer.delta = np.ones(shape=inpt.shape, dtype=float)
  delta = np.zeros(shape=inpt.shape, dtype=float)
  layer.backward(delta, copy=True)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Activation Layer\nfunction : {}'.format(activation_func.name))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
