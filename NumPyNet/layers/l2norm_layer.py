#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'L2Normalization Layer'


class L2Norm_layer(object):

  def __init__(self, axis=-1, **kwargs):
    '''
    L2Norm layer
    '''
    self.axis = axis

    self.scales = None
    self.output, self.delta = (None, None)
    self._out_shape = None

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape
    return 'l2norm            {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(
           batch, out_width, out_height, out_channels)

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self._out_shape = previous_layer.out_shape
    return self

  @property
  def out_shape(self):
    return self._out_shape

  def forward(self, inpt):
    '''
    Forward of the l2norm layer, apply the l2 normalization over
    the input along the given axis

    Parameters:
      inpt: the input to be normaliza
    '''
    self._out_shape = inpt.shape

    norm = (inpt * inpt).sum(axis=self.axis, keepdims=True)
    norm = 1. / np.sqrt(norm + 1e-8)
    self.output = inpt * norm
    self.scales = (1. - self.output) * norm
    self.delta  = np.zeros(shape=self.out_shape, dtype=float)

  def backward(self, delta, copy=False):
    '''
    Compute the backward of the l2norm layer

    Parameter:
      delta : global error to be backpropagated
    '''

    self.delta += self.scales
    delta[:]   += self.delta


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

  # add batch = 1
  inpt = np.expand_dims(inpt, axis=0)

  layer = L2Norm_layer()

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output
  print(layer)

  # BACKWARD

  delta = np.zeros(shape=inpt.shape, dtype=float)
  layer.backward(delta, copy=True)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('L2Normalization Layer')

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()
