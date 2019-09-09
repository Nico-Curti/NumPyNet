#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Dropout Layer'

class Dropout_layer(object):

  def __init__(self, prob, **kwargs):
    '''
    DropOut Layer: drop a random selection of Inputs. This helps avoid overfitting

    Parameters:
      prob : float between 0. and 1., probability for every entry to be set to zero
    '''

    self.probability = prob
    if prob != 1.:
      self.scale = 1. / (1. - prob)
    else:
      self.scale = 1. # it doesn't matter anyway, since everything is zero


    self.output, self.delta = (None, None)
    self._out_shape = None

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape
    return 'Dropout       p = {:.2f}        {:4d}, {:4d}, {:4d}, {:4d}  ->  {:4d}, {:4d}, {:4d}, {:4d}'.format(
           self.probability,
           batch, out_width , out_height , out_channels,
           batch, out_width , out_height , out_channels)

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
    Forward function of the Dropout layer: it create a random mask for every input
      in the batch and set to zero the chosen values. Other pixels are scaled
      with the scale variable.

    Parameters :
      inpt : array of shape (batch, w, h, c), input of the layer
    '''

    self._out_shape = inpt.shape

    self.output = inpt.copy()

    self.rnd = np.random.uniform(low=0., high=1., size=self.output.shape) < self.probability
    self.output[self.rnd] = 0.
    self.rnd = ~self.rnd
    self.output[self.rnd] *= self.scale

  def backward(self, delta=None):
    '''
    Backward function of the Dropout layer: given the same mask as the layer
      it backprogates delta only to those pixel which values has not been set to zero
      in the forward.

    Parameters :
      delta : array of shape (batch, w, h, c), default value is None.
            If given, is the global delta to be backpropagated
    '''

    if delta is not None:
      delta[self.rnd] *= self.scale
      self.rnd = ~self.rnd
      delta[self.rnd] = 0.


if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  np.random.seed(123)

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)

  inpt = np.expand_dims(inpt, axis=0)

  prob = 0.1

  layer = Dropout_layer(prob)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output

  print(layer)

  # BACKWARD

  delta = np.ones(shape=inpt.shape, dtype=float)
  layer.delta = np.ones(shape=layer.out_shape, dtype=float)
  layer.backward(delta)

  # Visualitations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Dropout Layer\nDrop Probability : {}'.format(prob))
  # Shown first image of the batch
  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(layer.output[0]))
  ax2.set_title('Forward')
  ax2.axis('off')

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()




