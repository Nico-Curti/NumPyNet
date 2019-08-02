#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = "Softmax layer"


class Softmax_layer():

  def __init__(self, groups = 1, spatial = False, temperature = 1.):
    """
    Softmax layer: perfoms a Softmax transformation of its input

    Parameters:
      groups       : int, default is 1, indicates how many groups
        every images is divided into. Used only if spatial is False
      spatial      : boolean, default is False. if True performs the softmax
        computing max and sum over the entire image. if False max and sum are computed over
        the last axes (channels)
      temperature  : float, default is 1.. divide max and input in the softmax formulation.
    """

    self.batch, self.w,self.h,self.c = (0, 0, 0, 0)
    self.output, self.delta, self.loss  = (None,None,None)

    self.groups = groups
    self.spatial = spatial
    self.temperature = 1./temperature

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape()
    return 'softmax x entropy                            {:4d} x{:4d} x{:4d} x{:4d}'.format(
           batch, out_width, out_height, out_channels)


  def out_shape(self):
    return (self.batch, self.w, self.h, self.c)


  def forward(self, inpt, truth = None) :
    """
    Forward function of the Softmax Layer.

    Parameters:
      inpt  : array of shape (batch, w, h, c), input array
      truth : array of shape (batch, w, h, c), default is None, target vector.
        if a value is passed, the function compute the cross entropy cost
    """

    self.batch, self.w, self.h, self.c = inpt.shape

    if self.spatial:
      self.output = np.exp(inpt - inpt.max(axis = -1, keepdims=True))
      s = self.output.sum(axis = -1, keepdims = True)

    else : #groups is still fixed to 1
      self.output = np.exp((inpt - np.max(inpt, axis=(1,2,3), keepdims=True))*self.temperature)
      s = self.output.sum(axis=(1,2,3), keepdims=True)

    s = 1./s
    self.output *= s

#    value of delta with if truth is None
    self.delta = np.zeros(shape = self.out_shape())

    if truth is not None:
      out = self.output * (1. / self.output.sum())
      self.cost = -np.sum(truth*np.log(out))
#      Updateof delta given truth
      self.delta = truth-out

  def backward(self, delta=None):
    """
    Backward function of the Softmax Layer.

    Parameters:
      delta : array of shape (batch, w, h, c), default is None. If an array is passed,
        it's the global delta to be backpropagated
    """

#    s = (self.output * delta).sum()
#    delta[:] += self.temperature * self.output * (self.delta - s) #maybe output normalized

    if delta is not None:
      delta[:] += self.delta


if __name__ == "__main__":

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

  spatial     = False
  groups      = 1
  temperature = 1.

  if spatial:
    axis = -1
  else :
    axis = (1,2,3)

  np.random.seed(123)
#  inpt = np.random.uniform(low = 0., high = 1., size = (2,10,10,3))

  batch, w, h, c = inpt.shape

#  Ground truth definition, is random so don't expect much
  truth = np.random.choice([0., 1.], p = [.5,.5], size=(batch,w,h,c))

  layer = Softmax_layer(groups = 1, temperature = 1., spatial = spatial)

#  FORWARD

  layer.forward(inpt, truth)
  forward_out = layer.output
  layer_loss = layer.cost

  print(layer)
  print('Loss: {:.3f}'.format(layer_loss))

#  BACKWARD

  delta = np.zeros(shape = inpt.shape)
  layer.backward(delta)

  #Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle("SoftMax Layer \n\n loss : {:3.1f}, \n spatial : {}, temperature : {}, groups : {}".format(layer_loss, spatial, temperature, groups))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title("Original Image")
  ax1.axis("off")

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title("Backward")
  ax3.axis("off")

  plt.show()









