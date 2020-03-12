#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Softmax_layer(BaseLayer):

  def __init__(self, input_shape=None, groups=1, spatial=False, temperature=1., **kwargs):
    '''
    Softmax layer: perfoms a Softmax transformation of its input

    Parameters
    ----------
      input_shape : tuple of 4 integers: input shape of the layer.
      groups       : int, default is 1, indicates how many groups
        every images is divided into. Used only if spatial is False
      spatial      : boolean, default is False. if True performs the softmax
        computing max and sum over the entire image. if False max and sum are computed over
        the last axes (channels)
      temperature  : float, default is 1.. divide max and input in the softmax formulation
        used only is spatial is False
    '''

    if isinstance(groups, int) and groups > 0:
       self.groups = groups
    else:
      raise ValueError('Softmax Layer : parameter "groups" must be an integer and > 0')

    self.spatial = spatial
    self.cost = 0.
    self.loss = None

    if temperature > 0:
      self.temperature = 1./temperature
    else :
      raise ValueError('Softmax Layer : parameter "temperature" must be > 0')

    super(Softmax_layer, self).__init__(input_shape=input_shape)

  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape
    return 'softmax x entropy                                   {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(batch, out_width, out_height, out_channels)

  def forward(self, inpt, truth=None) :
    '''
    Forward function of the Softmax Layer.

    Parameters
    ----------
      inpt  : numpy array of shape (batch, w, h, c), input array
      truth : numpyarray of shape (batch, w, h, c), default is None, target vector.
        if a value is passed, the function compute the cross entropy cost

    Returns
    -------
      Softmax layer object
    '''

    self._check_dims(shape=self.out_shape, arr=inpt, func='Forward')

    if self.spatial:
      self.output = np.exp(inpt - inpt.max(axis=-1, keepdims=True))
      s = 1. / self.output.sum(axis=-1, keepdims=True)
      self.output *= s

    else : # first implementation with groups, inspired from darknet, mhe
      self.output = np.empty(inpt.shape)
      inputs = np.prod(self.input_shape[1:])
      group_offset = inputs // self.groups
      flat_input = inpt.ravel()
      flat_outpt = self.output.ravel()
      for b in range(self.input_shape[0]):
        for g in range(self.groups):
          idx = b * inputs + g * group_offset
          inp = flat_input[idx : idx + group_offset]
          out = flat_outpt[idx : idx + group_offset]
          out[:]  = np.exp((inp - inp.max()) * self.temperature)
          out[:] *= 1. / out.sum()

      self.output = flat_outpt.reshape(inpt.shape)

      # Original implementation of spatial false
      # self.output = np.exp((inpt - np.max(inpt, axis=(1,2,3), keepdims=True)) * self.temperature)
      # s = self.output.sum(axis=(1,2,3), keepdims=True)

    # value of delta if truth is None
    # self.delta = np.zeros(shape=self.out_shape, dtype=float)

    if truth is not None:
      self._check_dims(shape=self.out_shape, arr=truth, func='Forward')
      out = self.output * (1. / self.output.sum())
      out = np.clip(out, 1e-8, 1. - 1e-8)
      self.cost = - np.sum(truth * np.log(out))
      self.delta = np.clip(self.output, 1e-8, 1. - 1e-8) - truth

    return self

  def backward(self, delta=None):
    '''
    Backward function of the Softmax Layer.

    Parameters
    ----------
      delta : array of shape (batch, w, h, c), default is None. If an array is passed,
        it's the global delta to be backpropagated

    Returns
    -------
     Softmax layer object
    '''

    check_is_fitted(self, 'output')
    self._check_dims(shape=self.out_shape, arr=delta, func='Backward')

    # This is an approximation
    if delta is not None:
      # print('In BACKWARD','\n', self.delta[0,0,0,:], '\n')
      delta[:] += self.delta
      # print('\nDELTA is', delta[0,0,0,:],'\n')

      ## darknet issue version
      # dot = (self.output * self.delta).sum(axis=(1, 2, 3), keepdims=True)
      # delta[:] += self.temperature * self.output * (self.delta - dot) # maybe output normalized

      ## softmax gradient formula
      # s = self.output.reshape(-1, 1)
      # delta[:] += np.diagflat(s) - np.dot(s, s.T)

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

  spatial     = True
  groups      = 4
  temperature = 1.5

  np.random.seed(123)
  batch, w, h, c = inpt.shape

  # truth definition, it's random so don't expect much
  truth = np.random.choice([0., 1.], p=[.5, .5], size=(batch, w, h, c))

  # Model initialization
  layer = Softmax_layer(input_shape=inpt.shape, groups=groups, temperature=temperature, spatial=spatial)

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

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10,5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle(('SoftMax Layer\n' +
               'loss : {:.3f}, \n' +
               'spatial : {}, temperature : {}, groups : {}').format(layer_loss, spatial, temperature, groups))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title('Forward')
  ax2.axis('off')

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
