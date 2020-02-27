#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import functools
from enum import Enum

import numpy as np
from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


# Enum of cost_function, declarations inside class
class cost_type(Enum):
  mse = 0 # mean square error
  masked = 1
  mae = 2 # mean absolute error
  seg = 3
  smooth = 4
  wgan = 5
  hellinger = 6
  hinge = 7
  logcosh = 8

class Cost_layer(object):

  SECRET_NUM = 12345

  def __init__(self, cost_type, input_shape=None, scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0., **kwargs):
    '''
    Cost layer, compute the cost of the output based on the selected cost function

    Parameters:
      input_shape : tuple of int, shape of the input of the layer
      cost_type : cost function to be applied to the layer, from the enum cost_type
      scale     : float, default = 1.,
      ratio     : float, default = 0.,
      noobject_scale : float, default = 1.,
      threshold : float, default = 0.,
      smooothing: float, default = 0.,
    '''

    self.cost = 0.
    self.cost_type = cost_type
    self.scale = scale
    self.ratio = ratio
    self.noobject_scale = noobject_scale
    self.threshold = threshold
    self.smoothing = smoothing

    self._out_shape = input_shape
    # Need an empty initialization to work out _smooth_l1 and _wgan
    self._output = np.empty(shape=self._out_shape)
    self.output = None
    self.delta  = None # np.empty(shape=self._out_shape)

  def __str__(self):
    return 'cost                   {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(*self.out_shape)

  def __call__(self, previous_layer):

    self._out_shape = previous_layer.out_shape

    if previous_layer.out_shape is None or self.out_shape != previous_layer.out_shape:
      class_name = self.__class__.__name__
      prev_name  = previous_layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    return self

  @property
  def out_shape(self):
    return self._out_shape

  def forward(self, inpt, truth=None):
    '''
    Forward function for the cost layer. Using the chosen
    cost function, computes output, delta and cost.

    Parameters:
      inpt: the output of the previous layer.
      truth: truth values, it should have the same
        dimension as inpt.
    '''
    self.delta  = np.empty(shape=self._out_shape)
    self._out_shape = inpt.shape
    self.output = inpt[:]

    if truth is not None:

      if self.smoothing: self._smoothing(truth)                              # smooth is applied on truth

      if   self.cost_type == cost_type.smooth:    self._smooth_l1(inpt, truth)  # smooth_l1 if smooth not zero
      elif self.cost_type == cost_type.mae:       self._l1(inpt, truth)         # call for l1 if mae is cost
      elif self.cost_type == cost_type.wgan:      self._wgan(inpt, truth)       # call for wgan
      elif self.cost_type == cost_type.hellinger: self._hellinger(inpt, truth)  # call for hellinger distance
      elif self.cost_type == cost_type.hinge:     self._hinge(inpt, truth)      # call for hellinger distance
      elif self.cost_type == cost_type.logcosh:   self._logcosh(inpt, truth)    # call for hellinger distance
      else:                                       self._l2(inpt, truth)         # call for l2 if mse or nothing

      if self.cost_type == cost_type.seg and self.noobject_scale != 1.:      # seg if noobject_scale is not 1.
        self._seg(truth)

      if self.cost_type == cost_type.masked:                                 # l2 Masked truth values if selected
        self._masked(inpt, truth)

      if self.ratio:                                                         #
        self._ratio(truth)

      if self.threshold:                                                     #
        self._threshold()


      norm = 1. / self.delta.size                                            # normalization of delta!
      self.delta *= norm

      self.cost = np.mean(self._output)                                      # compute the cost

    return self

  def backward(self, delta):
    '''
    Backward function of the cost_layer, it updates the delta
    variable to be backpropagated. self.delta is updated inside the cost function.

    Parameters:
      delta: array, error of the network, to be backpropagated
    '''
    check_is_fitted(self, 'delta')

    delta[:] += self.scale * self.delta

    return self

  def _smoothing(self, truth):
    '''
    _smoothing function :

    Parameters:
      truth : array, truth values
    '''

    scale = 1. - self.smoothing
    bias  = self.smoothing / self._output.size

    truth[:] = truth * scale + bias

  def _smooth_l1(self, inpt, truth):
    '''
    _smooth_l1 function:

    Parameters:
      inpt : array output of the network
      truth : array, truth values
    '''

    diff = inpt - truth
    abs_diff = np.abs(diff)

    mask_index = abs_diff < 1.
    self._output[ mask_index ] = diff[ mask_index ] * diff[ mask_index ]
    self.delta  [ mask_index ] = diff[ mask_index ]

    mask_index = ~mask_index
    self._output[ mask_index ] = 2. * abs_diff[ mask_index ] - 1.
    self.delta  [ mask_index ] = - np.sign(diff[ mask_index ])

  def _l1(self, inpt, truth):
    '''
    cost function for the l1 norm of the ouput.
    It computes the absolute difference between truth and inpt and
    updates output and delta. Called for mae cost_type.

    Parameters:
      inpt  : array output of the network
      truth : array, truth values for comparisons
    '''

    diff = truth - inpt

    self._output = np.abs(diff)
    self.delta = -np.sign(diff)

  def _wgan(self, inpt, truth):
    '''
    _wgan function : where truth is not 0, the output is the inverted input. Input
    is forwarded as it is otherwise.

    Parameters:
      inpt  : array output of the network
      truth : array, truth values
    '''
    mask_index = truth != 0
    # mask_index = truth[ truth != 0 ]
    self._output[mask_index] = -inpt[mask_index]
    mask_index = ~mask_index
    self._output[mask_index] =  inpt[mask_index]

    self.delta = np.sign(truth)


  def _l2(self, inpt, truth):
    '''
    cost function for the l2 norm.
    It computes the square difference (truth -  inpt)**2
    and modifies output and delta. Called for mse cost_type.

    Parameters:
      inpt: output of the previous layer of the network
      truth: truth values.
    '''

    diff = truth - inpt

    self._output = diff * diff
    self.delta  = -2. * diff

  def _hinge(self, inpt, truth):
    '''
    cost function for the Hinge loss.
    The gradient is computed as the smoothed version of Rennie and Srebro

    Parameters:
      inpt: output of the previous layer of the network
      truth: truth values.
    '''
    diff = truth * inpt
    self._output = np.maximum(0, 1. - diff)
    self.delta  = diff
    check1 = np.vectorize(lambda t:   t <= 0.)
    check2 = np.vectorize(lambda t: ( t >  0.) and ( t <= 1.))
    check3 = np.vectorize(lambda t:   t >= 1.)
    self.delta[check1(diff)] = .5 - diff[check1(diff)]
    self.delta[check2(diff)] = .5 * (1. - diff[check2(diff)]**2)
    self.delta[check3(diff)] = 0.

  def _hellinger(self, inpt, truth):
    '''
    cost function for the Hellinger distance.
    It computes the square difference (sqrt(truth) -  sqrt(inpt))**2
    and modifies output and delta. Called for hellinger cost_type.

    Parameters:
      inpt: output of the previous layer of the network
      truth: truth values.
    '''
    diff = np.sqrt(truth) - np.sqrt(inpt)
    self._output = diff * diff
    self.delta  = -diff / np.sqrt(2 * inpt)

  def _logcosh(self, inpt, truth):
    '''
    cost function for the Log-Cosh.

    Parameters:
      inpt: output of the previous layer of the network
      truth: truth values.
    '''
    diff = truth - inpt
    self._output = np.log(np.cosh(diff))
    self.delta  = np.tanh(-diff)

  def _seg(self, truth):
    '''
     _seg function, where truth is zero, scale output and delta for noobject_scale

     Paramters:
       truth : array of truth values
    '''

    mask_index = truth == 0.

    self._output[mask_index] *= self.noobject_scale
    self.delta[ mask_index] *= self.noobject_scale

  def _masked(self, inpt, truth):
    '''
    _masked function : set to zero the part of the input where the condition is true.
     used to ignore certain classes

    Parameters:
      inpt  : array output of the network
      truth : array, truth values
    '''
    # utils is not here yet
    inpt[truth == self.SECRET_NUM] = 0.

  def _ratio(self, truth):
    '''
    _ratio function: called if self.ratio is not zero.

    Parameters :
      truth : array, truth values
    '''
    abs_compare = lambda x, y: (abs(x) > abs(y)) ^ (abs(x) < abs(y))
    compare = functools.cmp_to_key(abs_compare)

    self.delta = sorted(self.delta, key=compare)

    #index = int(1. - self.ratio) * len(delta)
    thr = 0 # np.abs(self.delta[index])

    self.delta[ self.delta * self.delta < thr ] = 0.

  def _threshold(self):
    '''
    _threshold function: set a global threshold to delta

    Parameters :
    '''
    scale = self.threshold / self._output.size
    scale *= scale

    self.delta[ self.delta * self.delta < scale ] = 0.


if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  input = np.asarray(Image.open(filename), dtype=float)
  input.setflags(write=1)
  input = img_2_float(input)

  # batch == 1
  input = np.expand_dims(input, axis=0)

  cost_type = cost_type.mse
  scale = 1.
  ratio = 0.
  noobject_scale = 1.
  threshold = 0.
  smoothing = 0.

  truth = np.random.uniform(low=0., high=1., size=input.shape)

  layer = Cost_layer(input_shape=input.shape, cost_type=cost_type, scale=scale, ratio=ratio, noobject_scale=noobject_scale, threshold=threshold, smoothing=smoothing, trainable=True)
  print(layer)

  layer.forward(input, truth)
  forward_out = layer.output

  print('Cost: {:.3f}'.format(layer.cost))

  delta = np.zeros(shape=input.shape, dtype=float)
  layer.backward(delta)

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Cost Layer:\n{0}'.format(cost_type))

  ax1.imshow(float_2_img(input[0]))
  ax1.axis('off')
  ax1.set_title('Original Image')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.axis('off')
  ax2.set_title('Forward Image')

  ax3.imshow(float_2_img(delta[0]))
  ax3.axis('off')
  ax3.set_title('Delta Image')

  fig.tight_layout()
  plt.show()
