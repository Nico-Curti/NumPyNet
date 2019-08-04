#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import functools
from enum import Enum

import numpy as np
#import utils     # not here yet

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Cost Layer'

# Enum of cost_function, declarations inside class
class cost_type(Enum):
  mse = 0 # mean square error
  masked = 1
  mae = 2 # mean absolute error
  seg = 3
  smooth = 4
  wgan = 5

class Cost_layer(object):

  def __init__(self, inputs, cost_type, scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0.):
    '''
    Cost layer, compute the cost of the output based on the selected cost function

    Parameters:
      inputs : tuple of int, shape of the input of the layer
      cost_type : cost function to be applied to the layer, from the enum cost_type
      scale     : float, default = 1.,
      ratio     : float, default = 0.,
      noobject_scale : flaot, default = 1.,
      threshold : float, default = 0.,
      smooothing: float, default = 0.,
    '''

    try:
      if len(inputs):
        self.outputs = np.prod(inputs)
    except:
      self.outputs = inputs

    self.cost_type = cost_type
    self.scale = scale
    self.ratio = ratio
    self.noobject_scale = noobject_scale
    self.threshold = threshold
    self.smoothing = smoothing

    # Need an empty initialization to work out _smooth_l1 and _wgan
    self.output = np.empty(shape=outputs)
    self.delta = np.empty(shape=outputs)


  def __str__(self):
    return 'cost                                           {:>4d}'.format(self.outputs)

  def out_shape(self):
    return (self.outputs)

  def forward(self, inpt, truth=None):
    '''
    Forward function for the cost layer. Using the chosen
    cost function, computes output, delta and cost.

    Parameters:
      inpt: the output of the previous layer.
      truth: truth values, it should have the same
        dimension as inpt.
    '''

    if truth is not None:

      if self.smoothing: self._smoothing(truth)                              # smooth is applied on truth

      if   self.cost_type == cost_type.smooth: self._smooth_l1(inpt, truth)  # smooth_l1 if smooth not zero
      elif self.cost_type == cost_type.mae:    self._l1(inpt, truth)         # call for l1 if mae is cost
      elif self.cost_type == cost_type.wgan:   self._wgan(inpt, truth)       # call for wgan
      else:                                    self._l2(inpt, truth)         # call for l2 if mse or nothing

      if self.cost_type == cost_type.seg and self.noobject_scale != 1.:      # seg if noobject_scale is not 1.
        self._seg(truth)

      if self.cost_type == cost_type.masked:                                 # l2 Masked truth values if selected
        self._masked(inpt, truth)

      if self.ratio:                                                         #
        self._ratio()

      if self.threshold:                                                     #
        self._threshold()


      norm = 1. / len(self.delta)                                            # normalization of delta!
      self.delta *= norm

      self.cost = np.mean(self.output)                                       # compute the cost



  def backward(self, delta):
    '''
    Backward function of the cost_layer, it updates the delta
    variable to be backpropagated. self.delta is updated inside the cost function.

    Parameters:
      delta: array, error of the network, to be backpropagated
    '''
    delta += self.scale * self.delta

  def _smoothing(self, truth):
    '''
    _smoothing function :

    Parameters:
      truth : array, truth values
    '''

    scale = 1. / self.smoothing
    bias  = self.smoothing / self.outputs

    truth = truth * scale + bias

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
    self.output[ mask_index ] = diff[ mask_index ] * diff[ mask_index ]
    self.delta [ mask_index ] = diff[ mask_index ]

    mask_index = ~mask_index
    self.output[ mask_index ] = 2. * abs_diff[ mask_index ] - 1.
    self.delta [ mask_index ] = - np.sign(diff[ mask_index ])

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

    self.output = np.abs(diff)
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
#    mask_index = truth[ truth != 0 ]
    self.output[mask_index] = -inpt[mask_index]
    mask_index = ~mask_index
    self.output[mask_index] =  inpt[mask_index]

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

    self.output = diff * diff
    self.delta = -2. * diff

  def _seg(self, truth):
    '''
     _seg function, where truth is zero, scale output and delta for noobject_scale

     Paramters:
       truth : array of truth values
    '''

    mask_index = truth == 0.

    self.output[mask_index] *= self.noobject_scale
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
    inpt[truth == utils.SECRET_NUM] = 0.

  def _ratio(self, truth):
    '''
    _ratio function: called if self.ratio is not zero.

    Parameters :
      truth : array, truth values
    '''
    abs_compare = lambda x, y: ( abs(x) > abs(y) ) - ( abs(x) < abs(y) )
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
    scale = self.threshold / self.outputs
    scale *= scale

    self.delta[ self.delta * self.delta < scale ] = 0.


if __name__ == '__main__':

  np.random.seed(123)

  outputs = 100

  # Random input and truth values
  truth = np.random.uniform(low=0., high=1., size=(outputs,))
  inpt = np.random.uniform(low=0., high=1., size=(outputs,))

  # select cost type
  cost = cost_type.mse

  # Model initialization
  layer = Cost_layer(inpt.size, cost, scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0.)

  # FORWARD

  layer.forward(inpt, truth)
  byron_loss = layer.cost

  print(layer)
  print(layer.out_shape())
  print('Byron loss: {:.3f}'.format(byron_loss))

  # BACKWARD

  delta = np.zeros(shape=inpt.shape)
  layer.backward(delta)

