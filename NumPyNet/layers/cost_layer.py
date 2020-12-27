#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import functools

import numpy as np
from NumPyNet.utils import check_is_fitted
from NumPyNet.utils import cost_type
from NumPyNet.utils import _check_cost
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Cost_layer(BaseLayer):
  '''
  Cost layer

  Compute the cost of the output based on the selected cost function.

  Parameters:
    input_shape : tuple (default=None)
      Shape of the input in the format (batch, w, h, c), None is used when the layer is part of a Network model.

    cost_type : cost_type or str
      Cost function to be applied to the layer, from the enum cost_type.

    scale : float (default=1.)

    ratio : float (default=0.)

    noobject_scale : float (default=1)

    threshold : float (default=0.)

    smooothing: float (default=0.)

  Example
  -------
  >>> import os
  >>>
  >>> import numpy as np
  >>> import pylab as plt
  >>> from PIL import Image
  >>>
  >>> img_2_float = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 1.)).astype(float)
  >>> float_2_img = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 255.)).astype(np.uint8)
  >>>
  >>> filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  >>> inpt = np.asarray(Image.open(filename), dtype=float)
  >>> inpt.setflags(write=1)
  >>> inpt = img_2_float(inpt)
  >>>
  >>> # batch == 1
  >>> inpt = np.expand_dims(inpt, axis=0)
  >>>
  >>> cost_type = cost_type.mse
  >>> scale = 1.
  >>> ratio = 0.
  >>> noobject_scale = 1.
  >>> threshold = 0.
  >>> smoothing = 0.
  >>>
  >>> truth = np.random.uniform(low=0., high=1., size=inpt.shape)
  >>>
  >>> layer = Cost_layer(input_shape=inpt.shape,
  >>>                    cost_type=cost_type, scale=scale,
  >>>                    ratio=ratio,
  >>>                    noobject_scale=noobject_scale,
  >>>                    threshold=threshold,
  >>>                    smoothing=smoothing,
  >>>                    trainable=True)
  >>> print(layer)
  >>>
  >>> layer.forward(inpt, truth)
  >>> forward_out = layer.output
  >>>
  >>> print('Cost: {:.3f}'.format(layer.cost))

  References
  ----------
    - TODO
  '''

  SECRET_NUM = 12345

  def __init__(self, cost_type, input_shape=None, scale=1., ratio=0., noobject_scale=1., threshold=0., smoothing=0., **kwargs):

    self.cost = 0.
    self.cost_type = _check_cost(self, cost_type)
    self.scale = scale
    self.ratio = ratio
    self.noobject_scale = noobject_scale
    self.threshold = threshold
    self.smoothing = smoothing

    # Need an empty initialization to work out _smooth_l1 and _wgan
    super(Cost_layer, self).__init__(input_shape=input_shape)
    self.loss = np.empty(shape=self.out_shape)

  def __str__(self):
    '''
    PRINTER
    '''
    return 'cost                   {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}   ->  {0:>4d} x{1:>4d} x{2:>4d} x{3:>4d}'.format(*self.out_shape)

  def forward(self, inpt, truth=None):
    '''
    Forward function for the cost layer. Using the chosen
    cost function, computes output, delta and cost.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth: array-like
        truth values, it must have the same dimension as inpt.

    Returns
    -------
      self
    '''
    self._check_dims(shape=self.input_shape, arr=inpt, func='Forward')

    self.delta = np.empty(shape=self.out_shape)
    self.output = inpt[:]

    if truth is not None:

      if self.smoothing: truth = self._smoothing(truth)                         # smooth is applied on truth

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

      if self.ratio:
        self._ratio(truth)

      if self.threshold:
        self._threshold()

      norm = 1. / self.delta.size                                            # normalization of delta!
      self.delta *= norm

      self.cost = np.mean(self.loss)                                         # compute the cost

    return self

  def backward(self, delta):
    '''
    Backward function of the cost_layer, it updates the delta
    variable to be backpropagated. `self.delta` is updated inside the cost function.

    Parameters
    ----------
      delta : array-like
        delta array of shape (batch, w, h, c). Global delta to be backpropagated.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')
    self._check_dims(shape=self.input_shape, arr=delta, func='Backward')

    delta[:] += self.scale * self.delta

    return self

  def _smoothing(self, truth):
    '''
    _smoothing function

    Parameters
    ----------
      truth: array-like
        truth values, it must have the same dimension as inpt.

    Returns
    -------
      array-like
        smoothed values of the input array
    '''

    scale = 1. - self.smoothing
    bias = self.smoothing / np.prod(self.loss.shape[1:])

    return truth * scale + bias

  def _smooth_l1(self, inpt, truth):
    '''
    _smooth_l1 cost function

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''

    diff = inpt - truth
    abs_diff = np.abs(diff)

    mask_index = abs_diff < 1.
    self.loss[mask_index] = diff[mask_index] * diff[mask_index]
    self.delta[mask_index] = diff[mask_index]

    mask_index = ~mask_index
    self.loss[mask_index] = 2. * abs_diff[mask_index] - 1.
    self.delta[mask_index] = - np.sign(diff[mask_index])

  def _l1(self, inpt, truth):
    '''
    cost function for the l1 norm of the ouput.
    It computes the absolute difference between truth and inpt and
    updates output and delta. Called for mae cost_type.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''

    diff = truth - inpt

    self.loss = np.abs(diff)
    self.delta = -np.sign(diff)

  def _wgan(self, inpt, truth):
    '''
    wgan cost function: where truth is not 0, the output is the inverted input. Input
    is forwarded as it is otherwise.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''
    mask_index = truth != 0
    # mask_index = truth[ truth != 0 ]
    self.loss[mask_index] = -inpt[mask_index]
    mask_index = ~mask_index
    self.loss[mask_index] = inpt[mask_index]

    self.delta = np.sign(truth)

  def _l2(self, inpt, truth):
    '''
    Cost function for the l2 norm.
    It computes the square difference (truth -  inpt)**2
    and modifies output and delta. Called for mse cost_type.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''

    diff = truth - inpt

    self.loss = diff * diff
    self.delta = -2. * diff

  def _hinge(self, inpt, truth):
    '''
    Cost function for the Hinge loss.
    The gradient is computed as the smoothed version of Rennie and Srebro

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''
    diff = truth * inpt
    self.loss = np.maximum(0, 1. - diff)
    self.delta = diff
    check1 = np.vectorize(lambda t: t <= 0.)
    check2 = np.vectorize(lambda t: (t > 0.) and (t <= 1.))
    check3 = np.vectorize(lambda t: t >= 1.)
    self.delta[check1(diff)] = .5 - diff[check1(diff)]
    self.delta[check2(diff)] = .5 * (1. - diff[check2(diff)]**2)
    self.delta[check3(diff)] = 0.

  def _hellinger(self, inpt, truth):
    '''
    cost function for the Hellinger distance.
    It computes the square difference (sqrt(truth) -  sqrt(inpt))**2
    and modifies output and delta. Called for hellinger cost_type.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''
    diff = np.sqrt(truth) - np.sqrt(inpt)
    self.loss = diff * diff
    self.delta = -diff / np.sqrt(2 * inpt)

  def _logcosh(self, inpt, truth):
    '''
    Cost function for the Log-Cosh.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''
    diff = truth - inpt
    self.loss = np.log(np.cosh(diff))
    self.delta = np.tanh(-diff)

  def _seg(self, truth):
    '''
     _seg function, where truth is zero, scale output and delta for noobject_scale

    Parameters
    ----------
      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''

    mask_index = truth == 0.

    self.loss[mask_index] *= self.noobject_scale
    self.delta[mask_index] *= self.noobject_scale

  def _masked(self, inpt, truth):
    '''
    _masked function: set to zero the part of the input where the condition is true.
     used to ignore certain classes

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''
    # utils is not here yet
    inpt[truth == self.SECRET_NUM] = 0.

  def _ratio(self, truth):
    '''
    _ratio function: called if self.ratio is not zero.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c).

      truth : array-like
        truth values, it must have the same dimension as `inpt`
    '''

    compare = functools.cmp_to_key(lambda x, y: (abs(x) > abs(y)) ^ (abs(x) < abs(y)))

    ordered = sorted(self.delta.ravel(), key=compare)
    self.delta = np.asarray(ordered).reshape(self.delta.shape)

    # index = int(1. - self.ratio) * len(delta)
    thr = 0  # np.abs(self.delta[index])

    self.delta[(self.delta * self.delta) < thr] = 0.

  def _threshold(self):
    '''
    _threshold function: set a global threshold to delta
    '''
    scale = self.threshold / self.loss.size
    scale *= scale

    self.delta[(self.delta * self.delta) < scale] = 0.


if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1. / (im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)

  # batch == 1
  inpt = np.expand_dims(inpt, axis=0)

  cost_type = cost_type.mse
  scale = 1.
  ratio = 0.
  noobject_scale = 1.
  threshold = 0.
  smoothing = 0.

  truth = np.random.uniform(low=0., high=1., size=inpt.shape)

  layer = Cost_layer(input_shape=inpt.shape,
                     cost_type=cost_type, scale=scale,
                     ratio=ratio,
                     noobject_scale=noobject_scale,
                     threshold=threshold,
                     smoothing=smoothing,
                     trainable=True)
  print(layer)

  layer.forward(inpt, truth)
  forward_out = layer.output

  print('Cost: {:.3f}'.format(layer.cost))

  delta = np.zeros(shape=inpt.shape, dtype=float)
  layer.backward(delta)

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Cost Layer:\n{0}'.format(cost_type))

  ax1.imshow(float_2_img(inpt[0]))
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
