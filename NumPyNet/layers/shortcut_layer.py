#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted

import numpy as np
from itertools import product
from NumPyNet.exception import LayerError
from NumPyNet.layers.base import BaseLayer

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Shortcut_layer(BaseLayer):
  '''
  Shortcut layer: activation of the linear combination of the output of two layers

              layer1 * alpha + layer2 * beta = output

  Now working only with same shapes input

  Parameters
  ----------
    activation : str or Activation object
      Activation function of the layer.

    alpha : float, (default = 1.)
      first weight of the combination.

    beta : float, (default = 1.)
      second weight of the combination.

  Examples
  --------
  >>> import pylab as plt
  >>>
  >>> from NumPyNet import activations
  >>>
  >>> img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  >>> float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)
  >>>
  >>> # Set seed to have same input
  >>> np.random.seed(123)
  >>>
  >>> layer_activ = activations.Relu()
  >>>
  >>> batch = 2
  >>>
  >>> alpha = 0.75
  >>> beta  = 0.5
  >>>
  >>> # Random input
  >>> inpt1      = np.random.uniform(low=-1., high=1., size=(batch, 100, 100, 3))
  >>> inpt2      = np.random.uniform(low=-1., high=1., size=inpt1.shape)
  >>> b, w, h, c = inpt1.shape
  >>>
  >>>
  >>> # model initialization
  >>> layer = Shortcut_layer(activation=layer_activ,
  >>>                        alpha=alpha, beta=beta)
  >>>
  >>> # FORWARD
  >>>
  >>> layer.forward(inpt1, inpt2)
  >>> forward_out = layer.output.copy()
  >>>
  >>> print(layer)
  >>>
  >>> # BACKWARD
  >>>
  >>> delta      = np.zeros(shape=inpt1.shape, dtype=float)
  >>> delta_prev = np.zeros(shape=inpt2.shape, dtype=float)
  >>>
  >>> layer.delta = np.ones(shape=layer.out_shape, dtype=float)
  >>> layer.backward(delta, delta_prev)
  >>>
  >>> # Visualizations
  >>>
  >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  >>> fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  >>> fig.suptitle('Shortcut Layer\nalpha : {}, beta : {}, activation : {} '.format(alpha, beta, layer_activ.name))
  >>>
  >>> ax1.imshow(float_2_img(inpt1[0]))
  >>> ax1.set_title('Original Image')
  >>> ax1.axis('off')
  >>>
  >>> ax2.imshow(float_2_img(forward_out[0]))
  >>> ax2.set_title('Forward')
  >>> ax2.axis('off')
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

  def __init__(self, activation=Activations, alpha=1., beta=1., **kwargs):

    activation = _check_activation(self, activation)

    self.activation = activation.activate
    self.gradient = activation.gradient

    self.alpha, self.beta = alpha, beta

    self.ix, self.jx, self.kx = (None, ) * 3
    self.iy, self.jy, self.ky = (None, ) * 3

    super(Shortcut_layer, self).__init__()

  def __str__(self):
    (b1, w1, h1, c1), (b2, w2, h2, c2) = self.input_shape
    return 'res                    {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d} x{:>4d}'.format(b2, w2, h2, c2, b1, w1, h1, c1)

  def __call__(self, previous_layer):

    prev1, prev2 = previous_layer

    if prev1.out_shape is None or prev2.out_shape is None:
      class_name = self.__class__.__name__
      prev_name = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.input_shape = [prev1.out_shape, prev2.out_shape]

    self._stride_index(prev1.out_shape, prev2.out_shape)

    return self

  @property
  def out_shape(self):
    return max(self.input_shape)

  def _stride_index(self, shape1, shape2):
    '''
    Evaluate the strided indexes if the input shapes are different
    '''

    _, w2, h2, c2 = shape1
    _, w1, h1, c1 = shape2
    stride = w1 // w2
    sample = w2 // w1
    stride = stride if stride > 0 else 1
    sample = sample if sample > 0 else 1

    if not (stride == h1 // h2 and sample == h2 // h1):
      class_name = self.__class__.__name__
      prev_name = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    idx = product(range(0, w2, sample), range(0, h2, sample), range(0, c2, sample))
    self.ix, self.jx, self.kx = zip(*idx)

    idx = product(range(0, w1, stride), range(0, w1, stride), range(0, c1, stride))
    self.iy, self.jy, self.ky = zip(*idx)

  def forward(self, inpt, prev_output, copy=False):
    '''
    Forward function of the Shortcut layer: activation of the linear combination between input.

    Parameters
    ----------
      inpt : array-like
        Input batch of images in format (batch, in_w, in_h, in _c)

      prev_output : array-like
        second input of the layer

    Returns
    -------
      self
    '''
    # assert inpt.shape == prev_output.shape
    # TODO: find a better solution to initialize the input shape in the constructor

    self.input_shape = [inpt.shape, prev_output.shape]

    if inpt.shape == prev_output.shape:
      self.output = self.alpha * inpt[:] + self.beta * prev_output[:]

    else:

      # If the layer are combined the smaller one is distributed according to the
      # sample stride
      # Example:
      #
      # inpt = [[1, 1, 1, 1],        prev_output = [[1, 1],
      #         [1, 1, 1, 1],                       [1, 1]]
      #         [1, 1, 1, 1],
      #         [1, 1, 1, 1]]
      #
      # output = [[2, 1, 2, 1],
      #           [1, 1, 1, 1],
      #           [2, 1, 2, 1],
      #           [1, 1, 1, 1]]

      # TODO: Not working with different dimensions
      if (self.ix, self.jx, self.kx) == (None, None, None):
        self._stride_index(inpt.shape, prev_output.shape)

      self.output = inpt.copy()
      self.output[:, self.ix, self.jx, self.kx] = self.alpha * self.output[:, self.ix, self.jx, self.kx] + self.beta * prev_output[:, self.iy, self.jy, self.ky]

    self.output = self.activation(self.output, copy=copy)
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta, prev_delta, copy=False):
    '''
    Backward function of the Shortcut layer

    Parameters
    ----------
      delta : array-like
        delta array of shape (batch, w, h, c). Global delta to be backpropagated.

      delta_prev : array-like
        second delta to be backporpagated.

      copy : bool (default=False)
        States if the activation function have to return a copy of the input or not.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')

    # derivatives of the activation funtion w.r.t. to input
    self.delta *= self.gradient(self.output, copy=copy)

    delta[:]   += self.delta * self.alpha

    if (self.ix, self.iy, self.kx) == (None, None, None): # same shapes
      prev_delta[:] += self.delta[:] * self.beta

    else: # different shapes
      prev_delta[:, self.ix, self.jx, self.kx] += self.beta * self.delta[:, self.iy, self.jy, self.ky]

    return self

if __name__ == '__main__':

  import pylab as plt

  from NumPyNet import activations

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  # Set seed to have same input
  np.random.seed(123)

  layer_activ = activations.Relu()

  batch = 2

  alpha = 0.75
  beta  = 0.5

  # Random input
  inpt1      = np.random.uniform(low=-1., high=1., size=(batch, 100, 100, 3))
  inpt2      = np.random.uniform(low=-1., high=1., size=inpt1.shape)
  b, w, h, c = inpt1.shape


  # model initialization
  layer = Shortcut_layer(activation=layer_activ,
                         alpha=alpha, beta=beta)

  # FORWARD

  layer.forward(inpt1, inpt2)
  forward_out = layer.output.copy()

  print(layer)

  # BACKWARD

  delta      = np.zeros(shape=inpt1.shape, dtype=float)
  delta_prev = np.zeros(shape=inpt2.shape, dtype=float)

  layer.delta = np.ones(shape=layer.out_shape, dtype=float)
  layer.backward(delta, delta_prev)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('Shortcut Layer\nalpha : {}, beta : {}, activation : {} '.format(alpha, beta, layer_activ.name))

  ax1.imshow(float_2_img(inpt1[0]))
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
