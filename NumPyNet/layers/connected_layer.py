#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.activations import Activations
from NumPyNet.utils import _check_activation
from NumPyNet.utils import check_is_fitted
from NumPyNet.layers.base import BaseLayer

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Connected_layer(BaseLayer):
  '''
  Connected layer

  It's the equivalent of a Dense layer in keras, or a single layer
  of an MLP in scikit-learn

  Parameters
  ----------
    outputs : int
      Number of outputs of the layers. It's also the number of Neurons of the layer.

    activation : str or Activation object
      Activation function of the layer.

    input_shape : tuple (default=None)
      Shape of the input in the format (batch, w, h, c), None is used when the layer is part of a Network model.

    weights : array-like (default=None)
      Array of shape (w * h * c, outputs), default is None. Weights of the dense layer.
      If None, weights initialization is random and follows a uniform distribution in the range [-scale, scale] where:

        scale = sqrt(2 / (w * h* c))

    bias : array-like (default=None)
      Array of shape (outputs,). Bias of the fully-connected layer.
      If None, bias inititialization is zeros

  Example
  -------
  >>> import os
  >>>
  >>> import numpy as np
  >>> import matplotlib.pyplot as plt
  >>> from PIL import Image
  >>>
  >>> from NumPyNet import activations
  >>>
  >>> img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  >>> float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)
  >>>
  >>> filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  >>> inpt = np.asarray(Image.open(filename), dtype=float)
  >>> inpt.setflags(write=1)
  >>> inpt = img_2_float(inpt)
  >>>
  >>> # from (w, h, c) to shape (1, w, h, c)
  >>> inpt = np.expand_dims(inpt, axis=0)  # just to add the 'batch' dimension
  >>>
  >>> # Number of outputs
  >>> outputs = 10
  >>> layer_activation = activations.Relu()
  >>> batch, w, h, c = inpt.shape
  >>>
  >>> # Random initialization of weights with shape (w * h * c) and bias with shape (outputs,)
  >>> np.random.seed(123)  # only if one want always the same set of weights
  >>> weights = np.random.uniform(low=-1., high=1., size=(np.prod(inpt.shape[1:]), outputs))
  >>> bias = np.random.uniform(low=-1., high=1., size=(outputs,))
  >>>
  >>> # Model initialization
  >>> layer = Connected_layer(outputs, input_shape=inpt.shape,
  >>>                         activation=layer_activation, weights=weights, bias=bias)
  >>> print(layer)
  >>>
  >>> # FORWARD
  >>>
  >>> layer.forward(inpt)
  >>> forward_out = layer.output.copy()
  >>>
  >>> # BACKWARD
  >>>
  >>> layer.delta = np.ones(shape=(layer.out_shape), dtype=float)
  >>> delta = np.zeros(shape=(batch, w, h, c), dtype=float)
  >>> layer.backward(inpt, delta=delta, copy=True)
  >>>
  >>> # print('Output: {}'.format(', '.join( ['{:.3f}'.format(x) for x in forward_out[0]] ) ) )
  >>>
  >>> # Visualizations
  >>>
  >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  >>> fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  >>> fig.suptitle('Connected Layer\nactivation : {}'.format(layer_activation.name))
  >>>
  >>> ax1.imshow(float_2_img(inpt[0]))
  >>> ax1.set_title('Original Image')
  >>> ax1.axis('off')
  >>>
  >>> ax2.matshow(forward_out[:, 0, 0, :], cmap='bwr')
  >>> ax2.set_title('Forward', y=4)
  >>> ax2.axes.get_yaxis().set_visible(False)         # no y axis tick
  >>> ax2.axes.get_xaxis().set_ticks(range(outputs))  # set x axis tick for every output
  >>>
  >>> ax3.imshow(float_2_img(delta[0]))
  >>> ax3.set_title('Backward')
  >>> ax3.axis('off')
  >>>
  >>> fig.tight_layout()
  >>> plt.show()

  References
  ----------
    - TODO
  '''

  def __init__(self, outputs, activation=Activations, input_shape=None, weights=None, bias=None, **kwargs):

    if isinstance(outputs, int) and outputs > 0:
      self.outputs = outputs
    else:
      raise ValueError('Parameter "outputs" must be an integer and > 0')

    self.weights = weights
    self.bias = bias

    activation = _check_activation(self, activation)

    self.activation = activation.activate
    self.gradient = activation.gradient

    super(Connected_layer, self).__init__(input_shape=input_shape)

    if input_shape is not None:
      self._build()

    self.weights_update = None
    self.bias_update = None
    self.optimizer = None

  @property
  def inputs(self):
    '''
    Number of inputs of the layer, not considering the batch size
    '''
    return np.prod(self.input_shape[1:])

  @property
  def out_shape(self):
    '''
    Returns the output shape in the format (batch, 1, 1, outputs)
    '''
    return (self.input_shape[0], 1, 1, self.outputs)

  def __str__(self):
    '''
    PRINTER
    '''
    b, w, h, c = self.input_shape
    return 'connected              {0:4d} x{1:4d} x{2:4d} x{3:4d}   ->  {0:4d} x{4:4d}'.format(b, w, h, c, self.outputs)

  def _build(self):
    if self.weights is None:
      scale = np.sqrt(2. / self.inputs)
      self.weights = np.random.uniform(low=-scale, high=scale, size=(self.inputs, self.outputs))

    if self.bias is None:
      self.bias = np.zeros(shape=(self.outputs,), dtype=float)

  def __call__(self, previous_layer):

    super(Connected_layer, self).__call__(previous_layer)
    self._build()

    return self

  def load_weights(self, chunck_weights, pos=0):
    '''
    Load weights from full array of model weights

    Parameters
    ----------
      chunck_weights : array-like
        model weights and bias

      pos : int (default=0)
        Current position of the array

    Returns
    ----------
      pos : int
        Updated stream position.
    '''
    self.bias = chunck_weights[pos: pos + self.outputs]
    pos += self.outputs

    self.weights = chunck_weights[pos: pos + self.weights.size]
    self.weights = self.weights.reshape(self.inputs, self.outputs)
    pos += self.weights.size

    return pos

  def save_weights(self):
    '''
    Return the biases and weights in a single ravel fmt to save in binary file
    '''
    return np.concatenate([self.bias.ravel(), self.weights.ravel()], axis=0).tolist()

  def forward(self, inpt, copy=False):
    '''
    Forward function of the connected layer. It computes the matrix product
    between inpt and weights, add bias and activate the result with the
    chosen activation function.

    Parameters
    ----------
      inpt : array-like
        Input batch in format (batch, in_w, in_h, in _c)

      copy : bool (default=False)
        If False the activation function modifies its input, if True make a copy instead

    Returns
    -------
      self
    '''

    # shape (batch, w*h*c)
    inpt = inpt.reshape(inpt.shape[0], -1)
    self._check_dims(shape=(self.input_shape[0], self.inputs), arr=inpt, func='Forward')

    # shape (batch, outputs)
    z = np.einsum('ij, jk -> ik', inpt, self.weights, optimize=True) + self.bias

    # shape (batch, outputs), activated
    self.output = self.activation(z, copy=copy).reshape(-1, 1, 1, self.outputs)
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, inpt, delta=None, copy=False):
    '''
    Backward function of the connected layer, updates the global delta of the
    network to be Backpropagated, he weights upadtes and the biases updates

    Parameters
    ----------
      delta : array-like
        delta array of shape (batch, w, h, c). Global delta to be backpropagated.

      copy : bool (default=False)
        States if the activation function have to return a copy of the input or not.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')

    # reshape to (batch , w * h * c)
    inpt = inpt.reshape(inpt.shape[0], -1)
    self._check_dims(shape=(self.input_shape[0], self.inputs), arr=inpt, func='Backward')
    # out  = self.output.reshape(-1, self.outputs)

    self.delta *= self.gradient(self.output, copy=copy)
    self.delta = self.delta.reshape(-1, self.outputs)

    self.bias_update = self.delta.sum(axis=0)   # shape : (outputs,)

    # self.weights_update += inpt.transpose() @ self.delta') # shape : (w * h * c, outputs)
    self.weights_update = np.einsum('ji, jk -> ik', inpt, self.delta, optimize=True)

    if delta is not None:
      delta_shaped = delta.reshape(inpt.shape[0], -1)  # it's a reshaped VIEW
      self._check_dims(shape=(self.input_shape[0], self.inputs), arr=delta_shaped, func='Backward')

      # shapes : (batch , w * h * c) = (batch , w * h * c) + (batch, outputs) @ (outputs, w * h * c)

      # delta_shaped[:] += self.delta @ self.weights.transpose()')  # I can modify delta using its view
      delta_shaped[:] += np.einsum('ij, kj -> ik', self.delta, self.weights, optimize=True)

    return self

  def update(self):
    '''
    Update function for the convolution layer.
    Optimizer must be assigned externally as an optimizer object.

    Returns
    -------
      self
    '''

    check_is_fitted(self, 'delta')

    self.bias, self.weights = self.optimizer.update(params=[self.bias, self.weights],
                                                    gradients=[self.bias_update, self.weights_update]
                                                    )

    return self


if __name__ == '__main__':

  import os

  import matplotlib.pyplot as plt
  from PIL import Image

  from NumPyNet import activations

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)

  # from (w, h, c) to shape (1, w, h, c)
  inpt = np.expand_dims(inpt, axis=0)  # just to add the 'batch' dimension

  # Number of outputs
  outputs = 10
  layer_activation = activations.Relu()
  batch, w, h, c = inpt.shape

  # Random initialization of weights with shape (w * h * c) and bias with shape (outputs,)
  np.random.seed(123)  # only if one want always the same set of weights
  weights = np.random.uniform(low=-1., high=1., size=(np.prod(inpt.shape[1:]), outputs))
  bias = np.random.uniform(low=-1., high=1., size=(outputs,))

  # Model initialization
  layer = Connected_layer(outputs, input_shape=inpt.shape,
                          activation=layer_activation, weights=weights, bias=bias)
  print(layer)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output.copy()

  # BACKWARD

  layer.delta = np.ones(shape=(layer.out_shape), dtype=float)
  delta = np.zeros(shape=(batch, w, h, c), dtype=float)
  layer.backward(inpt, delta=delta, copy=True)

  # print('Output: {}'.format(', '.join( ['{:.3f}'.format(x) for x in forward_out[0]] ) ) )

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('Connected Layer\nactivation : {}'.format(layer_activation.name))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original Image')
  ax1.axis('off')

  ax2.matshow(forward_out[:, 0, 0, :], cmap='bwr')
  ax2.set_title('Forward', y=4)
  ax2.axes.get_yaxis().set_visible(False)         # no y axis tick
  ax2.axes.get_xaxis().set_ticks(range(outputs))  # set x axis tick for every output

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
