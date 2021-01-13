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


class Avgpool_layer(BaseLayer):

  '''
  Avgpool layer

  Parameters
  ----------
    size : tuple or int
      Size of the kernel to slide over the input image. If a tuple, it must contains two integers, (kx, ky).
      If a int, size = kx = ky.

    stride  : tuple or int (default=None)
      Represents the horizontal and vertical stride of the kernel (sx, sy).
      If None or 0, stride is assigned the same values as `size`.

    input_shape : tuple (default=None)
      Input shape of the layer. The default value is used when the layer is part of a network.

    pad : bool, (default=False)
      If False the image is cut to fit the size and stride dimensions, if True the
      image is padded following keras SAME padding, see references for details.

  Examples
  --------
  >>> import os

  >>> import pylab as plt
  >>> from PIL import Image
  >>>
  >>> img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  >>> float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)
  >>>
  >>> filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  >>> inpt = np.asarray(Image.open(filename), dtype=float)
  >>> inpt.setflags(write=1)
  >>> inpt = img_2_float(inpt)
  >>>
  >>> inpt = np.expand_dims(inpt, axis=0)
  >>> pad  = False
  >>>
  >>> size   = 3
  >>> stride = 2
  >>>
  >>> # Model initialization
  >>> layer = Avgpool_layer(input_shape=inpt.shape, size=size, stride=stride, padding=pad)
  >>>
  >>> # FORWARD
  >>>
  >>> layer.forward(inpt)
  >>> forward_out = layer.output.copy()
  >>>
  >>> print(layer)
  >>>
  >>> # BACKWARD
  >>>
  >>> delta = np.random.uniform(low=0., high=1.,size=inpt.shape)
  >>> layer.delta = np.ones(layer.out_shape, dtype=float)
  >>> layer.backward(delta)
  >>>
  >>> # Visualizations
  >>>
  >>> fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  >>> fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  >>>
  >>> fig.suptitle('Average Pool Layer\n\nsize : {}, stride : {}, padding : {}'.format(size, stride, pad))
  >>>
  >>> ax1.imshow(float_2_img(inpt)[0])
  >>> ax1.set_title('Original image')
  >>> ax1.axis('off')
  >>>
  >>> ax2.imshow(float_2_img(layer.output[0]))
  >>> ax2.set_title('Forward')
  >>> ax2.axis('off')
  >>>
  >>> ax3.imshow(float_2_img(delta[0]))
  >>> ax3.set_title('Backward')
  >>> ax3.axis('off')
  >>>
  >>> fig.tight_layout()
  >>> plt.show()

  .. image:: ../../../NumPyNet/images/average_3-2.png
  .. image:: ../../../NumPyNet/images/average_30-20.png

  References
  ----------
    - https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html
    - https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy
    - https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave
  '''

  def __init__(self, size, stride=None, pad=False, input_shape=None, **kwargs):

    self.size = size

    if not hasattr(self.size, '__iter__'):
      self.size = (int(self.size), int(self.size))

    if self.size[0] <= 0. or self.size[1] <= 0.:
      raise LayerError('Avgpool layer. Incompatible size dimensions. They must be both > 0')

    if not stride:
      self.stride = size
    else:
      self.stride = stride

    if not hasattr(self.stride, '__iter__'):
      self.stride = (int(self.stride), int(self.stride))

    if len(self.size) != 2 or len(self.stride) != 2:
      raise LayerError('Avgpool layer. Incompatible stride/size dimensions. They must be a 1D-2D tuple of values')

    # for padding
    self.pad = pad
    self.pad_left, self.pad_right, self.pad_bottom, self.pad_top = (0, 0, 0, 0)

    super(Avgpool_layer, self).__init__(input_shape=input_shape)
    self._build(input_shape)

  def _build(self, input_shape=None):
    if input_shape is not None:

      if self.pad:
        self._evaluate_padding()

  def __str__(self):
    batch, w, h, c = self.input_shape
    _, out_width, out_height, out_channels = self.out_shape
    return 'avg         {} x {} / {}  {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d}'.format(
           self.size[0], self.size[1], self.stride[0],
           batch, w, h, c,
           out_width, out_height, out_channels)

  @property
  def out_shape(self):
    batch, w, h, c = self.input_shape
    out_height = (h + self.pad_left + self.pad_right - self.size[1]) // self.stride[1] + 1
    out_width = (w + self.pad_top + self.pad_bottom - self.size[0]) // self.stride[0] + 1
    out_channels = c
    return (batch, out_width, out_height, out_channels)

  def _asStride(self, arr):
    '''
    _asStride returns a view of the input array such that a kernel of size = (kx,ky)
    is slided over the image with stride = (st1, st2)

    better reference here :
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html

    see also:
    https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy

    Parameters
    ----------
      arr : array-like
        Input batch of images to be convoluted with shape = (b, w, h, c)

    Returns
    -------
      subs : array-view
        View of the input array with shape (batch, out_w, out_h, kx, ky, out_c)
    '''

    batch_stride, s0, s1 = arr.strides[:3]
    batch, w, h = arr.shape[:3]
    kx, ky = self.size
    st1, st2 = self.stride

    # Shape of the final view
    view_shape = (batch, 1 + (w - kx) // st1, 1 + (h - ky) // st2) + arr.shape[3:] + (kx, ky)

    # strides of the final view
    strides = (batch_stride, st1 * s0, st2 * s1) + arr.strides[3:] + (s0, s1)

    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides=strides)
    # returns a view with shape = (batch, out_w, out_h, out_c, kx, ky)
    return subs

  def _evaluate_padding(self):
    '''
    Compute padding dimensions, following keras VALID and SAME criteria. See:
    https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave
    '''
    _, w, h, _ = self.input_shape
    # Compute how many raws are needed to pad the image in the 'w' axis
    if (w % self.stride[0] == 0):
      pad_w = max(self.size[0] - self.stride[0], 0)
    else:
      pad_w = max(self.size[0] - (w % self.stride[0]), 0)

    # Compute how many Columns are needed
    if (h % self.stride[1] == 0):
      pad_h = max(self.size[1] - self.stride[1], 0)
    else:
      pad_h = max(self.size[1] - (h % self.stride[1]), 0)

    # Number of raws/columns to be added for every directons
    self.pad_top = pad_w >> 1  # bit shift, integer division by two
    self.pad_bottom = pad_w - self.pad_top
    self.pad_left = pad_h >> 1
    self.pad_right = pad_h - self.pad_left

  def _pad(self, inpt):
    '''
    Padd every image in a batch with np.nan following keras SAME padding
    See also:
      https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave

    Parameters
    ----------
      inpt : array-like
        Input images in the format (batch, width, height, channels).

    Returns
    -------
      array-like
        A padded batch of images, following keras SAME padding.
    '''

    # return the nan-padded image, in the same format as inpt (batch, width + pad_w, height + pad_h, channels)
    return np.pad(inpt, pad_width=((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                  mode='constant', constant_values=(np.nan, np.nan))

  def forward(self, inpt):
    '''
    Forward function of the average pool layer: it slide a kernel of size (kx,ky) = size
    and with step (st1, st2) = strides over every image in the batch. For every sub-matrix
    it computes the average value without considering NAN value (padding), and passes it
    to the output.

    Parameters
    ----------
      inpt : array-like
        Input batch of images, with shape (batch, input_w, input_h, input_c).

    Returns
    -------
      self
    '''
    self._check_dims(shape=self.input_shape, arr=inpt, func='Forward')

    kx, ky = self.size
    sx, sy = self.stride
    _, w, h, _ = self.input_shape
    inpt = inpt.astype('float64')

    # Padding
    if self.pad:
      mat_pad = self._pad(inpt)
    else:
      # If padding false, it cuts images' raws/columns
      mat_pad = inpt[:, : (w - kx) // sx * sx + kx, : (h - ky) // sy * sy + ky, ...]

    # 'view' is the strided input image, shape = (batch, out_w, out_h, out_c, kx, ky)
    view = self._asStride(mat_pad)

    # Mean of every sub matrix, computed without considering the padd(np.nan)
    self.output = np.nanmean(view, axis=(4, 5))
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta):
    '''
    Backward function of the averagepool layer: the function modifies the net delta
    to be backpropagated.

    Parameters
    ----------
      delta : array-like
        Global delta to be backpropagated with shape (batch, out_w, out_h, out_c).

    Returns
    ----------
      self
    '''

    check_is_fitted(self, 'delta')
    self._check_dims(shape=self.input_shape, arr=delta, func='Backward')
    delta[:] = delta.astype('float64')

    # kx, ky = self.size

    # Padding delta for a coherent _asStrided dimension
    if self.pad:
      mat_pad = self._pad(delta)
    else:
      mat_pad = delta

    # _asStrid of padded delta let me access every pixel of the memory in the order I want.
    # This is used to create a 1-1 correspondence between output and input pixels.
    net_delta_view = self._asStride(mat_pad)

    # norm = 1./(kx*ky) # needs to count only no nan values for keras
    _, w, h, c = self.output.shape

    # The indexes are necessary to access every pixel value one at a time, since
    # modifing the same memory address more times at once doesn't produce the correct result

    # norm = 1. / (kx*ky)
    norm = self.delta * (1. / np.count_nonzero(~np.isnan(net_delta_view), axis=(4, 5)))
    net_delta_review = np.moveaxis(net_delta_view, source=[1, 2, 3], destination=[0, 1, 2])

    for (i, j, k), n in zip(np.ndindex(w, h, c), np.nditer(norm)):
      net_delta_review[i, j, k, ...] += n
    # net_delta_view *= norm

    # Here delta is updated correctly
    if self.pad:
      _, w_pad, h_pad, _ = mat_pad.shape
      # Excluding the padded part of the image
      delta[:] = mat_pad[:, self.pad_top: w_pad - self.pad_bottom, self.pad_left: h_pad - self.pad_right, :]
    else:
      delta[:] = mat_pad

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
  pad = False

  size = 3
  stride = 2

  # Model initialization
  layer = Avgpool_layer(input_shape=inpt.shape, size=size, stride=stride, padding=pad)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output.copy()

  print(layer)

  # BACKWARD

  delta = np.random.uniform(low=0., high=1., size=inpt.shape)
  layer.delta = np.ones(layer.out_shape, dtype=float)
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Average Pool Layer\n\nsize : {}, stride : {}, padding : {}'.format(size, stride, pad))

  ax1.imshow(float_2_img(inpt)[0])
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(layer.output[0]))
  ax2.set_title('Forward')
  ax2.axis('off')

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  fig.tight_layout()
  plt.show()
