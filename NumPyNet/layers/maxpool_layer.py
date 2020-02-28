#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Maxpool_layer(object):

  def __init__(self, size, stride=None, pad=None, **kwargs):

    '''
    MaxPool Layer: perfmors a downsample of the image through the slide of a kernel
    of shape equal to size = (kx, ky) with step stride = (st1, st2)

    Parameters
    ----------
      size   : int or tuple of int. Size of the kernel with shape (kx, ky)
      stride : int or tuple of int, default None. Step of the kernel with shape (st1, st2).
              If None, stride is set equal to size.
      pad    : boolean, default None. If True pad the image following keras SAME
        padding. If False the image is not padded
    '''

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
      raise LayerError('Maxpool layer. Incompatible stride/size dimensions. They must be a 1D-2D tuple of values')

    self.batch, self.w, self.h, self.c = (None, None, None, None)

    # for padding
    self.pad = pad
    self.pad_left, self.pad_right, self.pad_bottom, self.pad_top = (0, 0, 0, 0)

    self.output, self.indexes, self.delta = (None, None, None)


  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape
    return 'max         {} x {} / {}  {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d} x{:>4d}'.format(
           self.size[0], self.size[1], self.stride[0],
           self.batch, self.w, self.h, self.c,
           batch, out_width, out_height, out_channels)

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.batch, self.w, self.h, self.c = previous_layer.out_shape

    if self.pad:
      self._evaluate_padding()

    return self

  @property
  def out_shape(self):
    out_height   = (self.h + self.pad_left + self.pad_right - self.size[1]) // self.stride[1] + 1
    out_width    = (self.w + self.pad_top + self.pad_bottom - self.size[0]) // self.stride[0] + 1
    out_channels = self.c
    return (self.batch, out_width, out_height, out_channels)

  def _asStride(self, inpt):
    '''
    _asStride returns a view of the input array such that a kernel of size = (kx,ky)
    is slided over the image with stride = (st1, st2)

    better reference here :
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html

    see also:
    https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy

    Parameters
    ----------
      inpt : numpy array, input batch of images to be strided with shape = (batch, w, h, c)

    Returns
    -------
      View on input with shape (batch, out_w, out_h, out_c, kx, ky)
    '''

    batch_stride, s0, s1, s3 = inpt.strides
    batch, w, h, c = inpt.shape
    kx, ky   = self.size
    st1, st2 = self.stride

    out_w = 1 + (w - kx)//st1
    out_h = 1 + (h - ky)//st2

    # Shape of the final view
    view_shape = (batch, out_w , out_h, c) + (kx, ky)

    # strides of the final view
    strides = (batch_stride, s0*st1, s1*st2, s3) + (s0, s1)

    subs = np.lib.stride_tricks.as_strided(inpt, view_shape, strides=strides)
    return subs

  def _evaluate_padding(self):
    '''
    Compute padding dimensions
    '''

    # Compute how many raws are needed to pad the image in the 'w' axis
    if (self.w % self.stride[0] == 0):
      pad_w = max(self.size[0] - self.stride[0], 0)
    else:
      pad_w = max(self.size[0] - (self.w % self.stride[0]), 0)

    # Compute how many Columns are needed
    if (self.h % self.stride[1] == 0):
      pad_h = max(self.size[1] - self.stride[1], 0)
    else:
      pad_h = max(self.size[1] - (self.h % self.stride[1]), 0)

    # Number of raws/columns to be added for every directons
    self.pad_top    = pad_w >> 1 # bit shift, integer division by two
    self.pad_bottom = pad_w - self.pad_top
    self.pad_left   = pad_h >> 1
    self.pad_right  = pad_h - self.pad_left

  def _pad(self, inpt):
    '''
    Padd every image in a batch with np.nan following keras SAME padding
    See also:
      https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave

    Parameters
    ----------
      inpt : numpy array, input images in the format (batch, w, h, c)

    Returns
    -------
      padded array.
    '''

    # return the nan-padded image, in the same format as inpt (batch, width + pad_w, height + pad_h, channels)
    return np.pad(inpt, pad_width=((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                  mode='constant', constant_values=(np.nan, np.nan))

  def forward(self, inpt):
    '''
    Forward function of the maxpool layer: It slides a kernel over every input image and return
    the maximum value of every sub-window.
    the function _asStride returns a view of the input arrary with shape
    (batch, out_w, out_h , c, kx, ky), where, for every image in the batch we have:
    out_w * out_h * c sub matrixes kx * ky, containing pixel values.

    Parameters
    ----------
      inpt : input images in the format (batch, input_w, input_h, input_c)

    Returns
    -------
      Maxpool layer object
    '''

    self.batch, self.w, self.h, self.c = inpt.shape
    kx , ky  = self.size
    st1, st2 = self.stride

    if self.pad:
      self._evaluate_padding()
      mat_pad = self._pad(inpt)
    else:
      # If no padding, cut the last raws/columns in every image in the batch
      mat_pad = inpt[:, : (self.w - kx) // st1*st1 + kx, : (self.h - ky) // st2*st2 + ky, ...]

    # Return a strided view of the input array, shape: (batch, 1+(w-kx)//st1,1+(h-ky)//st2 ,c, kx, ky)
    view = self._asStride(mat_pad)

    # final shape (batch, out_w, out_h, c)

    self.output = np.nanmax(view, axis=(4,5))

    # New shape for view, to retrieve indexes
    new_shape = view.shape[:4] + (kx*ky, )

    self.indexes = np.nanargmax(view.reshape(new_shape), axis=4)

    # self.indexes = np.unravel_index(self.indexes.ravel(), (kx, ky)) ?
    try:
      self.indexes = np.unravel_index(self.indexes.ravel(), shape=(kx, ky))
    except TypeError: # retro-compatibility for Numpy version older than 1.16
      self.indexes = np.unravel_index(self.indexes.ravel(), dims=(kx, ky))

    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta):
    '''
    Backward function of maxpool layer: it access avery position where in the input image
    there's a chosen maximum and add the correspondent self.delta value.
    Since we work with a 'view' of delta, the same pixel may appear more than one time,
    and an atomic acces to it's value is needed to correctly modifiy it.

    Parameters
    ----------
      delta : numpy array, the global delta to be backpropagated with shape (batch, w, h, c)

    Returns
    -------
      maxpool layer object.
    '''

    check_is_fitted(self, 'delta')

    # Padding delta in order to create another view
    if self.pad:
      mat_pad = self._pad(delta)
    else:
      mat_pad = delta

    # Create a view of mat_pad, following the padding true or false
    net_delta_view = self._asStride(mat_pad)

    b, w, h, c = self.output.shape

    # those indexes are usefull to access 'Atomically'(one at a time) every element in net_delta_view
    for (i, j, k, l), m, o, D in zip(np.ndindex(b, w, h, c), self.indexes[0], self.indexes[1], np.nditer(self.delta)):
      net_delta_view[i, j, k, l, m, o] += D

    # Here delta is correctly modified
    if self.pad:
      _ , w_pad, h_pad, _ = mat_pad.shape
      delta[:] = mat_pad[:, self.pad_top : w_pad-self.pad_bottom, self.pad_left : h_pad - self.pad_right, :]
    else:
      delta[:] = mat_pad

    return self

if __name__ == '__main__':

  import os

  import pylab as plt
  from PIL import Image

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname('__file__'), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)

  inpt = np.expand_dims(inpt, axis=0) # Add the batch shape.
  b, w, h, c = inpt.shape

  size = (3, 3)
  stride = (2, 2)
  pad = False

  layer = Maxpool_layer(size=size, stride=stride, padding=pad)

  # FORWARD

  layer.forward(inpt)

  forward_out = layer.output

  print(layer)

  # BACKWARD

  delta = np.zeros(inpt.shape, dtype=float)
  layer.delta = np.ones(layer.out_shape, dtype=float)
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)
  fig.suptitle('MaxPool Layer\nsize : {}, stride : {}, padding : {} '.format(size, stride, pad))

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
