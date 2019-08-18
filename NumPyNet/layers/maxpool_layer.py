#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import itertools

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Maxpool Layer'


class Maxpool_layer(object):

  def __init__(self, size, stride=None, padding=None):

    '''
    MaxPool Layer: perfmors a downsample of the image through the slide of a kernel
    of shape equal to size = (kx, ky) with step stride = (st1, st2)

    Parameters:
      size   : tuple of int, size of the kernel with shape (kx, ky)
      stride : tuple of int, step of the kernel with shape (st1, st2)
      padding: boolean, default is None. If True pad the image following keras SAME
        padding. If False the image is not padded
    '''

    self.size = size

    if stride is not None:
      self.stride = stride
    else:
      self.stride = size

    self.batch, self.w, self.h, self.c = (0, 0, 0, 0)

    # for padding
    self.pad = padding
    self.pad_left, self.pad_right, self.pad_bottom, self.pad_top = (0, 0, 0, 0)

    self.output, self.indexes, self.delta = (None, None, None)


  def __str__(self):
    batch, out_width, out_height, out_channels = self.out_shape()
    return 'MaxPool      {} x {} / {}  {:>4d} x{:>4d} x{:>4d} x{:>4d}   ->  {:>4d} x{:>4d} x{:>4d} x{:>4d}'.format(
           self.size[0], self.size[1], self.stride[0],
           self.batch, self.w, self.h, self.c,
           batch, out_width, out_height, out_channels)

  def out_shape(self):
    out_height   = (self.h + self.pad_left + self.pad_right - self.size[1]) // self.stride[1] + 1
    out_width    = (self.w + self.pad_top + self.pad_bottom - self.size[0]) // self.stride[0] + 1
    out_channels = self.c
    return (self.batch, out_width, out_height, out_channels)

  def _asStride(self, inpt, size, stride):
    '''
    _asStride returns a view of the input array such that a kernel of size = (kx,ky)
    is slided over the image with stride = (st1, st2)

    better reference here :
    https://docs.scipy.org/doc/numpy/reference/generated/numpy.lib.stride_tricks.as_strided.html

    see also:
    https://stackoverflow.com/questions/42463172/how-to-perform-max-mean-pooling-on-a-2d-array-using-numpy

    Parameters:
      inpt  : input batch of images to be stride with shape = ()
      size  : a tuple indicating the horizontal and vertical size of the kernel
      stride: a tuple indicating the horizontal and vertical steps of the kernel
    '''
    try:

      exec('batch_stride, s0, s1, *_ = inpt.strides')
      exec('batch,        w,  h,  *_ = inpt.shape')

    except SyntaxError: # old python compatibility

      batch_stride, s0, s1, _ = inpt.strides[0], inpt.strides[1], inpt.strides[2], inpt.strides[3:]
      batch,        w,  h,  _ = inpt.shape[0],   inpt.shape[1],   inpt.shape[2],   inpt.shape[3:]

    kx, ky     = size
    st1, st2   = stride

    out_w = 1 + (w - kx)//st1
    out_h = 1 + (h - ky)//st2

    # Shape of the final view
    view_shape = (batch, out_w , out_h) + inpt.shape[3:] + (kx, ky)

    # strides of the final view
    strides = (batch_stride, st1 * s0, st2 * s1) + inpt.strides[3:] + (s0, s1)

    subs = np.lib.stride_tricks.as_strided(inpt, view_shape, strides = strides)
    return subs

  def _pad(self, inpt, size, stride):
    '''
    Padd every image in a batch with np.nan, following keras SAME padding.
    See also:
      https://stackoverflow.com/questions/53819528/how-does-tf-keras-layers-conv2d-with-padding-same-and-strides-1-behave

    Parameters:
      inpt    : input images in the format (batch, in_w, in_h, in_c)
      size    : tuple, size of the kernel in the format (kx, ky)
      stride  : tuple, size of the strides of the kernel in the format (st1, st2)
    '''

    _, w, h, c = inpt.shape

    # Compute how many raws are needed to pad the image in the 'w' axis
    if (w % stride[0] == 0):
      pad_w = max(size[0] - stride[0], 0)
    else:
      pad_w = max(size[0] - (w % stride[0]), 0)

    # Compute how many Columns are needed
    if (h % stride[1] == 0):
      pad_h = max(size[1] - stride[1], 0)
    else:
      pad_h = max(size[1] - (h % stride[1]), 0)

    # Number of raws/columns to be added for every directons
    self.pad_top    = pad_w >> 1 # bit shift, integer division by two
    self.pad_bottom = pad_w - self.pad_top
    self.pad_left   = pad_h >> 1
    self.pad_right  = pad_h - self.pad_left

    # return the nan-padded image, in the same format as inpt (batch, width + pad_w, height + pad_h, channels)
    return np.pad(inpt, ((0, 0), (self.pad_top, self.pad_bottom), (self.pad_left, self.pad_right), (0, 0)),
                  mode='constant', constant_values=(np.nan, np.nan))

  def forward(self, inpt):

    '''
    Forward function of the maxpool layer: It slides a kernel over every input image and return
    the maximum value of every sub-window.
    the function _asStride returns a view of the input arrary with shape
    (batch, out_w, out_h , c, kx, ky), where, for every image in the batch we have:
    out_w * out_h * c sub matrixes kx * ky, containing pixel values.

    Parameters:
      inpt : input images in the format (batch, input_w, input_h, input_c)
    '''

    self.batch, self.w, self.h, self.c = inpt.shape
    kx , ky  = self.size
    st1, st2 = self.stride

    if self.pad:
      mat_pad = self._pad(inpt, self.size, self.stride)
    else:
      # If no padding, cut the last raws/columns in every image in the batch
      mat_pad = inpt[:,: (self.w - kx) // st1*st1 + kx, : (self.h - ky) // st2*st2 + ky, ...]

    # Return a strided view of the input array, shape: (batch, 1+(w-kx)//st1,1+(h-ky)//st2 ,c, kx, ky)
    view = self._asStride(mat_pad, self.size, self.stride)

    self.output = np.nanmax(view, axis=(4, 5)) # final shape (batch, out_w, out_h, c)

    # New shape for view, to access the single sub matrix and retrieve couples of indexes
    new_shape = (np.prod(view.shape[:-2]), view.shape[-2], view.shape[-1])

    # Retrives a tuple of indexes (x,y) for every sub-matrix of the view array, that indicates
    # where the maximum value is.
    # In the loop I change the shape of view in order to have access to its last 2 dimension with r.
    # r take the values of every sub matrix
    self.indexes = [np.unravel_index(np.nanargmax(r), r.shape) for r in view.reshape(new_shape)]
    self.indexes = np.asarray(self.indexes).T

  def backward(self, delta):
    '''
    Backward function of maxpool layer: it access avery position where in the input image
    there's a chosen maximum and add the correspondent self.delta value.
    Since we work with a 'view' of delta, the same pixel may appear more than one time,
    and an atomic acces to it's value is needed to correctly modifiy it.

    Parameters:
      delta : the global delta to be backpropagated with shape (batch, w, h, c)
    '''

    # Padding delta in order to create another view
    if self.pad:
      mat_pad = self._pad(delta, self.size, self.stride)
    else:
      mat_pad = delta

    # Create a view of net delta, following the padding true or false
    net_delta_view = self._asStride(mat_pad, self.size, self.stride) #that is a view on mat_pad

    # Create every possibile combination of index for the first four dimensions of
    # a six dimensional array
    b, w, h, c = self.output.shape
    combo = itertools.product(range(b), range(w), range(h), range(c))
    combo = np.asarray(list(combo)).T
    # here I left the transposition, because of self.indexes

    # those indexes are usefull to acces 'Atomically'(one at a time) every element in net_delta_view
    # that needs to be modified
    for b, i, j, k, x, y in zip(combo[0], combo[1], combo[2], combo[3], self.indexes[0], self.indexes[1]):
      net_delta_view[b, i, j, k, x, y] += self.delta[b, i, j, k]

    # Here delta is correctly modified
    if self.pad:
      _ , w_pad, h_pad, _ = mat_pad.shape
      delta[:] = mat_pad[:, self.pad_top : w_pad-self.pad_bottom, self.pad_left : h_pad - self.pad_right, :]
    else:
      delta[:] = mat_pad

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

  inpt = np.expand_dims(inpt, axis=0) # Add the batch shape.
  b, w, h, c = inpt.shape

  size = (30, 30)
  stride = (20, 20)
  pad = False

  layer = Maxpool_layer(size=size, stride=stride, padding=pad)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output

  print(layer) # after the forward, to load all the variable

  # BACKWARD

  delta = np.zeros(inpt.shape)
  layer.delta = np.ones(layer.out_shape())
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

  plt.show()
