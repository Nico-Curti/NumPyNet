#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np
from numpy.lib.stride_tricks import as_strided

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Upsample Layer'


class Upsample_layer(object):

  def __init__(self, stride=(2, 2), scale=1., **kwargs):
    '''
    Upsample / Downsample layer

    Parameters:
      stride : scaling factor of the input;
        Repeats the rows and columns of the data by stride[0] and stride[1] respectively.
      scale  : floating point scale factor of the input
    '''

    self.scale = float(scale)
    self.stride = stride

    if not hasattr(self.stride, '__iter__'):
      self.stride = (int(stride), int(stride))

    assert len(self.stride) == 2

    if self.stride[0] < 0 and self.stride[1] < 0: # downsample
      self.stride = (-self.stride[0], -self.stride[1])
      self.reverse = True

    elif self.stride[0] > 0 and self.stride[1] > 0: # upsample
      self.reverse = False

    else:
      raise NotImplementedError('Mixture upsample/downsample are not yet implemented')

    self.output, self.delta = (None, None)


  def __str__(self):
    out_w = self.w // self.stride[0] if self.reverse else self.w * self.stride[0]
    out_h = self.h // self.stride[1] if self.reverse else self.h * self.stride[1]

    if self.reverse: # downsample
      return 'downsample         {0:>2d}/{1:>2d}x {2:>4d} x{3:>4d} x{4:>4d}  -> {5:>4d} x{6:>4d} x{7:4d}'.format(
        self.stride[0], self.stride[1], self.w, self.h, self.c, out_w, out_h, self.c)
    else:            # upsample
      return 'upsample           {0:>2d}/{1:>2d}x {2:>4d} x{3:>4d} x{4:>4d}  -> {5:>4d} x{6:>4d} x{7:4d}'.format(
        self.stride[0], self.stride[1], self.w, self.h, self.c, out_w, out_h, self.c)

  @property
  def out_shape(self):
    return self.output.shape

  def _downsample (self, inpt):
    batch, w, h, c = inpt.shape
    scale_w = w // self.stride[0]
    scale_h = h // self.stride[1]

    return inpt.reshape(batch, scale_w, self.stride[0], scale_h, self.stride[1], c).mean(axis=(2, 4))

  def _upsample (self, inpt):
    batch, w,  h,  c  = inpt.shape     # number of rows/columns
    b,     ws, hs, cs = inpt.strides   # row/column strides

    x = as_strided(inpt, (batch, w, self.stride[0], h, self.stride[1], c), (b, ws, 0, hs, 0, cs)) # view a as larger 4D array
    return x.reshape(batch, w * self.stride[0], h * self.stride[1], c)                                     # create new 2D array

  def forward(self, inpt):
    '''
    Forward of the upsample layer, apply a bilinear upsample/downsample to
    the input according to the sign of stride

    Parameters:
      inpt: the input to be up-down sampled
    '''
    self.batch, self.w, self.h, self.c = inpt.shape

    if self.reverse: # Downsample
      self.output = self._downsample(inpt) * self.scale

    else:            # Upsample
      self.output = self._upsample(inpt) * self.scale

    self.delta = np.zeros(shape=inpt.shape, dtype=float)

  def backward(self, delta):
    '''
    Compute the inverse transformation of the forward function
    on the gradient

    Parameters:
      delta : global error to be backpropagated
    '''

    if self.reverse: # Upsample
      delta[:] = self._upsample(self.delta) * (1. / self.scale)

    else:            # Downsample
      delta[:] = self._downsample(self.delta) * (1. / self.scale)



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

  # add batch = 1
  inpt = np.expand_dims(inpt, axis=0)

  stride = 2
  scale = 1.5

  layer = Upsample_layer(scale=scale, stride=stride)

  # FORWARD

  layer.forward(inpt)
  forward_out = layer.output
  print(layer)

  # BACKWARD

  layer.delta = layer.output
  delta = np.empty(shape=inpt.shape, dtype=float)
  layer.backward(delta)

  # Visualizations

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize=(10, 5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle('Upsample Layer\nscale : {:.3f}, stride : {:d}'.format(scale, stride))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title('Original image')
  ax1.axis('off')

  ax2.imshow(float_2_img(forward_out[0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title('Backward')
  ax3.axis('off')

  plt.show()