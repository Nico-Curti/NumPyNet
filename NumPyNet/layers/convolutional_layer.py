#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import itertools

from NumPyNet.activations import Activations

import numpy as np


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Convolutional layer'


class Convolutional_layer(object):

  def __init__(self, inputs, channels_out, weights=None, bias=None,
               size=(3,3), stride=(1,1),
               padding=False,
               activation=Activations):
    """
    Convolution Layer: the output is the convolution of of the input batch
    with a group of kernel of shape size = (kx,ky) with step stride.

    Parameters:
      inputs       : tuple, shape of the input batch of image (batch, w, h, channels_in)
      channels_out : int, Number of filters to be slided over the input, and also
              the number of channels of the output
      weights     :  filters array, with shape (kx, ky, channels_in, channels_out)
      padding     : boolean, if False the image is cutted along the last raws and columns, if True
                    the input is padded following keras SAME padding
      activation  : activation function of the layer
    """

    self.batch, self.w, self.h, self.c = inputs
    self.channels_out = channels_out
    self.size = size
    self.stride = stride
    self.pad = padding

    #Activation function
    self.activation = activation.activate
    self.gradient   = activation.gradient

    self.delta, self.output = (None,None)

    #Weights and bias
    if weights is None:
      self.weights = np.random.uniform(0., 1., size=(*size, self.c, self.channels_out))
    else :
      self.weights = weights

    if bias is None:
      self.bias = np.random.uniform(0., 1., size=(self.channels_out))
    else :
      self.bias = bias

    #Updates
    self.filters_updates, self.bias_updates = (None,None)


  def out_shape(self):
    return (self.batch, self.out_w, self.out_h, self.channels_out)

  def _asStride(self, arr, sub_shape, stride):
    """
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
    """
    B, s0, s1, *_ = arr.strides
    b, m1, n1, *_ = arr.shape
    m2, n2        = sub_shape
    st1, st2      = stride

    self.out_w = 1 + (m1 - m2) // st1 #output final shapes out_w, out_h
    self.out_h = 1 + (n1 - n2) // st2

#    Shape of the final view
    view_shape = (b,) + (self.out_w, self.out_h) + (m2, n2) + arr.shape[3:]

#    strides of the final view
    strides =   (B,) + (st1 * s0,st2 * s1) + (s0, s1) + arr.strides[3:]

    subs = np.lib.stride_tricks.as_strided(arr, view_shape, strides = strides)
      # without any reshape, it's indeed a view of the input
    return subs

  def _pad(self, inpt, size, stride):
    """
    Padd every image in a batch with zeros, following keras SAME padding

    Parameters:
      inpt    : input images in the format (batch, in_w, in_h, in_c)
      size    : tuple, size of the kernel in the format (kx, ky)
      stride  : tuple, size of the strides of the kernel in the format (st1, st2)
    """

    b,w,h,c = inpt.shape

    #Compute how many Raws are needed to pad the image in the "w" axis
    if (w % stride[0] == 0):
      pad_w = max(size[0] - stride[0], 0)
    else:
      pad_w = max(size[0] - (w % stride[0]), 0)

    #Compute how many Columns are needed to pad the image in "h" axis
    if (h % stride[1] == 0):
      pad_h = max(size[1] - stride[1], 0)
    else:
      pad_h = max(size[1] - (h % stride[1]), 0)

    #Number of raws/columns to be added for every directons
    self.pad_top    = pad_w >> 1
    self.pad_bottom = pad_w - self.pad_top
    self.pad_left   = pad_h >> 1
    self.pad_right  = pad_h - self.pad_left

    #return the zeros-padded image, in the same format as inpt (batch, in_w + pad_w, in_h + pad_h, in_c)
    return np.pad(inpt, ((0,0),(self.pad_top,self.pad_bottom),(self.pad_left, self.pad_right),(0,0)),
                  mode = "constant", constant_values = (0., 0.))

  def forward(self, inpt, copy=False):
    """
    Forward function of the Convolutional Layer: it convolves an image with "channels_out"
    filters with dimension (kx,ky, channels_in). In doing so, it creates a view of the image
    with shape (batch * out_w * out_h, in_c * kx * ky) in order to perform a single matrix
    multiplication with the reshaped filters array, wich shape is (in_c * kx * ky, out_c)


    Parameters:
      inpt : input batch of images in format (batch, in_w, in_h, in _c)
      copy : boolean, default is False. If False the activation function
            modifies it's input, if True make a copy instead
    """

    kx, ky = self.size
    sx, sy = self.stride

    #Padding
    if self.pad :
      mat_pad = self._pad(inpt, self.size, self.stride)
    else :
      #If no pad, every image in the batch is cut
      mat_pad = inpt[:,: (self.w - kx) // sx*sx + kx, : (self.h - ky) // sy*sy + ky, ...]

    #Create the view of the array with shape (batch, out_w ,out_h, kx, ky, in_c)
    self.view = self._asStride(mat_pad, self.size, self.stride) #self, is used also in backward. Better way?

    #the choice of numpy.einsum is due to reshape of self.view is a copy and not a view
    #it seems to be slower though

    z = np.einsum("lmnijk,ijko -> lmno", self.view, self.weights) # + self.bias

    self.output = self.activation(z, copy=copy) # (batch * out_w * out_h, out_c)
    #final output shape : (batch, out_w, out_h, out_c)


  def backward(self, delta, copy=False):
    """
    Backward function of the Convolutional layer

    Parameters:
      delta : array of shape (batch, in_w, in _h, in_c). Global delta to be
        backpropagated
      copy :

    """

#    delta padding to match dimension with padded input when computing the view
    if self.pad:
      mat_pad = self._pad(delta, self.size, self.stride)
    else  :
      mat_pad = delta

#    View on delta, I can use this to modify it
    delta_view = self._asStride(mat_pad, self.size, self.stride)

    self.delta *= self.gradient(self.output, copy=copy)
    self.weights_updates = np.einsum("ijklmn, ijko -> lmno", self.view, self.delta)

#    out_c number of bias_updates.
    self.bias_updates    = self.delta.sum(axis = (0,1,2)) # shape = (channels_out,)

#     to access every pixel one at a time I need to create evry combinations of indexes
    b, w, h, kx, ky, c = delta_view.shape
    combo = itertools.product(range(b), range(w), range(h) , range(kx), range(ky), range(c))

#    Actual operation to be performed, it's basically the convolution of self.delta with filters.transpose
    operator = np.einsum("ijkl, mnol -> ijkmno", self.delta, self.weights)

#    Atomically modify , really slow as for maxpool and avgpool
    for i,j,k,l,m,n in combo:
      delta_view[i,j,k,l,m,n] += operator[i,j,k,l,m,n]

#    Here delta is updated correctly
    if self.pad :
      _ , w_pad, h_pad, _ = mat_pad.shape
      delta[:] = mat_pad[:, self.pad_top : w_pad-self.pad_bottom, self.pad_left : h_pad - self.pad_right ,:]
    else  :
      delta[:] = mat_pad


  def update(self, momentum=0., decay=0., lr=1e-2, lr_scale=1.):
    """
    update function for the convolution layer

    Parameters:
      momentum : float, default = 0., scale factor of weight update
      decay    : float, default = 0., determines the decay of weights_update
      lr       : float, default = 1e-02, learning rate of the layer
      lr_scale : float, default = 1., learning rate scale of the layer
    """
    lr *= lr_scale
    lr /= self.batch

    #Bias updates
    self.bias += self.bias_updates*lr

    #Weights_updates
    self.weights_updates += (-decay)*self.batch*self.filters
    self.filters         += lr * self.weights_updates
    self.weights_updates *= momentum


if __name__ == "__main__":

  import os

  from PIL import Image
  import pylab as plt

  from NumPyNet import activations

  img_2_float = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 1.)).astype(float)
  float_2_img = lambda im : ((im - im.min()) * (1./(im.max() - im.min()) * 255.)).astype(np.uint8)

  filename = os.path.join(os.path.dirname(__file__), '..', '..', 'data', 'dog.jpg')
  inpt = np.asarray(Image.open(filename), dtype=float)
  inpt.setflags(write=1)
  inpt = img_2_float(inpt)
#  Relu activation constrain
  inpt = inpt * 2 - 1

  inpt = np.expand_dims(inpt, axis = 0) # shape from (w, h, c) to (1, w, h, c)

  channels_out = 5
  size         = (3,3)
  stride       = (2,2)
  pad          = True

  layer_activation = activations.Relu()

  np.random.seed(123)

  b, w, h, c = inpt.shape
  filters    = np.random.uniform(-1., 1., size =  (*size,c, channels_out))
  bias       = np.random.uniform(-1., 1., size = (channels_out,))

  layer = Convolutional_layer(inputs=inpt.shape, channels_out=channels_out ,
                              weights=filters,
                              bias=bias,
                              activation=layer_activation,
                              size=size, stride=stride,
                              padding=pad)

  #FORWARD

  layer.forward(inpt)
  forward_out = layer.output.copy()

  #BACKWARD

  layer.delta  = np.ones(layer.out_shape())
  delta = np.zeros(shape = inpt.shape)
  layer.backward(delta)

  #Visualization

  fig, (ax1, ax2, ax3) = plt.subplots(nrows=1, ncols=3, figsize = (10,5))
  fig.subplots_adjust(left=0.1, right=0.95, top=0.95, bottom=0.15)

  fig.suptitle(('Convolutional Layer \n\n activation: {}, '+
                'size : {}, stride : {}, '+
                'output channels : {}').format(layer_activation.name, size, stride, channels_out))

  ax1.imshow(float_2_img(inpt[0]))
  ax1.set_title("Original image")
  ax1.axis("off")
                        #here every filter effect on the image can be shown
  ax2.imshow(float_2_img(forward_out[0,:,:,0]))
  ax2.set_title("Forward")
  ax2.axis("off")

  ax3.imshow(float_2_img(delta[0]))
  ax3.set_title("Backward")
  ax3.axis("off")

  plt.show()

