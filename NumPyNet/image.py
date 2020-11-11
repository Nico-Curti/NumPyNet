#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import cv2
import numpy as np

from NumPyNet.image_utils import image_utils
from NumPyNet.image_utils import normalization

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Image (object):

  '''
  Constructor of the image object. If filename the load function loads the image file.

  Parameters
  ----------
    filename : str (default=None)
      Image filename
  '''

  def __init__ (self, filename=None):

    if filename is not None:
      self.load(filename)

    else:
      self._data = None

  @property
  def shape (self):
    '''
    Get the image dimensions
    '''
    return self._data.shape

  def add_single_batch (self):
    '''
    Add batch dimension for testing layer

    Returns
    -------
      self
    '''
    self._data = np.expand_dims(self._data, axis=0)
    return self

  def remove_single_batch (self):
    '''
    Remove batch dimension for testing layer

    Returns
    -------
      self
    '''
    self._data = np.squeeze(self._data, axis=0)
    return self


  def _image2cv (self, img):
    '''
    Convert image from image-fmt to opencv fmt

    Parameters
    ----------
      img : array-like
        Input image to convert

    Returns
    -------
      cv_img : array-like
        Image as uint8 nd-array

    Notes
    -----
    .. note::
      The channels are automatically converted from
      RGB 2 BGR for OpenCV compatibility
    '''
    # constrain
    img = np.clip(img, 0., 1.)

    # normalize between [0, 255]
    img *= 255.

    # rgb 2 bgr
    img = img[..., ::-1].astype('uint8')

    return img


  def _cv2image (self, img):
    '''
    Convert image from opencv-fmt to image-fmt

    Parameters
    ----------
      img : array-like
        Input image to convert

    Returns
    -------
      Image_img : array-like
        Image as float [0., 1.] nd-array

    Notes
    -----
    .. note::
      The channels are automatically converted from
      BGR 2 RGB for Image compatibility
    '''
    img = img.astype('float64')

    # bgr 2 rgb
    img = img[..., ::-1]

    # normalize between [0, 1]
    img *= 1. / 255.

    return img

  def __array__ (self):
    '''
    Compatibility with numpy array.
    In this way np.array(Image_object) is a valid 3D array and you can
    also simply call plt.imshow(Image_object) without other intermediate steps.
    '''
    return self.data

  def _get_color (self, x, max):
    '''
    Get the color
    '''

    ratio = (x / max) * len(image_utils.num_box_colors - 1)
    i, j = np.floor(ratio), np.ceil(ratio)

    ratio -= i
    r, g, b = image_utils.colors[i]

    return ( (1. - ratio) * b + ratio * b,
             (1. - ratio) * g + ratio * g,
             (1. - ratio) * r + ratio * r )

  def get (self):
    '''
    Return the data object as a numpy array

    Returns
    -------
      data : array-like
        Image data as numpy array
    '''
    return self._data


  def load (self, filename):
    '''
    Read Image from file

    Parameters
    ----------
      filename : str
        Image filename path

    Returns
    -------
      self
    '''

    if not os.path.isfile(filename):
      raise IOError('Could not open or find the data file. Given: {}'.format(filename))

    # read image from file
    img = cv2.imread(filename,  cv2.IMREAD_COLOR)

    self._data = self._cv2image(img)
    return self


  def standardize (self, means, process=normalization.normalize):
    '''
    Remove or add train mean-image from current image

    Parameters
    ----------
      means : array_like
        Array of means to apply to the image

      process : normalization (int, default = normalize)
        Switch between normalization (0) and denormalization (1)

    Returns
    -------
      self
    '''
    if process is normalization.normalize:
      self._data += means

    elif process is normalization.denormalize:
      self._data -= means

    return self

  def rescale (self, var, process=normalization.normalize):
    '''
    Divide or multiply by train variance-image

    Parameters
    ----------
      variances : array_like
        Array of variances to apply to the image

      process : normalization (int)
        Switch between normalization and denormalization

    Returns
    -------
      self
    '''
    if process is normalization.normalize:
      inv_vars = 1. / var
      self._data *= inv_vars

    elif process is normalization.denormalize:
      self._data *= var

    return self

  def scale (self, scaling, process=normalization.normalize):
    '''
    Scale image values

    Parameters
    ----------
      scale : float
        Scale factor to apply to the image

      process : normalization (int, default = normalize)
        Switch between normalization (0) and denormalization (1)

    Returns
    -------
      self
    '''
    if process is normalization.normalize:
      self._data *= scaling

    elif process is normalization.denormalize:
      inv_scaling = 1. / scaling
      self._data *= inv_scaling

    return self

  def scale_between (self, minimum, maximum):
    '''
    Rescale image value between min and max

    Parameters
    ----------
      minimum : float (default = 0.)
        Min value

      maximum : float (default = 1.)
        Max value

    Returns
    -------
      self
    '''
    diff = maximum - minimum
    self._data = self._data * diff + minimum
    return self

  def mean_std_norm (self):
    '''
    Normalize the current image as

    .. code-block:: python

      image = (image - mean) / variance


    Returns
    -------
      self
    '''
    mean = np.mean(self._data)
    var  = 1. / np.var(self._data)
    self._data = (self._data - mean) * var
    return self

  def flip (self, axis=-1):
    '''
    Flip the image along given axis (0 - horizontal, 1 - vertical)

    Parameters
    ----------
      axis : int (default=0)
        Axis to flip

    Returns
    -------
      self
    '''
    cv2.flip(self._data, axis)
    return self

  def transpose (self):
    '''
    Transpose width and height

    Returns
    -------
      self
    '''
    self._data = self._data.transpose(1, 0, 2)
    return self

  def rotate (self, angle):
    '''
    Rotate the image according to the given angle in degree fmt.

    Parameters
    ----------
      angle : float
        Angle in degree fmt

    Returns
    -------
      rotated : Image
        Rotated image

    Note
    ----
    .. note::
      This rotation preserves the original size so some original parts can be removed
      from the rotated image.
      See 'rotate_bound' for a conservative rotation.

    References
    ----------
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''
    h, w = self._data.shape[:2]

    center = (w // 2, h // 2)
    rotation_matrix = cv2.getRotationMatrix2D(center=center, angle=angle, scale=1.0)
    self._data = cv2.warpAffine(self._data, M=rotation_matrix, dsize=(w, h))
    return self

  def rotate_bound (self, angle):
    '''
    Rotate the image according to the given angle in degree fmt.

    Parameters
    ----------
      angle : float
        Angle in degree fmt

    Returns
    -------
      rotated : Image
        Rotated image

    Note
    ----
    .. note::
      This rotation preserves the original image, so the output can be greater than the
      original size.
      See 'rotate' for a rotation which preserves the size.

    References
    ----------
    https://www.pyimagesearch.com/2017/01/02/rotate-images-correctly-with-opencv-and-python/
    '''

    # grab the dimensions of the image and then determine the
    # center
    (h, w) = self._data.shape[:2]
    (cX, cY) = (w // 2, h // 2)

    # grab the rotation matrix (applying the negative of the
    # angle to rotate clockwise), then grab the sine and cosine
    # (i.e., the rotation components of the matrix)
    M = cv2.getRotationMatrix2D(center=(cX, cY), angle=angle, scale=1.0)
    cos = np.abs(M[0, 0])
    sin = np.abs(M[0, 1])

    # compute the new bounding dimensions of the image
    nW = int((h * sin) + (w * cos))
    nH = int((h * cos) + (w * sin))

    # adjust the rotation matrix to take into account translation
    M[0, 2] += (nW / 2) - cX
    M[1, 2] += (nH / 2) - cY

    # perform the actual rotation and return the image
    self._data = cv2.warpAffine(self._data, M=M, dsize=(nW, nH))
    return self

  def crop (self, dsize, size):
    '''
    Crop the image according to the given dimensions [dsize[0] : dsize[0] + size[0], dsize[1] : dsize[1] + size[1]]

    Parameters
    ----------
      dsize : 2D iterable
        (X, Y) of the crop

      size : 2D iterable
        (width, height) of the crop

    Returns
    -------
      cropped : Image
        Cropped image
    '''
    dx, dy = dsize
    sx, sy = size

    self._data = self._data[dx : dx + sx, dy : dy + sy]
    return self

  def rgb2rgba (self):
    '''
    Add alpha channel to the original image

    Returns
    -------
      self

    Notes
    -----
    .. note::
      Pay attention to the value of the alpha channel!
      OpenCV does not set its values to null but they are and empty (garbage) array.
    '''
    self._data = cv2.cvtColor(self._data, cv2.COLOR_RGB2RGBA)
    return self


  def show (self, window_name, ms=0, fullscreen=None):
    '''
    show the image

    Parameters
    ----------
      window_name : str
        Name of the plot

      ms : int (default=0)
        Milliseconds to wait

    Returns
    -------
      check : bool
        True if everything is ok
    '''
    img = self._image2cv(self._data)

    # show image
    if ms == 0:
      print('Press ESC to close the window', flush=True)

    cv2.imshow(window_name, img)

    if fullscreen:
      cv2.setWindowProperty(window_name, cv2.WND_PROP_FULLSCREEN, cv2.WINDOW_FULLSCREEN)

    cv2.waitKey(ms)

    return True


  def save (self, filename):
    '''
    save the image

    Parameters
    ----------
      filename : str
        Output filename of the image

    Returns
    -------
      True if everything is ok
    '''
    img = self._image2cv(self._data)

    # save image
    cv2.imwrite(filename + '.png', img)

    return True

  def from_numpy_matrix (self, array):
    '''
    Use numpy array as the image

    Parameters
    ----------
      array : array_like
        buffer of the input image as (width, height, channel)

    Returns
    -------
      self
    '''
    self._data = array
    return self

  def from_frame (self, array):
    '''
    Use opencv frame array as the image
    '''
    self._data = self._cv2image(array)
    return self

  def resize (self, dsize=None, scale_factor=(None, None)):
    '''
    Resize the image according to the new shape given

    Parameters
    ----------
      dsize : 2D iterable (default=None)
        Destination size of the image

      scale_factor : 2D iterable (default=(None, None))
        width scale factor, height scale factor

    Returns
    -------
      res : Image
        Resized Image

    Notes
    -----
    .. note::
      The resize is performed using the LANCZOS interpolation.
    '''
    fx, fy = scale_factor

    return cv2.resize(self._data, dsize=dsize, fx=fx, fy=fy,
                      interpolation=cv2.INTER_LANCZOS4)

  def letterbox (self, net_dim):
    '''
    resize image with unchanged aspect ratio using padding

    Parameters
    ----------
      net_dim : 2D iterable
        width and height outputs

    Returns
    -------
      resized : Image
        Resized Image
    '''
    resized = Image()

    img_h, img_w, _ = self._data.shape
    net_w, net_h = net_dim

    mins = min(net_w / img_w, net_h / img_h)

    new_w, new_h = int(img_w * mins), int(img_h * mins)

    resized_image = cv2.resize(self._data, (new_w, new_h),
                               interpolation=cv2.INTER_LANCZOS4)

    delta_w, delta_h = net_w - new_w, net_h - new_h
    top,    left  = delta_h // 2,  delta_w // 2
    bottom, right = delta_h - top, delta_w - left

    resized._data = cv2.copyMakeBorder(resized_image, top, bottom, left, right,
                                cv2.BORDER_CONSTANT,
                                value=(0.5, 0.5, 0.5))

    return resized

  def draw_detections (self, dets, thresh, names):
    '''
    Draw the detections into the current image

    Parameters
    ----------
      dets : Detection list
        List of pre-computed detection objects

      thresh : float
        Probability threshold to filter the boxes

      names : iterable
        List of object names as strings

    Returns
    -------
      self
    '''
    width = 1 if self.height < 167 else int(self.height * 6e-3)

    for d in dets:

      index = np.where(d.prob > thresh)[0]
      labels = None

      if index:
        labels = ', '.join(names[index])

        perf = '\n'.join(['{0}: {1:.3f}%'.format(name, prob * 100.)
                          for name, prob in zip(names[index], d.prob[index])
                          ])
        print(perf)

      if labels is not None:

        num_obj = labels.count(',') + 1
        offset = num_obj * 123457 % len(names)

        r, g, b = self._get_color(offset, len(names))

        left, right, top, bottom = ( (d.bbox.x - d.bbox.w * .5) * self.width,
                                     (d.bbox.x + d.bbox.w * .5) * self.width,
                                     (d.bbox.y - d.bbox.h * .5) * self.height,
                                     (d.bbox.y + d.bbox.h * .5) * self.height
                                    )

        left   = np.clip(left,   0, self.width)
        right  = np.clip(right,  0, self.width)
        top    = np.clip(top,    0, self.height)
        bottom = np.clip(bottom, 0, self.height)

        # detection box
        cv2.rectangle(img=self._data, pt1=(left, top), pt2=(right, bottom), color=(r, g, b), thickness=width)

        # get string text size
        (label_w, label_h), baseline = cv2.getTextSize(labels, fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, thickness=1)

        # boxe around the string
        cv2.rectangle(img=self._data, pt1=(left, top - label_h - baseline), pt2=(left + label_w, top), color=(r, g, b), thickness=cv2.FILLED)

        # label string
        cv2.putText(img=self._data, text=labels, org=(left, top + baseline - label_h), fontFace=cv2.FONT_HERSHEY_COMPLEX_SMALL, fontScale=1, color=(0, 0, 0), thickness=1, lineType=1)

    return self

  @property
  def width(self):
    '''
    Get the image width
    '''
    return self._data.shape[0]

  @property
  def height(self):
    '''
    Get the image height
    '''
    return self._data.shape[1]

  @property
  def channels(self):
    '''
    Get the image number of channels
    '''
    return self._data.shape[2]


if __name__ == '__main__':


  filename = os.path.join(os.path.dirname(__file__), '..', 'data', 'dog.jpg')

  img = Image(filename)
  resized = img.letterbox(net_dim=(416, 416))

  cv2.namedWindow('Test', cv2.WINDOW_NORMAL)
  resized.show('Test')

  cv2.destroyAllWindows()
