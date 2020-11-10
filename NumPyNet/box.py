#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import operator
from functools import wraps

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Box (object):

  '''
  Detection box class

  Parameters
  ----------
    coords : tuple (default=None)
      Box Coordinates as (x, y, w, h)

  Example
  -------
  >>> import pylab as plt
  >>> from matplotlib.patches import Rectangle
  >>>
  >>> b1 = Box((.5, .3, .2, .1))
  >>> x_1, y_1, w_1, h_1 = b1.box
  >>> left_1, top_1, right_1, bottom_1 = b1.coords
  >>>
  >>> print('Box1: {}'.format(b1))
  >>>
  >>> b2 = Box((.4, .5, .2, .5))
  >>> x_2, y_2, w_2, h_2 = b2.box
  >>> left_2, top_2, right_2, bottom_2 = b2.coords
  >>>
  >>> print('Box2: {}'.format(b2))
  >>>
  >>> print('Intersection: {:.3f}'.format(b1.intersection(b2)))
  >>> print('Union: {:.3f}'.format(b1.union(b2)))
  >>> print('IOU: {:.3f}'.format(b1.iou(b2)))
  >>> print('rmse: {:.3f}'.format(b1.rmse(b2)))
  >>>
  >>> plt.figure()
  >>> axis = plt.gca()
  >>> axis.add_patch(Rectangle(xy=(left_1, top_1),
  >>>                          width=w_1, height=h_1,
  >>>                          alpha=.5, linewidth=2, color='blue'))
  >>> axis.add_patch(Rectangle(xy=(left_2, top_2),
  >>>                          width=w_2, height=h_2,
  >>>                          alpha=.5, linewidth=2, color='red'))
  '''

  def __init__ (self, coords=None):

    if coords is not None:

      try:

        self.x, self.y, self.w, self.h = coords

      except ValueError:
        class_name = self.__class__.__name__
        raise ValueError('{0}: inconsistent input shape. Expected a 4D (x, y, w, h) shapes and given {1}'.format(class_name, coords))

    else:
      self.x, self.y, self.w, self.h = (None, None, None, None)

  def _is_box (func):
    '''
    Decorator function to check if the input variable is a Box object
    '''

    @wraps(func)
    def _ (self, b):

      if isinstance(b, self.__class__):
        return func(self, b)
      else:
        raise ValueError('Box functions can be applied only on other Box objects')

    return _

  @property
  def box(self):
    '''
    Get the box coordinates

    Returns
    -------
      coords : tuple
        Box coordinates as (x, y, w, h)
    '''
    return (self.x, self.y, self.w, self.h)

  def __iter__ (self):
    '''
    Iter over coordinates as (x, y, w, h)
    '''

    yield self.x
    yield self.y
    yield self.w
    yield self.h


  def __eq__ (self, other):
    '''
    Check if the box coordinates are equal
    '''
    return isinstance(other, Box) and tuple(self) == tuple(other)

  def __ne__ (self, other):
    '''
    Check if the box coordinates are NOT equal
    '''
    return not (self == other)

  def __repr__ (self):
    '''
    Object representation
    '''
    return type(self).__name__ + repr(tuple(self))

  def _overlap (self, x1, w1, x2, w2):
    '''
    Compute the overlap between (left, top) | (right, bottom) of the coordinates

    Parameters
    ----------
      x1 : float
        X coordinate

      w1 : float
        W coordinate

      x2 : float

      w2 : float

    Returns
    -------
      overlap : float
        The overlapping are between the two boxes
    '''
    half_w1, half_w2 = w1 * .5, w2 * .5
    l1, l2 = x1 - half_w1, x2 - half_w2
    r1, r2 = x1 + half_w1, x2 + half_w2

    return min(r1, r2) - max(l1, l2)

  @_is_box
  def intersection (self, other):
    '''
    Common area between boxes

    Parameters
    ----------
      other : Box
        2nd term of the evaluation

    Returns
    -------
      intersection : float
        Intersection area of two boxes
    '''

    w = self._overlap(self.x, self.w, other.x, other.w)
    h = self._overlap(self.y, self.h, other.y, other.h)

    w = w if w > 0. else 0.
    h = h if h > 0. else 0.
    return w * h

  __and__ = intersection

  @_is_box
  def union (self, other):
    '''
    Full area without intersection

    Parameters
    ----------
      other : Box
        2nd term of the evaluation

    Returns
    -------
      union : float
        Union area of the two boxes
    '''

    return self.area + other.area - self.intersection(other)

  __add__ = union

  @_is_box
  def iou (self, other):
    '''
    Intersection over union

    Parameters
    ----------
      other : Box
        2nd term of the evaluation

    Returns
    -------
      iou : float
        Intersection over union between boxes
    '''

    union = self.union(other)

    return self.intersection(other) / union if union != 0. else float('nan')

  __sub__ = iou

  @_is_box
  def rmse (self, other):
    '''
    Root mean square error of the boxes

    Parameters
    ----------
      other : Box
        2nd term of the evaluation

    Returns
    -------
      rmse : float
        Root mean square error of the boxes
    '''

    diffs = tuple(map(operator.sub, self, other))
    dot = sum(map(operator.mul, diffs, diffs))
    return dot**(.5)

  @property
  def center(self):
    '''
    In the current storage the x,y are the center of the box

    Returns
    -------
      center : tuple
        Center of the current box.
    '''
    x, y, _, _ = self._object.box
    return (x, y)

  @property
  def dimensions(self):
    '''
    In the current storage the w,h are the dimensions of the rectangular box

    Returns
    -------
      dims : tuple
        Dimensions of the current box as (width, height).
    '''
    _, _, w, h = self._object.box
    return (w, h)

  @property
  def area(self):
    '''
    Compute the are of the box

    Returns
    -------
      area : float
        Area of the current box.
    '''
    return self.w * self.h

  @property
  def coords(self):
    '''
    Return box coordinates in clock order (left, top, right, bottom)

    Returns
    -------
      coords : tuple
        Coordinates as (left, top, right, bottom)
    '''
    x, y, w, h = self.box
    half_w, half_h = w * .5, h * .5
    return (x - half_w, y - half_h, x + half_w, y + half_h)

  def __str__(self):
    '''
    Printer
    '''
    fmt = '(left={0:.3f}, bottom={1:.3f}, right={2:.3f}, top={3:.3f})'.format(*self.coords)
    return fmt


if __name__ == '__main__':

  import pylab as plt
  from matplotlib.patches import Rectangle

  b1 = Box((.5, .3, .2, .1))
  x_1, y_1, w_1, h_1 = b1.box
  left_1, top_1, right_1, bottom_1 = b1.coords

  print('Box1: {}'.format(b1))

  b2 = Box((.4, .5, .2, .5))
  x_2, y_2, w_2, h_2 = b2.box
  left_2, top_2, right_2, bottom_2 = b2.coords

  print('Box2: {}'.format(b2))

  print('Intersection: {:.3f}'.format(b1.intersection(b2)))
  print('Union: {:.3f}'.format(b1.union(b2)))
  print('IOU: {:.3f}'.format(b1.iou(b2)))
  print('rmse: {:.3f}'.format(b1.rmse(b2)))

  plt.figure()
  axis = plt.gca()
  axis.add_patch(Rectangle(xy=(left_1, top_1), width=w_1, height=h_1, alpha=.5, linewidth=2, color='blue'))
  axis.add_patch(Rectangle(xy=(left_2, top_2), width=w_2, height=h_2, alpha=.5, linewidth=2, color='red'))

  plt.show()

