#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import operator

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Box (object):

  def __init__ (self, coords=None):

    if coords is not None:

      try:

        self.x, self.y, self.w, self.h = coords

      except ValueError:
        class_name = self.__class__.__name__
        raise ValueError('{0}: inconsistent input shape. Expected a 4D (x, y, w, h) shapes and given {1}'.format(class_name, coords))

    else:
      self.x, self.y, self.w, self.h = (None, None, None, None)

  @property
  def box(self):
    return (self.x, self.y, self.w, self.h)

  def __iter__ (self):
    yield self.x - self.w * .5
    yield self.y - self.h * .5
    yield self.x + self.w * .5
    yield self.y + self.h * .5


  def __eq__ (self, other):
    return isinstance(other, Box) and tuple(self) == tuple(other)

  def __ne__ (self, other):
    return not (self == other)

  def __repr__ (self):
    return type(self).__name__ + repr(tuple(self))

  def _overlap (self, x1, w1, x2, w2):
    '''
    Compute the overlap between (left, top) | (right, bottom) of the coordinates
    '''
    half_w1, half_w2 = w1 * .5, w2 * .5
    l1, l2 = x1 - half_w1, x2 - half_w2
    r1, r2 = x1 + half_w1, x2 + half_w2

    return min(r1, r2) - max(l1, l2)

  def intersection (self, other):
    '''
    Common area between boxes
    '''

    if not isinstance(other, Box):
      raise ValueError('intersection requires a Box object')

    w = self._overlap(self.x, self.w, other.x, other.w)
    h = self._overlap(self.y, self.h, other.y, other.h)

    w = w if w > 0. else 0.
    h = h if h > 0. else 0.
    return w * h

  __and__ = intersection

  def union (self, other):
    '''
    Full area without intersection
    '''

    if not isinstance(other, Box):
      raise ValueError('union requires a Box object')

    return self.area + other.area - self.intersection(other)

  __add__ = union

  def iou (self, other):
    '''
    Intersection over union
    '''

    if not isinstance(other, Box):
      raise ValueError('iou requires a Box object')

    return self.intersection(other) / self.union(other)

  __sub__ = iou

  def rmse (self, other):
    '''
    Root mean square error of the boxes
    '''

    if not isinstance(other, Box):
      raise ValueError('rmse requires a Box object')

    diffs = tuple(map(operator.sub, self, other))
    dot = sum(map(operator.mul, diffs, diffs))
    return dot**(.5)

  @property
  def center(self):
    '''
    In the current storage the x,y are the center of the box
    '''
    x, y, _, _ = self._object.box
    return (x, y)

  @property
  def dimensions(self):
    '''
    In the current storage the w,h are the dimensions of the rectangular box
    '''
    _, _, w, h = self._object.box
    return (w, h)

  @property
  def area(self):
    '''
    Compute the are of the box
    '''
    return self.w * self.h

  @property
  def coords(self):
    '''
    Return box coordinates in clock order (left, top, right, bottom)
    '''
    x, y, w, h = self.box
    half_w, half_h = w * .5, h * .5
    return (x - half_w, y - half_h, x + half_w, y + half_h)

  def __str__(self):
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

