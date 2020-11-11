#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import struct
from NumPyNet import ENABLE_FMATH

__author__ = ['Nico Curti']
__email__ = ['nico.curti2@unibo.it']


def pow2 (x):
  '''
  Fast math version of 'pow2' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  offset = 1     if x < 0.    else 0
  clipp  = -126. if x < -126. else x
  z      = clipp - int(clipp) + offset

  packed_x = struct.pack('i', int((1 << 23) * (clipp + 121.2740575 + 27.7280233 / (4.84252568 - z) - 1.49012907 * z)))
  return struct.unpack('f', packed_x)[0]

def exp (x):
  '''
  Fast math version of 'exp' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  return pow2(1.442695040 * x)

def log2 (x):
  '''
  Fast math version of 'log2' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  packed_x = struct.pack('f', x)
  i = struct.unpack('i', packed_x)[0]
  mx = (i & 0x007FFFFF) | 0x3f000000
  packed_x = struct.pack('i', mx)

  f = struct.unpack('f', packed_x)[0]
  i *= 1.1920928955078125e-7
  return i - 124.22551499 - 1.498030302 * f - 1.72587999 / (0.3520887068 + f);

def log (x):
  '''
  Fast math version of 'log' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  packed_x = struct.pack('f', x)
  i  = struct.unpack('i', packed_x)[0]
  y  = (i - 1064992212.25472) / (1092616192. - 1064992212.25472)
  ey = exp(y)
  y -= (ey - x) / ey
  ey = exp(y)
  y -= (ey - x) / ey
  ey = exp(y)
  y -= (ey - x) / ey
  ey = exp(y)
  y -= (ey - x) / ey
  return y

def pow (a, b):
  '''
  Fast math version of 'pow' function

  Parameters
  ----------
    a : float
      Base

    b : float
      Exponent

  Returns
  -------
    res : float
      Result of the function
  '''
  return pow2(b * log2(a))

def log10 (x):
  '''
  Fast math version of 'log10' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  packed_x = struct.pack('f', x)
  i   = struct.unpack('i', packed_x)[0]
  y   = (i - 1064992212.25472) / (1092616192. - 1064992212.25472)
  y10 = pow(10, y)
  y -= (y10 - x) / (2.302585092994046 * y10)
  y10 = pow(10, y)
  y -= (y10 - x) / (2.302585092994046 * y10)
  return y

def atanh (x):
  '''
  Fast math version of 'atanh' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  return .5 * log((1. + x) / (1. - x))

def tanh (x):
  '''
  Fast math version of 'tanh' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''
  e = exp(-2 * x)
  return (1. - e) / (1. + e)

def hardtanh (x):
  '''
  Fast math version of 'hardtanh' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''

  if   x >= -1 and x <= 1.: return x
  elif x <  -1            : return -1.
  else                    : return 1.

def sqrt (x):
  '''
  Fast math version of 'sqrt' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''

  xhalf = .5 * x

  packed_x = struct.pack('f', x)
  i = struct.unpack('i', packed_x)[0]  # treat float's bytes as int
  i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
  packed_i = struct.pack('i', i)
  y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float

  y = y * (1.5 - (xhalf * y * y))  # Newton's method
  y = y * (1.5 - (xhalf * y * y))  # Newton's method
  return x * y

def rsqrt (x):
  '''
  Fast math version of 'rsqrt' function

  Parameters
  ----------
    x : float
      Value to evaluate

  Returns
  -------
    res : float
      Result of the function
  '''

  xhalf = .5 * x

  packed_x = struct.pack('f', x)
  i = struct.unpack('i', packed_x)[0]  # treat float's bytes as int
  i = 0x5f3759df - (i >> 1)            # arithmetic with magic number
  packed_i = struct.pack('i', i)
  y = struct.unpack('f', packed_i)[0]  # treat int's bytes as float

  y = y * (1.5 - (xhalf * y * y))  # Newton's method
  y = y * (1.5 - (xhalf * y * y))  # Newton's method
  return y


if ENABLE_FMATH:

  import numpy as np

  np.pow2 = pow2
  np.exp = exp
  np.log2 = log2
  np.log = log
  np.pow = pow
  np.log10 = log10
  np.atanh = atanh
  np.tanh = tanh
  np.hardtanh = hardtanh
  np.sqrt = sqrt
  np.rsqrt = rsqrt
