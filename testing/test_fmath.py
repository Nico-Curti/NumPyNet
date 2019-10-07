# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from NumPyNet.fmath import pow2
from NumPyNet.fmath import exp
from NumPyNet.fmath import pow
from NumPyNet.fmath import log2
from NumPyNet.fmath import log10
from NumPyNet.fmath import log
from NumPyNet.fmath import atanh
from NumPyNet.fmath import tanh
from NumPyNet.fmath import sqrt
from NumPyNet.fmath import rsqrt

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Fast Math functions testing'

def test_fmath():

  x = np.pi
  c = 1e-2

  assert np.isclose(2**x,               pow2(x),    atol=1e-3)
  assert np.isclose(np.exp(x),          exp(x),     atol=1e-5)
  assert np.isclose(x**.2,              pow(x, .2), atol=1e-4)
  assert np.isclose(np.log2(x),         log2(x),    atol=1e-4)
  assert np.isclose(np.log10(x),        log10(x),   atol=1e-3)
  assert np.isclose(np.log(x),          log(x),     atol=1e-4)
  assert np.isclose(np.arctanh(x*c),    atanh(x*c), atol=1e-4)
  assert np.isclose(np.tanh(x),         tanh(x),    atol=1e-5)
  assert np.isclose(np.sqrt(x),         sqrt(x),    atol=1e-5)
  assert np.isclose(1. / np.sqrt(x),    rsqrt(x),   atol=1e-5)



if __name__ == '__main__':

  test_fmath()

