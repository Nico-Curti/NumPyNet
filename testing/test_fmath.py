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

import timeit
import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Fast Math functions testing'

def _timing_np (func, args=None):
  SETUP_CODE = '''
import numpy as np

def np_pow2 (x):
  return 2 ** x

def np_rsqrt (x):
  return 1. / np.sqrt(x)

func = eval('{func}')
arr  = range(0, 10000)
'''.format(**{'func' : func})

  if args is not None:
    TEST_CODE = '''
y = map(lambda x : func(x, {args}), arr)
'''.format(**{'args' : args})
  else:
    TEST_CODE = '''
y = map(func, arr)
'''

  return timeit.repeat(setup=SETUP_CODE,
                       stmt=TEST_CODE,
                       repeat=100,
                       number=1000)

def _timing_fmath (func, args=None):
  SETUP_CODE = '''
from __main__ import {func}

arr = range(0, 10000)
'''.format(**{'func' : func})

  if args is not None:
    TEST_CODE = '''
y = map(lambda x : {func}(x, {args}), arr)
'''.format(**{'func' : func, 'args' : args})
  else:
    TEST_CODE = '''
y = map({func}, arr)
'''.format(**{'func' : func})

  return timeit.repeat(setup=SETUP_CODE,
                       stmt=TEST_CODE,
                       repeat=100,
                       number=1000)


def timeit_fmat ():

  np_pow2     = min(_timing_np(    'np_pow2'   ))
  fmath_pow2  = min(_timing_fmath( 'pow2'      ))
  np_exp      = min(_timing_np(    'np.exp'    ))
  fmath_exp   = min(_timing_fmath( 'exp'       ))
  np_pow      = min(_timing_np(    'np.power', .2))
  fmath_pow   = min(_timing_fmath( 'pow'     , .2))
  np_log2     = min(_timing_np(    'np.log2'   ))
  fmath_log2  = min(_timing_fmath( 'log2'      ))
  np_log10    = min(_timing_np(    'np.log10'  ))
  fmath_log10 = min(_timing_fmath( 'log10'     ))
  np_log      = min(_timing_np(    'np.log'    ))
  fmath_log   = min(_timing_fmath( 'log'       ))
  np_atanh    = min(_timing_np(    'np.arctanh'))
  fmath_atanh = min(_timing_fmath( 'atanh'     ))
  np_tanh     = min(_timing_np(    'np.tanh'   ))
  fmath_tanh  = min(_timing_fmath( 'tanh'      ))
  np_sqrt     = min(_timing_np(    'np.sqrt'   ))
  fmath_sqrt  = min(_timing_fmath( 'sqrt'      ))
  np_rsqrt    = min(_timing_np(    'np_rsqrt'  ))
  fmath_rsqrt = min(_timing_fmath( 'rsqrt'     ))

  print('                   CMath     FMath')
  print('pow2  function : {:.9f}     {:.9f}'.format(np_pow2, fmath_pow2))
  print('exp   function : {:.9f}     {:.9f}'.format(np_exp, fmath_exp))
  print('pow   function : {:.9f}     {:.9f}'.format(np_pow, fmath_pow))
  print('log2  function : {:.9f}     {:.9f}'.format(np_log2, fmath_log2))
  print('log10 function : {:.9f}     {:.9f}'.format(np_log10, fmath_log10))
  print('log   function : {:.9f}     {:.9f}'.format(np_log, fmath_log))
  print('atanh function : {:.9f}     {:.9f}'.format(np_atanh, fmath_atanh))
  print('tanh  function : {:.9f}     {:.9f}'.format(np_tanh, fmath_tanh))
  print('sqrt  function : {:.9f}     {:.9f}'.format(np_sqrt, fmath_sqrt))
  print('rsqrt function : {:.9f}     {:.9f}'.format(np_rsqrt, fmath_rsqrt))

  #                    CMath           FMath
  # pow2  function : 0.000387600     0.000341400
  # exp   function : 0.000342000     0.000346200
  # pow   function : 0.000583300     0.000539600
  # log2  function : 0.000380200     0.000382200
  # log10 function : 0.000384900     0.000341400
  # log   function : 0.000380500     0.000342200
  # atanh function : 0.000427400     0.000377600
  # tanh  function : 0.000372500     0.000375100
  # sqrt  function : 0.000372100     0.000341400
  # rsqrt function : 0.000376100     0.000341800

def test_pow2 ():

  x = np.pi
  assert np.isclose(2**x, pow2(x), atol=1e-3)

def test_exp ():

  x = np.pi
  assert np.isclose(np.exp(x), exp(x), atol=1e-5)

def test_pow ():

  x = np.pi
  assert np.isclose(x**.2, pow(x, .2), atol=1e-4)

def test_log2 ():

  x = np.pi
  assert np.isclose(np.log2(x), log2(x), atol=1e-4)

def test_log10 ():

  x = np.pi
  assert np.isclose(np.log10(x), log10(x), atol=1e-3)

def test_log ():

  x = np.pi
  assert np.isclose(np.log(x), log(x), atol=1e-4)

def test_arctanh ():

  c = 1e-2
  x = np.pi
  assert np.isclose(np.arctanh(x*c), atanh(x*c), atol=1e-4)

def test_tanh ():

  x = np.pi
  assert np.isclose(np.tanh(x), tanh(x), atol=1e-5)

def test_sqrt ():

  x = np.pi
  assert np.isclose(np.sqrt(x), sqrt(x), atol=1e-5)

def test_rsqrt ():

  x = np.pi
  assert np.isclose(1. / np.sqrt(x), rsqrt(x), atol=1e-5)

