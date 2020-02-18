#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import struct
import itertools
import numpy as np

__author__  = ['Nico Curti']
__email__   = ['nico.curti2@unibo.it']


def read_tokenized_data (filename):

  with open(filename, 'rb') as fp:
    tokens = [struct.unpack("1022I", fp.read(4088)) for _ in range(1022)]

  tokens = sum(tokens, [])
  return tokens

def read_tokens (filename):

  with open(filename, 'r', encoding='utf-8') as fp:
    lines = fp.read()

  lines.replace('<NEWLINE>', '\n')
  lines = map(ord, lines)
  return lines

def get_rnn_data (tokens, offsets, characters, lenght, batch, steps):

  x = np.empty(shape=(batch * steps * characters, ))
  y = np.empty(shape=(batch * steps * characters, ))

  for i, j in itertools.product(range(batch), range(steps)):
    offset = offsets[i]
    curr = tokens[offset % lenght]
    next = tokens[(offset + 1 ) % lenght]

    idx = (j * batch + i) * characters

    x[idx + curr] = 1
    y[idx + next] = 1

    offsets[i] = (offset + 1) % lenght

    if curr >= characters or curr < 0 or next >= characters or next < 0:
      raise ValueError('Bad char')

  return (x, y)


def sample_array (arr):

  s = 1. / np.sum(arr)
  arr *= s

  r = np.random.uniform(low=0., high=1., size=(1,))

  cumulative = np.cumsum(arr)
  r = r - cumulative
  pos = np.where(np.sign(r[:-1]) != np.sign(r[1:]))[0] + 1

  return pos if pos else len(arr) - 1

def print_symbol (n, tokens=None):

  print('{} '.format(tokens[n]), end='', flush=True) if tokens is not None else print('{}'.format(n))


if __name__ == '__main__':

  print('Insert testing here')
