#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

class Activations(object):

  def __init__(self, name, leaky_coef):
    self._name = name
    self._leaky_coef = leaky_coef

  def activate(x, copy = False):
    pass

  def gradient(x, copy = False):
    pass

  @property
  def name(self):
    return self._name


class Linear(Activations):

  def __init__(self):
    super(Linear, self).__init__('Linear', None)

  @staticmethod
  def activate(x, copy = False):
    return x

  @staticmethod
  def gradient(x, copy = False):
    return np.ones(shape = x.shape)



class Relu(Activations):

  def __init__(self):
    super(Relu, self).__init__('Relu', None)

  @staticmethod
  def activate(x, copy = False):
    if copy : tmp = x.copy()
    else    : tmp = x
    tmp[x < 0.] = 0.
    return tmp

  @staticmethod
  def gradient(x, copy = False):
    if copy: tmp = x.copy()
    else   : tmp = x
    tmp[x >  0.] = 1.
    tmp[x <= 0.] = 0.
    return tmp


class Tanh(Activations):

  def __init__(self):
    super(Tanh, self).__init__('Tanh', None)

  @staticmethod
  def activate(x, copy = False):
    exp = np.exp(x * 2.) #It can becomes inf with large numbers! That breaks the function
    return (exp - 1.) / (exp + 1.)

  @staticmethod
  def gradient(x, copy = False):
    return 1. - x * x


class Hardtan(Activations):

  def __init__(self):
    super(Hardtan, self).__init__('HardTan', None)

  @staticmethod
  def activate(x, copy = False):
    if copy : tmp = x.copy()
    else    : tmp = x
    tmp[x < 1.] = -1.
    tmp[x > 1.] = 1.
    return tmp

  @staticmethod
  def gradient(x, copy = False):
    if copy : tmp = x.copy()
    else    : tmp = x
    tmp[x > -1. and x < 1.] = 1.
    tmp[x < -1. and x > 1.] = 0.
    return tmp


class Logistic(Activations):

  def __init__(self):
    super(Logistic, self).__init__('Logistic', None)

  @staticmethod
  def activate(x, copy = False):
    return 1. / (1. + np.exp(-x))
  @staticmethod
  def gradient(x, copy = False):
    return (1. - x) * x


class Loggy(Activations):

  def __init__(self):
    super(Loggy, self).__init__('Loggy', None)

  @staticmethod
  def activate(x, copy = False):
    return 2. / (1. + np.exp(-x)) - 1.

  @staticmethod
  def gradient(x, copy = False):
    return 2. * (1. - (x + 1.) * .5) * (x + 1.) * .5



