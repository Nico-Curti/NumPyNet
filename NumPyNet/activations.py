#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Activation Functions'

class Activations(object):

  def __init__(self, name, leaky_coef):
    self._name = name
    self._leaky_coef = leaky_coef

  @staticmethod
  def activate(x, copy = False):
    pass

  @staticmethod
  def gradient(x, copy = False):
    pass

  @property
  def name(self):
    return self._name

  def __str__(self):
    return '{} activation function'.format(self._name)


class Logistic (Activations):

  def __init__(self):
    super(Logistic, self).__init__('Logistic', None)

  @staticmethod
  def activate (x, copy=False):
    return 1. / (1. + np.exp(-x))

  @staticmethod
  def gradient (x, copy=False):
    return (1. - x) * x

class Loggy (Activations):

  def __init__(self):
    super(Loggy, self).__init__('Loggy', None)

  @staticmethod
  def activate (x, copy=False):
    return 2. / (1. + np.exp(-x)) - 1.

  @staticmethod
  def gradient (x, copy=False):
    return 2. * (1. - (x + 1.) * .5) * (x + 1.) * .5

class Relu (Activations):

  def __init__(self):
    super(Relu, self).__init__('Relu', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x < 0.] = 0.
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x > 0.] = 1.
    y[x <= 0.] = 0.
    return y

class Elu (Activations):

  def __init__(self):
    super(Elu, self).__init__('Elu', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >= 0.] *= y[x >= 0.]
    y[x <  0.] *= np.exp(y[x < 0.] - 1.)
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    # MISS
    return

class Relie (Activations):

  def __init__(self):
    super(Relie, self).__init__('Relie', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x <= 0.] *= 1e-2
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >  0.] = 1
    y[x <= 0.] = 1e-2
    return y

class Ramp (Activations):

  def __init__(self):
    super(Ramp, self).__init__('Ramp', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >  0.] *= y[x > 0.]
    y[x <= 0.]  = 0
    return y + .1 * y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x > 0.] += .1
    return y

class Linear (Activations):

  def __init__(self):
    super(Linear, self).__init__('Linear', None)

  @staticmethod
  def activate(x, copy=False):
    return x

  @staticmethod
  def gradient(x, copy=False):
    return np.ones(shape=x.shape, dtype=float)

class Tanh (Activations):

  def __init__(self):
    super(Tanh, self).__init__('Tanh', None)

  @staticmethod
  def activate(x, copy=False):
    exp = np.exp(2. * x)
    return (exp - 1.) / (exp + 1.)

  @staticmethod
  def gradient(x, copy=False):
    return 1. - x * x

class Plse (Activations):

  def __init__(self):
    super(Plse, self).__init__('Plse', None)

  @staticmethod
  def activate (x, copy=False):
    y = x.copy()
    y[x < -4.] = (y[x < -4.] + 4.) * 1e-2
    y[x >  4.] = (y[x >  4.] - 4.) * 1e-2 + 1.
    y[x >= -4 & x <= 4.] = y[x >= -4 & x <= 4.] * .125 + .5
    return y

  @staticmethod
  def gradient (x, copy=False):
    y = x.copy()
    y[x <  0. | x >  1.] = 1e-2
    y[x >= 0. | x <= 1.] = .125
    return y

class Leaky (Activations):

  def __init__(self):
    super(Leaky, self).__init__('Leaky', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x <= 0.] *= self.leaky_coef
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >  0.] = 1.
    y[x <= 0.] = self.leaky_coef
    return y

class Stair (Activations):

  def __init__(self):
    super(Stair, self).__init__('Stair', None)

class HardTan (Activations):

  def __init__(self):
    super(HardTan, self).__init__('HardTan', None)

  @staticmethod
  def activate(x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x < 1.] = -1.
    y[x > 1.] = 1.
    return y

  @staticmethod
  def gradient(x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x > -1. and x < 1.] = 1.
    y[x < -1. and x > 1.] = 0.
    return y

class LhTan (Activations):

  def __init__(self):
    super(LhTan, self).__init__('LhTan', None)

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x < 0.] *= 1e-3
    y[x > 1.]  = (y[y > 1.] - 1.) * 1e-3
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x.copy()

    y[x >  0. & x <  1.] = 1
    y[x <= 0. | x >= 1.] = 1e-3
    return y

class Selu (Activations):

  def __init__(self):
    super(Selu, self).__init__('Selu', None)

