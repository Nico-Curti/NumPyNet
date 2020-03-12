#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Activations (object):

  def __init__ (self, name):
    self._name = name

  @staticmethod
  def activate (x, copy=False):
    raise NotImplementedError

  @staticmethod
  def gradient (x, copy=False):
    raise NotImplementedError

  @property
  def name (self):
    return self._name

  def __str__ (self):
    return '{} activation function'.format(self._name)


class Logistic (Activations):

  def __init__ (self):
    super(Logistic, self).__init__('Logistic')

  @staticmethod
  def activate (x, copy=False):
    return 1. / (1. + np.exp(-x))

  @staticmethod
  def gradient (x, copy=False):
    return (1. - x) * x


class Loggy (Activations):

  def __init__ (self):
    super(Loggy, self).__init__('Loggy')

  @staticmethod
  def activate (x, copy=False):
    return 2. / (1. + np.exp(-x)) - 1.

  @staticmethod
  def gradient (x, copy=False):
    return 2. * (1. - (x + 1.) * .5) * (x + 1.) * .5


class Relu (Activations):

  def __init__ (self):
    super(Relu, self).__init__('Relu')

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

  def __init__ (self):
    super(Elu, self).__init__('Elu')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x <  0.] = np.exp(y[x < 0.]) - 1.
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >= 0.]  = 1.
    y[x  < 0.] += 1.
    return y


class Relie (Activations):

  def __init__ (self):
    super(Relie, self).__init__('Relie')

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

  def __init__ (self):
    super(Ramp, self).__init__('Ramp')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x <= 0.]  = 0
    return y + .1 * x

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return (y > 0.) + .1


class Linear (Activations):

  def __init__ (self):
    super(Linear, self).__init__('Linear')

  @staticmethod
  def activate(x, copy=False):
    return x

  @staticmethod
  def gradient(x, copy=False):
    return np.ones(shape=x.shape, dtype=float)


class Tanh (Activations):

  def __init__ (self):
    super(Tanh, self).__init__('Tanh')

  @staticmethod
  def activate(x, copy=False):
    return np.tanh(x)

  @staticmethod
  def gradient(x, copy=False):
    return 1. - x * x


class Plse (Activations):

  def __init__ (self):
    super(Plse, self).__init__('Plse')

  @staticmethod
  def activate (x, copy=False):
    y = x.copy()
    y[x < -4.] = (y[x < -4.] + 4.) * 1e-2
    y[x >  4.] = (y[x >  4.] - 4.) * 1e-2 + 1.
    # this function  select elements bewteen -4 and 4
    # it solves problems with double conditions in array.
    func = np.vectorize(lambda t: (t >= -4.) and t<= 4.)
    y[func(x)] = y[func(x)] * .125 + .5
    return y

  @staticmethod
  def gradient (x, copy=False):
    y = x.copy()
    func  = np.vectorize(lambda t: (t<0.)  or (t>1.))
    func2 = np.vectorize(lambda t: (t>=0.) or (t<=1.))
    y[func2(x)] = .125
    y[func(x) ] = 1e-2
    return y


class Leaky (Activations):

  LEAKY_COEF  = 1e-1

  def __init__ (self):
    super(Leaky, self).__init__('Leaky')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x <= 0.] *= Leaky.LEAKY_COEF
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x >  0.] = 1.
    y[x <= 0.] = Leaky.LEAKY_COEF
    return y


class Stair (Activations):

  def __init__ (self):
    super(Stair, self).__init__('Stair')

  @staticmethod
  def activate(x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    n = np.floor(y)
    z = np.floor(y/2.)
    even = n % 2
    y[even == 0.] = z[even == 0]
    y[even != 0.] = ((x - n) + z)[even != 0]
    return y

  @staticmethod
  def gradient(x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    n = np.floor(y)
    y[n == y] = 0.
    y[n != y] = 1.
    return y


class Hardtan (Activations):

  def __init__ (self):
    super(Hardtan, self).__init__('HardTan')

  @staticmethod
  def activate(x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x < -1.] = -1.
    y[x >  1.] =  1.
    return y

  @staticmethod
  def gradient(x, copy=False):
    if copy: y = x.copy()
    else:    y = x
    y[:] = np.zeros(shape=y.shape)
    # this function select corrects indexes
    # solves problems with multiple conditions
    func = np.vectorize(lambda t: (t >- 1) and (t < 1.))
    y[func(x)] = 1.
    return y


class Lhtan (Activations):

  def __init__ (self):
    super(Lhtan, self).__init__('LhTan')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    y[x < 0.] *= 1e-3
    y[x > 1.]  = (y[x > 1.] - 1.) * 1e-3 + 1
    return y

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x.copy()
    # those functions select the correct elements
    # problems with double conditions
    func  = np.vectorize(lambda t: (t > 0.) and (t < 1.))
    func2 = np.vectorize(lambda t: (t <= 0.) or (t >= 1.))
    y[func2(x)] = 1e-3
    y[func(x)]  = 1
    return y


class Selu (Activations):

  def __init__ (self):
    super(Selu, self).__init__('Selu')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return (y >= 0.) * 1.0507 * y + (y < 0.) * 1.0507 * 1.6732 * (np.exp(x) - 1.)

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return (y >= 0.) * 1.0507 + (y < 0.) * (y + 1.0507 * 1.6732)


class Elliot (Activations):

  STEEPNESS = 1.

  def __init__ (self):
    super(Elliot, self).__init__('Elliot')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return .5 * Elliot.STEEPNESS * y / (1. + np.abs(y + Elliot.STEEPNESS)) + .5

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    last_fwd = 1. + np.abs(y * Elliot.STEEPNESS)
    return .5 * Elliot.STEEPNESS / (last_fwd * last_fwd)


class SymmElliot (Activations):

  STEEPNESS = 1.

  def __init__ (self):
    super(SymmElliot, self).__init__('SymmElliot')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return y * SymmElliot.STEEPNESS / (1. + np.abs(y * SymmElliot.STEEPNESS))

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    last_fwd = 1. + np.abs(y * SymmElliot.STEEPNESS)
    return SymmElliot.STEEPNESS / (last_fwd * last_fwd)


class SoftPlus (Activations):

  def __init__ (self):
    super(SoftPlus, self).__init__('SoftPlus')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return np.log(1. + np.exp(y))

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    ey = np.exp(y)
    return ey / (1. + ey)


class SoftSign (Activations):

  def __init__ (self):
    super(SoftSign, self).__init__('SoftSign')

  @staticmethod
  def activate (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    return y / (np.abs(y) + 1.)

  @staticmethod
  def gradient (x, copy=False):
    if copy: y = x.copy()
    else:    y = x

    fy = np.abs(y) + 1.
    return 1. / (fy * fy)
