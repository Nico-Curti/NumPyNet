#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Optimizer (object):
  '''
  '''

  def __init__ (self, lr=1e-3, decay=0., lr_min=0., lr_max=np.inf, *args, **kwargs):

    self.lr = lr
    self.decay = decay
    self.lr_min = lr_min
    self.lr_max = lr_max

    self.iterations = 0

  def update (self, params, gradients):
    '''
    '''
    self.lr *= 1. / (self.decay * self.iterations + 1.)
    self.lr  = np.clip(self.lr, self.lr_min, self.lr_max)

  def __str__ (self):
    return self.__class__.__name__


class SGD (Optimizer):
  '''
  '''

  def __init__ (self, *args, **kwargs):

    super(SGD, self).__init__(*args, **kwargs)

  def update (self, params, gradients):

    for p, g in zip(params, gradients):
      p -= self.lr * g#np.clip(g, -1., 1.)

    super(SGD, self).update(params, gradients)

    return params


class Momentum (Optimizer):
  '''
  '''

  def __init__ (self, momentum=.9, *args, **kwargs):

    super(Momentum, self).__init__(*args, **kwargs)
    self.momentum = momentum

    self.velocity = None

  def update (self, params, gradients):

    if self.velocity is None:
      self.velocity = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (v, p, g) in enumerate(zip(self.velocity, params, gradients)):
      v  = self.momentum * v - self.lr * g # np.clip(g, -1., 1.)
      p += v
      self.velocity[i] = v

    super(Momentum, self).update(params, gradients)

    return params


class NesterovMomentum (Optimizer):
  '''
  '''

  def __init__ (self, momentum=.9, *args, **kwargs):

    super(NesterovMomentum, self).__init__(*args, **kwargs)
    self.momentum = momentum

    self.velocity = None

  def update (self, params, gradients):

    if self.velocity is None:
      self.velocity = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (v, p, g) in enumerate(zip(self.velocity, params, gradients)):
      v  = self.momentum * v - self.lr * g # np.clip(g, -1., 1.)
      p += self.momentum * v - self.lr * g # np.clip(g, -1., 1.)
      self.velocity[i] = v

    super(NesterovMomentum, self).update(params, gradients)

    return params


class Adagrad (Optimizer):
  '''
  '''

  def __init__ (self, epsilon=1e-6, *args, **kwargs):

    super(Adagrad, self).__init__(*args, **kwargs)
    self.epsilon = epsilon

    self.cache = None

  def update (self, params, gradients):

    if self.cache is None:
      self.cache = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (c, p, g) in enumerate(zip(self.cache, params, gradients)):

      c += g * g
      p -= self.lr * g / (np.sqrt(c + self.epsilon))
      self.cache[i] = c

    super(Adagrad, self).update(params, gradients)

    return params


class RMSprop (Optimizer):
  '''
  '''

  def __init__ (self, rho=.9, epsilon=1e-6, *args, **kwargs):

    super(RMSprop, self).__init__(*args, **kwargs)

    self.rho = rho
    self.epsilon = epsilon

    self.cache = None
    self.iterations = 0

  def update (self, params, gradients):

    if self.cache is None:
      self.cache = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (c, p, g) in enumerate(zip(self.cache, params, gradients)):

      c = self.rho * c + (1 - self.rho) * g * g
      p -= (self.lr * g / np.sqrt(c + self.epsilon))
      self.cache[i] = c

    super(RMSprop, self).update(params, gradients)

    return params


class Adadelta (Optimizer):
  '''
  '''

  def __init__ (self, rho=0.9, epsilon=1e-6, *args, **kwargs):

    super(Adadelta, self).__init__(*args, **kwargs)

    self.rho = rho
    self.epsilon = epsilon

    self.cache = None
    self.delta = None

  def update (self, params, gradients):

    if self.cache is None:
      self.cache = [np.zeros(shape=p.shape, dtype=float) for p in params]

    if self.delta is None:
      self.delta = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (c, d, p, g) in enumerate(zip(self.cache, self.delta, params, gradients)):

      c = self.rho * c + (1 - self.rho) * g * g
      update = g * np.sqrt(d + self.epsilon) / np.sqrt(c + self.epsilon)
      p -= self.lr * update
      d = self.rho * d + (1 - self.rho) * update * update

      self.cache[i] = c
      self.delta[i] = d

    super(Adadelta, self).update(params, gradients)

    return params


class Adam (Optimizer):
  '''
  '''

  def __init__ (self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):

    super(Adam, self).__init__(*args, **kwargs)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.ms = None
    self.vs = None

  def update (self, params, gradients):
    self.iterations += 1

    a_t = self.lr * np.sqrt(1 - np.power(self.beta2, self.iterations)) / \
          (1 - np.power(self.beta1, self.iterations))

    if self.ms is None:
      self.ms = [np.zeros(shape=p.shape, dtype=float) for p in params]

    if self.vs is None:
      self.vs = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, gradients)):

      m = self.beta1 * m + (1 - self.beta1) * g
      v = self.beta2 * v + (1 - self.beta2) * g * g
      p -= a_t * m / (np.sqrt(v + self.epsilon))

      self.ms[i] = m
      self.vs[i] = v

    super(Adam, self).update(params, gradients)

    return params


class Adamax (Optimizer):
  '''
  '''

  def __init__ (self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):

    super(Adamax, self).__init__(*args, **kwargs)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.ms = None
    self.vs = None

  def update (self, params, gradients):
    self.iterations += 1

    a_t = self.lr / (1 - np.power(self.beta1, self.iterations))

    if self.ms is None:
      self.ms = [np.zeros(shape=p.shape, dtype=float) for p in params]

    if self.vs is None:
      self.vs = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, gradients)):
      m = self.beta1 * m + (1 - self.beta1) * g
      v = np.maximum(self.beta2 * v, np.abs(g))
      p -= a_t * m / (v + self.epsilon)

      self.ms[i] = m
      self.vs[i] = v

    super(Adamax, self).update(params, gradients)

    return params
