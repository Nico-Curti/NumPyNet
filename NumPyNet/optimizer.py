#!/usr/bin/env python
# -*- coding: utf-8 -*-


from __future__ import division
from __future__ import print_function

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class Optimizer (object):

  '''
  Abstract base class for the optimizers

  Parameters
  ----------
    lr : float (default=2e-2)
      Learning rate value

    decay : float (default=0.)
      Learning rate decay

    lr_min : float (default=0.)
      Minimum of learning rate domain

    lr_max : float (default=np.inf)
      Maximum of learning rate domain

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, lr=1e-3, decay=0., lr_min=0., lr_max=np.inf, *args, **kwargs):

    self.lr = lr
    self.decay = decay
    self.lr_min = lr_min
    self.lr_max = lr_max

    self.iterations = 1

  def update (self, params, gradients):
    '''
    Update the optimizer parameters

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      self
    '''
    self.lr *= 1. / (self.decay * self.iterations + 1.)
    self.lr  = np.clip(self.lr, self.lr_min, self.lr_max)

    self.iterations += 1

  def __repr__ (self):
    '''
    Representation
    '''
    class_name = self.__class__.__qualname__
    try:
      params = super(type(self), self).__init__.__code__.co_varnames
    except AttributeError:
      params = self.__init__.__code__.co_varnames

    params = set(params) - {'self', 'args', 'kwargs'}
    args = ', '.join(['{0}={1}'.format(k, str(getattr(self, k)))
                      if not isinstance(getattr(self, k), str) else '{0}="{1}"'.format(k, str(getattr(self, k)))
                      for k in params])
    return '{0}({1})'.format(class_name, args)

  def __str__ (self):
    '''
    Printer
    '''
    return self.__class__.__name__


class SGD (Optimizer):

  '''
  Stochastic Gradient Descent specialization

  Update the parameters according to the rule

  .. code-block:: python

    parameter -= learning_rate * gradient

  Parameters
  ----------
    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, *args, **kwargs):

    super(SGD, self).__init__(*args, **kwargs)

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''
    for p, g in zip(params, gradients):
      p -= self.lr * g  # np.clip(g, -1., 1.)

    super(SGD, self).update(params, gradients)

    return params


class Momentum (Optimizer):

  '''
  Stochastic Gradient Descent with Momentum specialiation

  Update the parameters according to the rule

  .. code-block:: python

    v = momentum * v - lr * gradient
    parameter += v - learning_rate * gradient


  Parameters
  ----------
    momentum : float (default=0.9)
      Momentum value

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, momentum=.9, *args, **kwargs):

    super(Momentum, self).__init__(*args, **kwargs)
    self.momentum = momentum

    self.velocity = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''

    if self.velocity is None:
      self.velocity = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (v, p, g) in enumerate(zip(self.velocity, params, gradients)):
      v  = self.momentum * v - self.lr * g  # np.clip(g, -1., 1.)
      p += v
      self.velocity[i] = v

    super(Momentum, self).update(params, gradients)

    return params


class NesterovMomentum (Optimizer):

  '''
  Stochastic Gradient Descent with Nesterov Momentum specialiation

  Update the parameters according to the rule

  .. code-block:: python

    v = momentum * v - lr * gradient
    parameter += momentum * v - learning_rate * gradient

  Parameters
  ----------
    momentum : float (default=0.9)
      Momentum value

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, momentum=.9, *args, **kwargs):

    super(NesterovMomentum, self).__init__(*args, **kwargs)
    self.momentum = momentum

    self.velocity = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''

    if self.velocity is None:
      self.velocity = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (v, p, g) in enumerate(zip(self.velocity, params, gradients)):
      v  = self.momentum * v - self.lr * g  # np.clip(g, -1., 1.)
      p += self.momentum * v - self.lr * g  # np.clip(g, -1., 1.)
      self.velocity[i] = v

    super(NesterovMomentum, self).update(params, gradients)

    return params


class Adagrad (Optimizer):

  '''
  Adagrad optimizer specialization

  Update the parameters according to the rule

  .. code-block:: python

    c += gradient * gradient
    parameter -= learning_rate * gradient / (sqrt(c) + epsilon)

  Parameters
  ----------
    epsilon : float (default=1e-6)
      Precision parameter to overcome numerical overflows

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, epsilon=1e-6, *args, **kwargs):

    super(Adagrad, self).__init__(*args, **kwargs)
    self.epsilon = epsilon

    self.cache = None

  def update (self, params, gradients):

    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''

    if self.cache is None:
      self.cache = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (c, p, g) in enumerate(zip(self.cache, params, gradients)):

      c += g * g
      p -= self.lr * g / (np.sqrt(c) + self.epsilon)
      self.cache[i] = c

    super(Adagrad, self).update(params, gradients)

    return params


class RMSprop (Optimizer):

  '''
  RMSprop optimization algorithm

  Update the parameters according to the rule

  .. code-block:: python

    c = rho * c + (1. - rho) * gradient * gradient
    parameter -= learning_rate * gradient / (sqrt(c) + epsilon)

  Parameters
  ----------
    rho : float (default=0.9)
      Decay factor

    epsilon : float (default=1e-6)
      Precision parameter to overcome numerical overflows

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, rho=.9, epsilon=1e-6, *args, **kwargs):

    super(RMSprop, self).__init__(*args, **kwargs)

    self.rho = rho
    self.epsilon = epsilon

    self.cache = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''

    if self.cache is None:
      self.cache = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (c, p, g) in enumerate(zip(self.cache, params, gradients)):

      c = self.rho * c + (1 - self.rho) * g * g
      p -= (self.lr * g / (np.sqrt(c) + self.epsilon))
      self.cache[i] = c

    super(RMSprop, self).update(params, gradients)

    return params


class Adadelta (Optimizer):

  '''
  AdaDelta optimization algorithm

  Update the parameters according to the rule

  .. code-block:: python

    c = rho * c + (1. - rho) * gradient * gradient
    update = gradient * sqrt(d + epsilon) / (sqrt(c) + epsilon)
    parameter -= learning_rate * update
    d = rho * d + (1. - rho) * update * update

  Parameters
  ----------
    rho : float (default=0.9)
      Decay factor

    epsilon : float (default=1e-6)
      Precision parameter to overcome numerical overflows

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, rho=0.9, epsilon=1e-6, *args, **kwargs):

    super(Adadelta, self).__init__(*args, **kwargs)

    self.rho = rho
    self.epsilon = epsilon

    self.cache = None
    self.delta = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''

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
  Adam optimization algorithm

  Update the parameters according to the rule

  .. code-block:: python

    at  = learning_rate * sqrt(1 - B2**iterations) / (1 - B1**iterations)
    m = B1 * m + (1 - B1) * gradient
    v = B2 * m + (1 - B2) * gradient * gradient
    parameter -= at * m / (sqrt(v) + epsilon)

  Parameters
  ----------
    beta1 : float (default=0.9)
      B1 factor

    beta2 : float (default=0.999)
      B2 factor

    epsilon : float (default=1e-8)
      Precision parameter to overcome numerical overflows

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):

    super(Adam, self).__init__(*args, **kwargs)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.ms = None
    self.vs = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''
    a_t = self.lr * np.sqrt(1 - np.power(self.beta2, self.iterations)) / \
          (1 - np.power(self.beta1, self.iterations))

    if self.ms is None:
      self.ms = [np.zeros(shape=p.shape, dtype=float) for p in params]

    if self.vs is None:
      self.vs = [np.zeros(shape=p.shape, dtype=float) for p in params]

    for i, (m, v, p, g) in enumerate(zip(self.ms, self.vs, params, gradients)):

      m = self.beta1 * m + (1 - self.beta1) * g
      v = self.beta2 * v + (1 - self.beta2) * g * g
      p -= a_t * m / (np.sqrt(v) + self.epsilon)

      self.ms[i] = m
      self.vs[i] = v

    super(Adam, self).update(params, gradients)

    return params


class Adamax (Optimizer):

  '''
  Adamax optimization algorithm

  Update the parameters according to the rule

  .. code-block:: python

    at  = learning_rate / (1 - B1**iterations)
    m = B1 * m + (1 - B1) * gradient
    v = max(B2 * v, abs(gradient))
    parameter -= at * m / (v + epsilon)

  Parameters
  ----------
    beta1 : float (default=0.9)
      B1 factor

    beta2 : float (default=0.999)
      B2 factor

    epsilon : float (default=1e-8)
      Precision parameter to overcome numerical overflows

    *args : list
      Class specialization variables.

    **kwargs : dict
      Class Specialization variables.
  '''

  def __init__ (self, beta1=0.9, beta2=0.999, epsilon=1e-8, *args, **kwargs):

    super(Adamax, self).__init__(*args, **kwargs)

    self.beta1 = beta1
    self.beta2 = beta2
    self.epsilon = epsilon

    self.ms = None
    self.vs = None

  def update (self, params, gradients):
    '''
    Update the given parameters according to the class optimization algorithm

    Parameters
    ----------
      params : list
        List of parameters to update

      gradients : list
        List of corresponding gradients

    Returns
    -------
      params : list
        The updated parameters
    '''
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
