#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class CfgfmtError (Exception):

  def __init__ (self, message, errors=1):

    super(CfgfmtError, self).__init__(message)

    self.errors = errors

class CfgVariableError (Exception):

  def __init__ (self, message, errors=1):

    super(CfgVariableError, self).__init__(message)

    self.errors = errors


class DataVariableError (Exception):

  def __init__ (self, message, errors=1):

    super(DataVariableError, self).__init__(message)

    self.errors = errors

class LayerError (Exception):

  def __init__ (self, message, errors=1):

    super(LayerError, self).__init__(message)

    self.errors = errors

class NetworkError (Exception):

  def __init__ (self, message, errors=1):

    super(NetworkError, self).__init__(message)

    self.errors = errors

class VideoError (Exception):

  def __init__ (self, message, errors=1):

    super(VideoError, self).__init__(message)

    self.errors = errors

class NotFittedError (Exception):

  def __init__ (self, message, errors=1):

    super(NotFittedError, self).__init__(message)

    self.errors = errors
