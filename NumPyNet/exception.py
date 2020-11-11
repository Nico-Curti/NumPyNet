#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class CfgfmtError (Exception):

  '''
  Config file format exception

  This exception is raised if something goes wrong in the
  format of the neural network configuration file.
  The error raised is set to 1.
  '''

  def __init__ (self, message, errors=1):

    super(CfgfmtError, self).__init__(message)

    self.errors = errors

class CfgVariableError (Exception):

  '''
  Config file variable exception

  This exception is raised if something goes wrong in the
  variables read in the neural network configuration file.
  The error raised is set to 2.
  '''

  def __init__ (self, message, errors=2):

    super(CfgVariableError, self).__init__(message)

    self.errors = errors


class DataVariableError (Exception):

  '''
  Data file variable exception

  This exception is raised if something goes wrong in the
  variables read in the neural network data file.
  The error raised is set to 3.
  '''

  def __init__ (self, message, errors=3):

    super(DataVariableError, self).__init__(message)

    self.errors = errors

class LayerError (Exception):

  '''
  Layer exception

  This exception is raised if something goes wrong in the
  construction or management of the Layer objects
  The error raised is set to 4.
  '''

  def __init__ (self, message, errors=4):

    super(LayerError, self).__init__(message)

    self.errors = errors

class MetricsError (Exception):

  '''
  Metrics exception

  This exception is raised if something goes wrong in the
  execution of the evaluation metrics for the neural network object.
  The error raised is set to 5.
  '''

  def __init__ (self, message, errors=5):

    super(MetricsError, self).__init__(message)

    self.errors = errors

class NetworkError (Exception):

  '''
  Network exception

  This exception is raised if something goes wrong
  during the building/training of the neural network object.
  The error raised is set to 6.
  '''

  def __init__ (self, message, errors=6):

    super(NetworkError, self).__init__(message)

    self.errors = errors

class VideoError (Exception):

  '''
  Video exception

  This exception is raised if something goes wrong during
  the video capture performed by the VideoCapture object.
  The error raised is set to 7.
  '''

  def __init__ (self, message, errors=7):

    super(VideoError, self).__init__(message)

    self.errors = errors

class NotFittedError (Exception):

  '''
  Not fitted exception

  This exception is raised if you can try to perform the
  model prediction before the training phase.
  The error raised is set to 8.
  '''

  def __init__ (self, message, errors=8):

    super(NotFittedError, self).__init__(message)

    self.errors = errors
