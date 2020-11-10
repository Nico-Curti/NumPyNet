#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import re
import ast
import configparser
from collections import OrderedDict

from NumPyNet.exception import CfgVariableError
from NumPyNet.exception import DataVariableError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class data_config (object):

  '''
  Data configuration parser

  Parameters
  ----------
    filename : str
      Configuration data filename or path

  Returns
  -------
    data_config object

  Notes
  -----
  The data configuration stores the global parameters for a given model (ex. cfg filename, weight filename, ...)
  The file must be saved in a dictionary format like "cfg = config_filename.cfg"
  '''

  _data = dict()

  def __init__ (self, filename):

    if not os.path.isfile(filename):
      raise IOError('Could not open or find the data file. Given: {}'.format(filename))

    with open (filename, 'r') as fp:
      rows = fp.read().splitlines()

    rows = [x.strip(' ') for x in rows]           # remove redundant blank spaces
    rows = [re.split(r'\s+=\s+', x) for x in rows] # split the string like 'key = value'

    for k, v in rows:
      try:
        self._data[k] = ast.literal_eval(v)
      except ValueError:
        self._data[k] = v


  def get (self, key, default=None):
    '''
    Getter function

    Parameters
    ----------
      key : str
        config dictionary key

      default : dtype (default=None)
        the default value if the key is not found in the data config
    '''

    try:
      return self._data[key] if key in self._data else default

    except:
      raise CfgVariableError('Type variable not recognized! Possible variables are only [bool, int, float, string, vector<int>, vector<float>, vector<string>].')

  def __str__ (self):
    return str(self._data)

  def __len__ (self):
    return len(self._data)



class net_config (object):

  '''
  Network config parser

  Parameters
  ----------
    filename : str
      Network config filename or path

  Returns
  -------
    net_config object

  Notes
  -----
  .. note::
    The network configuration file must be stored in INI format.
    Since multiple layers can have the same type the dictionary must be overloaded by a
    custom OrderedDict
  '''

  class multidict (OrderedDict):

    _unique = 0 # class variable

    def __setitem__ (self, key, val):

      if isinstance(val, dict):
        key += str(self._unique)
        self._unique += 1

      OrderedDict.__setitem__(self, key, val)

  def __init__ (self, filename):

    if not os.path.isfile(filename):
      raise IOError('Could not open or find the config file. Given: {}'.format(filename))

    self._data = configparser.ConfigParser(defaults=None, dict_type=self.multidict, strict=False)
    self._data.read(filename)
    self._index = 0

    first_section = self._data.sections()[0]

    if not first_section.startswith('net') and not first_section.startswith('network'):
      raise CfgVariableError('Config error! First section must be a network one (ex. [net] / [network]). Given: [{}]'.format(first_section))

  def __iter__ (self):
    self._index = 0
    return self

  def __next__ (self):
    self._index += 1

    if self._index < len(self):
      return self._data.sections()[self._index]
    else:
      raise StopIteration

  def next (self): # this should fix python2* problems with __iter__ and __next__
    return self.__next__()

  def get_params (self, section):
    '''
    Return all params in section as dict

    Parameters
    ----------
      section : str
        Layer name + position

    Returns
    -------
      params : dict
        Dictionary of all (keys, values) in section
    '''

    options = self._data.options(section)
    params = {}

    for opt in options:
      try:
        params[opt] = ast.literal_eval(self._data.get(section, opt))
      except ValueError:
        params[opt] = self._data.get(section, opt)

    return params

  def get (self, section, key, default=None):
    '''
    Getter function

    Parameters
    ----------
      section : str
        Layer name + position

      key : str
        config dictionary key

      default : dtype (default=None)
        the default value if the key is not found in the data config
    '''

    if section not in self._data:
      raise DataVariableError('Section not found in the config file. Given {}'.format(section))

    try:
      return ast.literal_eval(self._data.get(section, key)) if self._data.has_option(section, key) else default

    except ValueError: # it is a pure string
      return self._data.get(section, key) if self._data.has_option(section, key) else default

    except:
      raise DataVariableError('Type variable not recognized! Possible variables are only [int, float, string, vector<float>].')

  def __str__ (self):

    cfg = ''
    for k in self._data.keys():
      if k == 'DEFAULT':
        continue
      section = re.split(r'\d+', k)[0]
      cfg = '\n'.join((cfg, '[{}]'.format(section)))
      values = ('{} = {}'.format(key, val) for key, val in self._data.items(k))
      values = '\n'.join(values)
      cfg = '\n'.join((cfg, values, '\n'))

    return cfg

  def __len__ (self):
    return len(self._data) - 1 # the first section is the default one


# Global parser functions

def read_map (filename):
  '''
  Read the map file

  Parameters
  ----------
    filename : str
      Map filename or path

  Returns
  -------
    rows : list
      List of the maps read

  Notes
  -----
  .. note::
    This functioni is used by the Yolo layer
  '''

  with open(filename, 'r') as fp:
    rows = fp.read().splitlines()

  rows = list(map(int, rows))
  return rows

def get_labels (filename, classes=-1):
  '''
  Read the labels file

  Parameters
  ----------
    filename : str
      Labels filename or path

    classes : int (default = -1)
      Number of labels to read. If it is equal to -1 the full list of labels is read

  Returns
  -------
    labels : list
      The first 'classes' labels in the file.
  '''

  with open(filename, 'r') as fp:
    labels = fp.read().splitlines()

  labels = labels[:classes]
  return labels
