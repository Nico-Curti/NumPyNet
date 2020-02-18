#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import re
import configparser
from collections import OrderedDict
from ast import literal_eval as eval

from NumPyNet.exception import CfgVariableError
from NumPyNet.exception import DataVariableError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class data_config (object):

  _data = dict()

  def __init__ (self, filename):

    if not os.path.isfile(filename):
      raise FileNotFoundError('Could not open or find the data file. Given: {}'.format(filename))

    with open (filename, 'r', encoding='utf-8') as fp:
      rows = fp.read().splitlines()

    rows = [x.strip(' ') for x in rows]           # remove redundant blank spaces
    rows = [re.split(r'\s+=\s+', x) for x in rows] # split the string like 'key = value'

    for k, v in rows:
      try:
        self._data[k] = eval(v)
      except NameError:
        self._data[k] = v


  def get (self, key, default=None):

    try:
      return self._data if key in self._data else default

    except:
      raise CfgVariableError('Type variable not recognized! Possible variables are only [bool, int, float, string, vector<int>, vector<float>, vector<string>].')




class net_config (object):

  class multidict (OrderedDict):

    _unique = 0 # class variable

    def __setitem__ (self, key, val):

      if isinstance(val, dict):
        self._unique += 1
        key += str(self._unique)

      OrderedDict.__setitem__(self, key, val)


  def __init__ (self, filename):

    if not os.path.isfile(filename):
      raise FileNotFoundError('Could not open or find the config file. Given: {}'.format(filename))

    self._data = configparser.ConfigParser(defaults=None, dict_type=self.multidict, strict=False)
    self._data.read(filename)

    first_section = self._data.sections()[0]

    if not first_section.startswith('net') and not first_section.startswith('network'):
      raise CfgVariableError('Config error! First section must be a network one (ex. [net] / [network]). Given: [{}]'.format(first_section))


  def __len__ (self):
    return len(self._data.sections()) - 1 # net is the first

  def __iter__ (self):
    self.layer_index = 0
    return self

  def __next__ (self):
    if self.layer_index < len(self._data.sections()) - 1:
      self.layer_index += 1
      return self._data.sections()[self.layer_index]

    else:
      raise StopIteration

  def get_params (self, section):
    try:

      return self._data.items(section)

    except:
      raise CfgVariableError('Config error! Section "{}" does not exist'.format(section))

  def get (self, section, key, default=None):

    try:
      return eval(self._data.get(section, key)) if self._data.has_option(section, key) else default

    except NameError: # it is a pure string
      return self._data.get(section, key) if self._data.has_option(section, key) else default

    except:
      raise DataVariableError('Type variable not recognized! Possible variables are only [int, float, string, vector<float>].')


  def __str__ (self):

    cfg = ''
    for k in self._data.keys():
      section = re.split(r'\d+', k)[0]
      cfg += '[' + section + ']'
      cfg += '\n'

      for key, val in self._data.items(k):
        cfg += ' = '.join([key, val])
        cfg += '\n'

      cfg += '\n'

    return cfg



# Global parser functions

def read_map (filename):

  if not os.path.isfile(filename):
    raise FileNotFoundError('Could not open or find the map file. Given: {}'.format(filename))

  with open(filename, 'r', encoding='utf-8') as fp:
    rows = fp.read().splitlines()

  rows = list(map(int, rows))
  return rows

def get_labels (filename, classes=-1):

  if not os.path.isfile(filename):
    raise FileNotFoundError('Could not open or find the label file. Given: {}'.format(filename))

  with open(filename, 'r', encoding='utf-8') as fp:
    labels = fp.read().splitlines()

  labels = labels[:classes]
  return labels
