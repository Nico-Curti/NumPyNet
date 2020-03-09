#!/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import numpy as np

from NumPyNet.exception import CfgVariableError
from NumPyNet.exception import DataVariableError
from NumPyNet.parser import net_config
from NumPyNet.parser import data_config

import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class TestDataConfig:
  '''
  '''

  def test_constructor (self):

    here = os.path.dirname(__file__)
    filename = os.path.join(here, '..', 'data', 'yolov3.data')

    cfg = data_config(filename)
    assert len(cfg) == 7

    with pytest.raises(FileNotFoundError):
      filename = ''
      cfg = data_config(filename)

  def test_getter (self):

    here = os.path.dirname(__file__)
    filename = os.path.join(here, '..', 'data', 'yolov3.data')

    cfg = data_config(filename)

    assert cfg.get('cfg', '') == 'cfg/yolov3.cfg'
    assert cfg.get('None', 42) == 42

    with pytest.raises(CfgVariableError):
      res = cfg.get(['weights'], 32)
      assert res == 32

    assert cfg.get('weights', '') == 'data/yolov3.weights'
    assert cfg.get('names', '') == 'data/coco.names'
    assert cfg.get('thresh', 2.12) == .5
    assert cfg.get('hier', 3.14) == .5
    assert cfg.get('classes', 42) == 80

  def test_print (self):

    here = os.path.dirname(__file__)
    filename = os.path.join(here, '..', 'data', 'yolov3.data')

    cfg = data_config(filename)

    assert str(cfg) == str(cfg._data)
    evaluated = eval(str(cfg))

    assert isinstance(evaluated, dict)

    assert evaluated.get('cfg', '') == 'cfg/yolov3.cfg'
    assert evaluated.get('None', 42) == 42
    assert evaluated.get('weights', 32) != 32
    assert evaluated.get('weights', '') == 'data/yolov3.weights'
    assert evaluated.get('names', '') == 'data/coco.names'
    assert evaluated.get('thresh', 2.12) == .5
    assert evaluated.get('hier', 3.14) == .5
    assert evaluated.get('classes', 42) == 80

class TestNetConfig:
  '''
  '''

  def test_constructor (self):

    here = os.path.dirname(__file__)
    filename = os.path.join(here, '..', 'cfg', 'yolov3.cfg')

    cfg = net_config(filename)
    assert len(cfg) == 108

    print(cfg)

    with pytest.raises(FileNotFoundError):
      filename = ''
      cfg = net_config(filename)

  def test_getter (self):

    here = os.path.dirname(__file__)
    filename = os.path.join(here, '..', 'cfg', 'yolov3.cfg')

    cfg = net_config(filename)

    with pytest.raises(DataVariableError):
      res = cfg.get('net', 'batch', 42)
      assert res == 42

    assert cfg.get('net0', 'batch', 42) == 1
    assert cfg.get('convolutional1', 'stride', 3) == 1
