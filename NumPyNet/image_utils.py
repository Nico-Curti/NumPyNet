#!/usr/bin/env python
# -*- coding: utf-8 -*-

from enum import Enum

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class normalization (int, Enum):

  '''
  Utility class with enum for normalization
  algorithm.

  This class is used for a pure readability purpose.
  '''

  normalize = 0
  denormalize = 1


class image_utils (object):

  '''
  Utility class for the Image object.

  The class stores pre-determined set of
  values related to the colors and colormaps
  for the detection boxes.
  '''

  num_box_colors = 6
  num_map_colors = 19

  colors = [ (0., 0., 0.),
             (0., 1., 0.),
             (0., 1., 1.),
             (1., 0., 0.),
             (1., 0., 1.),
             (1., 1., 0.)
            ]

  rgb_cmap = [ (0.078431, 0.078431, 0.078431),
               (0.266667, 0.133333, 0.600000),
               (0.231373, 0.047059, 0.741176),
               (0.200000, 0.066667, 0.733333),
               (0.266667, 0.266667, 0.866667),
               (0.066667, 0.666667, 0.733333),
               (0.070588, 0.741176, 0.725490),
               (0.133333, 0.800000, 0.666667),
               (0.411765, 0.815686, 0.145098),
               (0.666667, 0.800000, 0.133333),
               (0.815686, 0.764706, 0.062745),
               (0.800000, 0.733333, 0.200000),
               (0.996078, 0.682353, 0.176471),
               (1.000000, 0.600000, 0.200000),
               (1.000000, 0.400000, 0.266667),
               (1.000000, 0.266667, 0.133333),
               (1.000000, 0.200000, 0.066667),
               (0.933333, 0.066667, 0.000000),
               (0.972549, 0.047059, 0.070588)
               ]


