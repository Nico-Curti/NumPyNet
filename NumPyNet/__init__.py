#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import all the objects in the package

from __future__ import division
from __future__ import print_function

from .activations import *
from .box import Box
from .detection import Detection
from .image import Image
from .image_utils import normalization
from .image_utils import image_utils

import parser
from NumPyNet import rnn_utils

__all__ = ['NumPyNet']

__package__ = 'NumPyNet'
__author__  = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
