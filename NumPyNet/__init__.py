#!/usr/bin/env python
# -*- coding: utf-8 -*-

# Import all the objects in the package

from __future__ import division
from __future__ import print_function

from .activations import *
from .box import Box
from .data import DataGenerator
from .detection import Detection
from .image import Image
from .image_utils import normalization
from .image_utils import image_utils
from .network import Network

from . import parser
from . import rnn_utils
from .utils import print_statistics
from .video import VideoCapture

from .__version__ import __version__

__author__  = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

# aliases

Model = Network

# define FMATH
ENABLE_FMATH = False
