#!/usr/bin/env python
# -*- coding: utf-8 -*-

# import layer objects

from .activation_layer import Activation_layer
from .avgpool_layer import Avgpool_layer
from .batchnorm_layer import BatchNorm_layer
from .connected_layer import Connected_layer
from .convolutional_layer import Convolutional_layer
from .cost_layer import Cost_layer, cost_type
from .dropout_layer import Dropout_layer
from .logistic_layer import Logistic_layer
from .input_layer import Input_layer
from .maxpool_layer import Maxpool_layer
from .route_layer import Route_layer
from .shortcut_layer import Shortcut_layer
from .softmax_layer import Softmax_layer

# Alias (keras)

Dense_layer = Connected_layer

__package__ = 'NumPyNet layers'
__author__  = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

