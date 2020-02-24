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
from .input_layer import Input_layer
from .l1norm_layer import L1Norm_layer
from .l2norm_layer import L2Norm_layer
from .logistic_layer import Logistic_layer
from .maxpool_layer import Maxpool_layer
from .rnn_layer import RNN_layer
from .route_layer import Route_layer
from .shortcut_layer import Shortcut_layer
from .shuffler_layer import Shuffler_layer
from .softmax_layer import Softmax_layer
from .upsample_layer import Upsample_layer
from .yolo_layer import Yolo_layer

# Alias (keras)

AvgPool2D = Avgpool_layer
Batchnorm = BatchNorm_layer
Dense = Connected_layer
Conv2D = Convolutional_layer
Dropout = Dropout_layer
L1Normalization = L1Norm_layer
L2Normalization = L2Norm_layer
MaxPool2D = Maxpool_layer
concatenate = Route_layer
Add = Shortcut_layer
SoftMax = Softmax_layer
UpSampling2D = Upsample_layer

__author__  = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

