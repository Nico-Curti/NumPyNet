#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import timeit
import argparse
import numpy as np
import pandas as pd
from time import time as now


__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Timing layer functions'

NUM_REPEATS = 10
NUMBER = 100
AVAILABLE_LAYERS = ['activation', 'avgpool', 'batchnorm', 'connected', 'convolutional', 'cost', 'dropout', 'input', 'logistic', 'maxpool', 'route', 'shortcut', 'shuffler', 'softmax', 'upsample', 'yolo']


def forward (layer, input_shape, params):

  SETUP_CODE = '''
import numpy as np
from NumPyNet.layers.{lower_layer} import {layer}
from NumPyNet.activations import Elu, Hardtan, Leaky, Lhtan, Linear, Loggy, Logistic, Plse, Ramp, Relie, Relu, Selu, Stair, Tanh

parameters = dict()
for k, v in {params}.items():
  try:
    v = eval(v)
    parameters[k] = v
  except:
    parameters[k] = v


input = np.random.uniform(low=-1., high=1., size={input_shape})
layer = {layer}(**parameters)
  '''.format(**{'input_shape' : input_shape,
                'layer' : layer,
                'params': params,
                'lower_layer' : layer.lower()
                })

  TEST_CODE = '''
layer.forward(input)
  '''

  times = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=NUM_REPEATS,
                        number=NUMBER)
  return times

def backward (layer, input_shape, params):

  SETUP_CODE = '''
import numpy as np
from NumPyNet.layers.{lower_layer} import {layer}
from NumPyNet.activations import Elu, Hardtan, Leaky, Lhtan, Linear, Loggy, Logistic, Plse, Ramp, Relie, Relu, Selu, Stair, Tanh

parameters = dict()
for k, v in {params}.items():
  try:
    v = eval(v)
    parameters[k] = v
  except:
    parameters[k] = v

input = np.random.uniform(low=-1., high=1., size={input_shape})
layer = {layer}(**parameters)
layer.forward(input)

delta = np.zeros(shape=input.shape, dtype=float)
  '''.format(**{'input_shape' : input_shape,
                'layer' : layer,
                'params': params,
                'lower_layer' : layer.lower()
                })

  TEST_CODE = '''
layer.backward(delta)
  '''

  times = timeit.repeat(setup=SETUP_CODE,
                        stmt=TEST_CODE,
                        repeat=NUM_REPEATS,
                        number=NUMBER)
  return times


def timing_activation_layer (input_shape):

  activations = ['Elu', 'HardTan', 'Leaky',
                 'LhTan', 'Linear', 'Loggy',
                 'Logistic', 'Plse', 'Ramp',
                 'Relie', 'Relu', 'Selu',
                 'Tanh',
                 #'Stair',
                 ]

  timing = []

  for activation in activations:
    params = {'activation' : activation}

    forward_times  = forward('Activation_layer', input_shape, params)
    backward_times = backward('Activation_layer', input_shape, params)

    timing.append( { 'layer' : 'Activation',
                     'input_shape' : input_shape,
                     **params,
                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing


def timing_avgpool_layer (input_shape):

  sizes = [(1, 1), (3, 3), (30, 30)]
  strides = [(1, 1), (2, 2), (20, 20)]
  pads = [False, True]

  timing = []

  for size in sizes:
    for stride in strides:
      for pad in pads:

        params = {'size' : size, 'stride' : stride, 'padding' : pad}

        forward_times  = forward('Avgpool_layer', input_shape, params)
        backward_times = backward('Avgpool_layer', input_shape, params)

        timing.append( { 'layer' : 'Avgpool',
                         'input_shape' : input_shape,
                         **params,
                         'num_repeatition' : NUM_REPEATS,
                         'number'        : NUMBER,
                         'forward_mean'  : np.mean(forward_times),
                         'forward_max'   : np.max(forward_times),
                         'forward_min'   : np.min(forward_times),
                         'forward_std'   : np.std(forward_times),
                         'backward_mean' : np.mean(backward_times),
                         'backward_max'  : np.max(backward_times),
                         'backward_min'  : np.min(backward_times),
                         'backward_std'  : np.std(backward_times),
                         })

  return timing


def timing_batchnorm_layer (input_shape):

  scales = np.random.uniform(low=0., high=1., size=input_shape[1:])
  bias   = np.random.uniform(low=0., high=1., size=input_shape[1:])

  timing = []

  params = {'scales' : str(scales.tolist()), 'bias' : str(bias.tolist())}

  forward_times  = forward('BatchNorm_layer', input_shape, params)
  backward_times = backward('BatchNorm_layer', input_shape, params)

  timing.append( { 'layer' : 'BatchNorm',
                   'input_shape' : input_shape,
                   #**params, # useless parameters
                   'num_repeatition' : NUM_REPEATS,
                   'number'        : NUMBER,
                   'forward_mean'  : np.mean(forward_times),
                   'forward_max'   : np.max(forward_times),
                   'forward_min'   : np.min(forward_times),
                   'forward_std'   : np.std(forward_times),
                   'backward_mean' : np.mean(backward_times),
                   'backward_max'  : np.max(backward_times),
                   'backward_min'  : np.min(backward_times),
                   'backward_std'  : np.std(backward_times),
                   })

  return timing


def timing_connected_layer (input_shape):

  outputs = [10, 50, 100]

  timing = []

  for output in outputs:

    weights = np.random.uniform(low=0., high=1., size=(np.prod(input_shape[1:]), output))
    bias    = np.random.uniform(low=0., high=1., size=(output,))

    params = {'input_shape' : input_shape, 'outputs' : output,
              'weights' : str(weights.tolist()), 'bias' : str(bias.tolist()),
              'activation' : 'Relu'}

    forward_times  = forward('Connected_layer', input_shape, params)
    backward_times = backward('Connected_layer', input_shape, params)

    timing.append( { 'layer' : 'Connected',
                     'input_shape' : input_shape,
                     #**params, # useless params
                     'outputs' : output,
                     'activation' : params['activation'],

                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing


def timing_convolutional_layer (input_shape):
  raise NotImplementedError

def timing_cost_layer (input_shape):

  from NumPyNet.layers import cost_layer

  costs = [cost_layer.mse, cost_layer.masked, cost_layer.mae, cost_layer.seg, cost_layer.smooth, cost_layer.wgan, cost_layer.hellinger, cost_layer.hinge, cost_layer.logcosh]

  timing = []

  for cost in costs:

    params = {'input_shape' : input_shape,
              'cost_type' : cost,
              'scale' : 1.5,
              'ratio' : 0.5,
              'noobject_scale' : 1.2,
              'threshold' : 0.2,
              'smoothing' : 0.5
              }

    forward_times  = forward('Cost_layer', input_shape, params)
    backward_times = backward('Cost_layer', input_shape, params)

    timing.append( { 'layer' : 'Cost',
                     **params,
                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing

def timing_dropout_layer (input_shape):

  probabilities = [0., .25, .5, .75, .1]

  timing = []

  for prob in probabilities:

    params = {'prob' : prob}

    forward_times  = forward('Dropout_layer', input_shape, params)
    backward_times = backward('Dropout_layer', input_shape, params)

    timing.append( { 'layer' : 'Dropout',
                     **params,
                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing

def timing_input_layer (input_shape):

  timing = []

  params = {'input_shape' : input_shape}

  forward_times  = forward('Input_layer', input_shape, params)
  backward_times = backward('Input_layer', input_shape, params)

  timing.append( { 'layer' : 'Input',
                   **params,
                   'num_repeatition' : NUM_REPEATS,
                   'number'        : NUMBER,
                   'forward_mean'  : np.mean(forward_times),
                   'forward_max'   : np.max(forward_times),
                   'forward_min'   : np.min(forward_times),
                   'forward_std'   : np.std(forward_times),
                   'backward_mean' : np.mean(backward_times),
                   'backward_max'  : np.max(backward_times),
                   'backward_min'  : np.min(backward_times),
                   'backward_std'  : np.std(backward_times),
                   })

  return timing

def timing_logistic_layer (input_shape):

  timing = []

  params = {}

  forward_times  = forward('Logistic_layer', input_shape, params)
  backward_times = backward('Logistic_layer', input_shape, params)

  timing.append( { 'layer' : 'Logistic',
                   **params,
                   'num_repeatition' : NUM_REPEATS,
                   'number'        : NUMBER,
                   'forward_mean'  : np.mean(forward_times),
                   'forward_max'   : np.max(forward_times),
                   'forward_min'   : np.min(forward_times),
                   'forward_std'   : np.std(forward_times),
                   'backward_mean' : np.mean(backward_times),
                   'backward_max'  : np.max(backward_times),
                   'backward_min'  : np.min(backward_times),
                   'backward_std'  : np.std(backward_times),
                   })

  return timing

def timing_maxpool_layer (input_shape):

  sizes = [(1, 1), (3, 3), (30, 30)]
  strides = [(1, 1), (2, 2), (20, 20)]
  pads = [False, True]

  timing = []

  for size in sizes:
    for stride in strides:
      for pad in pads:

        params = {'size' : size, 'stride' : stride, 'padding' : pad}

        forward_times  = forward('Maxpool_layer', input_shape, params)
        backward_times = backward('Maxpool_layer', input_shape, params)

        timing.append( { 'layer' : 'Maxpool',
                         'input_shape' : input_shape,
                         **params,
                         'num_repeatition' : NUM_REPEATS,
                         'number'        : NUMBER,
                         'forward_mean'  : np.mean(forward_times),
                         'forward_max'   : np.max(forward_times),
                         'forward_min'   : np.min(forward_times),
                         'forward_std'   : np.std(forward_times),
                         'backward_mean' : np.mean(backward_times),
                         'backward_max'  : np.max(backward_times),
                         'backward_min'  : np.min(backward_times),
                         'backward_std'  : np.std(backward_times),
                         })

  return timing

def timing_route_layer (input_shape):
  raise NotImplementedError

def timing_shortcut_layer (input_shape):
  raise NotImplementedError

def timing_shuffler_layer (input_shape):

  scales = (2, 3, 4)
  batch, w, h, c = input_shape

  for scale in scales:
    input_shape = (batch, w, h, c * scale*scale)

    params = {'scale' : scale}

    forward_times  = forward('Shuffler_layer', input_shape, params)
    backward_times = backward('Shuffler_layer', input_shape, params)

    timing.append( { 'layer' : 'Shuffler',
                     'input_shape' : input_shape,
                     #**params, # useless params
                     'activation' : params['activation'],

                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing


def timing_softmax_layer (input_shape):

  spatials = [False, True]
  temperature = 1.5
  #noloss = False
  #groups = 1

  for spatial in spatials:

    params = {'spatial' : spatial, 'temperature' : temperature}

    forward_times  = forward('Softmax_layer', input_shape, params)
    backward_times = backward('Softmax_layer', input_shape, params)

    timing.append( { 'layer' : 'Softmax',
                     **params,
                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing

def timing_upsample_layer (input_shape):

  strides = [1, 3, -2, -4]
  scale = 1.5

  timing = []

  for stride in strides:

    params = {'stride' : stride, 'scale' : scale}

    forward_times  = forward('Upsample_layer', input_shape, params)
    backward_times = backward('Upsample_layer', input_shape, params)

    timing.append( { 'layer' : 'Upsample',
                     **params,
                     'num_repeatition' : NUM_REPEATS,
                     'number'        : NUMBER,
                     'forward_mean'  : np.mean(forward_times),
                     'forward_max'   : np.max(forward_times),
                     'forward_min'   : np.min(forward_times),
                     'forward_std'   : np.std(forward_times),
                     'backward_mean' : np.mean(backward_times),
                     'backward_max'  : np.max(backward_times),
                     'backward_min'  : np.min(backward_times),
                     'backward_std'  : np.std(backward_times),
                     })

  return timing

def timing_yolo_layer (input_shape):
  raise NotImplementedError

def parse_args ():

  description = 'Timing layer functions'

  parser = argparse.ArgumentParser(description=description)

  parser.add_argument('--layer',
                      dest='layer',
                      required=True,
                      type=str,
                      action='store',
                      help='Layer type',
                      choices=AVAILABLE_LAYERS
                      )
  parser.add_argument('--output',
                      dest='out',
                      required=False,
                      type=str,
                      action='store',
                      help='Output filename',
                      default=''
                      )
  parser.add_argument('--n_rep',
                      dest='n_rep',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of repetition',
                      default=3
                      )
  parser.add_argument('--num',
                      dest='num',
                      required=False,
                      type=int,
                      action='store',
                      help='Number of iterations',
                      default=10
                      )

  args = parser.parse_args()

  NUM_REPEATS = args.n_rep
  NUMBER = args.num

  return args



if __name__ == '__main__':

  args = parse_args()

  input_shapes = [(4, 512, 512, 3)]

  timing_layers = {'activation' : timing_activation_layer,
                   'avgpool'    : timing_avgpool_layer,
                   'batchnorm'  : timing_batchnorm_layer,
                   'connected'  : timing_connected_layer,
                   'dropout'    : timing_dropout_layer,
                   'input'      : timing_input_layer,
                   'logistic'   : timing_logistic_layer,
                   'maxpool'    : timing_maxpool_layer,
                   'route'      : timing_route_layer,
                   'shortcut'   : timing_shortcut_layer,
                   'shuffler'   : timing_shuffler_layer,
                   'softmax'    : timing_softmax_layer,
                   'upsample'   : timing_upsample_layer,
                   'yolo'       : timing_yolo_layer
                   }

  timing = []
  for input_shape in input_shapes:

    tic = now()
    times = timing_layers[args.layer](input_shape)
    toc = now()

    print('Elapsed time: {:.3f} sec'.format(toc - tic))

    timing += [pd.DataFrame(times)]

  timing = pd.concat(timing)

  if args.out:
    timing.to_csv(args.out, sep=',', header=True, index=False)
  else:
    print(timing)
