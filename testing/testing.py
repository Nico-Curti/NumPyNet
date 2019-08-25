# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

from keras.models import Model
from keras.layers import Input, Activation
import keras.backend as K
import tensorflow as tf

import numpy as np

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'NumPyNet Layers testing'

np.random.seed(123)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_cost_layer():
  '''
  Tests:
        the fwd function of the cost layer.
        if the cost is the same for every cost_type (mse and mae)
        if the delta is correctly computed
  To be tested:
        _smoothing
        _threshold
        _ratio
        noobject_scale
        masked
        _seg
        _wgan
  '''
  from NumPyNet.layers import cost_layer as cl
  from NumPyNet.layers.cost_layer import Cost_layer

  from keras.losses import mean_squared_error
  from keras.losses import mean_absolute_error

  from math import isclose

  np.random.seed(123)

  losses = [mean_absolute_error, mean_squared_error]

  for loss_function in losses :

    keras_loss_type = mean_absolute_error

    outputs = 100
    truth = np.random.uniform(low=0., high=1., size=(outputs,))
    inpt = np.random.uniform(low=0., high=1., size=(outputs,))

    inp = Input(shape=(inpt.size, ))
    x = Activation(activation='linear')(inp)
    model = Model(inputs=[inp], outputs=x)

    # an input layer to feed labels
    truth_tf = K.variable(truth)

    if   keras_loss_type is mean_squared_error:  cost = cl.cost_type.mse
    elif keras_loss_type is mean_absolute_error: cost = cl.cost_type.mae

    numpynet_layer = Cost_layer(inpt.size, cost,
                             scale=1., ratio=0., noobject_scale=1.,
                             threshold=0., smoothing=0.)

    keras_loss = K.eval(keras_loss_type(truth_tf, inpt))
    numpynet_layer.forward(inpt, truth)
    numpynet_loss = numpynet_layer.cost

    assert isclose(keras_loss, numpynet_loss, abs_tol=1e-7)

    # BACKWARD

    # compute loss based on model's output and true labels
    if   keras_loss_type is mean_squared_error:
      loss = K.mean( K.square(truth_tf - model.output) )
    elif keras_loss_type is mean_absolute_error:
      loss = K.mean( K.abs(truth_tf - model.output) )

    # compute gradient of loss with respect to inputs
    grad_loss = K.gradients(loss, [model.input])

    # create a function to be able to run this computation graph
    func = K.function(model.inputs + [truth_tf], grad_loss)
    keras_delta = func([np.expand_dims(inpt, axis=0), truth])

    numpynet_delta = numpynet_layer.delta

    assert np.allclose(keras_delta, numpynet_delta)

    # all passed


# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


def test_batchnorm_layer():
  '''
  Tests:
    the forward and backward functions of the batchnorm layer against keras

  Problems:
    Precison of allclose change with the images, I tried sorting the input,
    but it doesn't get any better.

  to be:
    different batch size
    update functions
  '''



  from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
  from keras.layers import BatchNormalization

  batch = 5

  inpt = np.random.uniform(0.,1.,(batch,10,10,3))

  b, w, h, c = inpt.shape

  bias = np.random.uniform(0.,1., size = (w,h,c))  # random biases
  scales = np.random.uniform(0.,1.,size = (w,h,c)) # random scales

  # Numpy_net model
  numpynet = BatchNorm_layer(scales=scales, bias=bias)

  # initializers must be callable with this syntax, I need those for dimensionality problems
  def bias_init(shape, dtype = None):
    return bias

  def gamma_init(shape, dtype = None):
    return scales

  def mean_init(shape, dtype = None):
    return inpt.mean(axis = 0)

  def var_init(shape, dtype = None):
    return inpt.var(axis = 0)

  # Keras Model
  inp = Input(shape = (b,w,h,c))
  x = BatchNormalization(momentum = 1., epsilon=1e-8, center=True, scale=True,
                         axis = -1,
                         beta_initializer            = bias_init,
                         gamma_initializer           = gamma_init,
                         moving_mean_initializer     = mean_init,
                         moving_variance_initializer = var_init)(inp)
  model = Model(inputs = [inp], outputs =  x)

  # Keras forward
  forward_out_keras = model.predict(np.expand_dims(inpt,axis = 0))[0,:,:,:,:]

  numpynet.forward(inpt)
  forward_out_numpynet = numpynet.output

  # Comparing outputs
  assert forward_out_numpynet.shape == (b,w,h,c)
  assert forward_out_numpynet.shape == forward_out_keras.shape              # same shape
  assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-6)  # same output

  x_norm = (numpynet.x - numpynet.mean)*numpynet.var

  # Own variable updates comparisons
  assert np.allclose(numpynet.x, inpt)
  assert numpynet.mean.shape == (w,h,c)
  assert numpynet.var.shape == (w,h,c)
  assert x_norm.shape == numpynet.x.shape
  assert np.allclose(numpynet.x_norm, x_norm)

  # BACKWARD

  # Opens a TensorFlow Session to Initialize Variables
  sess = tf.InteractiveSession()

  # Computes analytical output gradients w.r.t input and w.r.t trainable_weights
  # Kept them apart for clarity
  grad      = K.gradients(model.output, [model.input])
  gradients = K.gradients(model.output, model.trainable_weights)

  # Define 2 functions to compute the numerical values of grad and gradients
  func  = K.function(model.inputs + [model.output], grad)
  func2 = K.function(model.inputs + model.trainable_weights + [model.output], gradients)

  # Initialization of variables, code won't work without it
  sess.run(tf.global_variables_initializer())

  # Assigns Numerical Values
  updates     = func2([np.expand_dims(inpt, axis = 0)])
  delta_keras = func([np.expand_dims(inpt,axis = 0)])[0][0,:,:,:,:]

  # Initialization of numpynet delta to one (multiplication) and an empty array to store values
  numpynet.delta = np.ones(shape=inpt.shape)
  delta_numpynet = np.empty(shape=inpt.shape)

  # numpynet bacward, updates delta_numpynet
  numpynet.backward(delta_numpynet)

  # Testing delta, the precision change with the image
  assert delta_keras.shape == delta_numpynet.shape       # 1e-1 for random image, 1e-8 for dog
  assert np.allclose(delta_keras, delta_numpynet ,         atol=1e-1)

  # Testing scales updates
  assert updates[0].shape == numpynet.scales_updates.shape
  assert np.allclose(updates[0], numpynet.scales_updates,  atol=1e-05)

  # Testing Bias updates
  assert updates[1].shape == numpynet.bias_updates.shape
  assert np.allclose(updates[1], numpynet.bias_updates,    atol=1e-08)

  # All passed, but precision it's not consistent, missing update functions

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_connected_layer():
  '''
  Tests:
    if the forward is coherent with keras
    if the updates (weight, bias) and delta computed by the backward are correct

  to be tested:
    update function, keras update not clear.
  '''

  from NumPyNet.activations import Relu, Logistic, Linear, Tanh
  from NumPyNet.layers.connected_layer import Connected_layer
  from keras.layers import Dense

  keras_activ = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activ = [Relu, Logistic, Tanh, Linear]

  # Usefull variables initialization
  outputs = 10
  batch, w, h, c = (5, 10, 10, 3)
  inpt = np.random.uniform(0.,1., (batch, w, h, c))

  weights = np.random.uniform(low=0., high=1., size=(w * h * c, outputs))
  bias    = np.random.uniform(low=0.,  high=1., size=(outputs))

  for activ in range(0,4):

    # Numpy_net model
    numpynet_layer = Connected_layer(inpt.shape, outputs,
                                  activation = numpynet_activ[activ],
                                  weights = weights, bias = bias)
    # Keras Model
    inp = Input(shape=(w * h * c), batch_shape=(batch, w * h * c))
    x = Dense(outputs,activation=keras_activ[activ], input_shape=(batch,inpt.size))(inp)
    model = Model(inputs=[inp], outputs=x)

    # Set weights in Keras Model.
    model.set_weights([weights, bias])

    # FORWARD

    # Keras forward output
    forward_out_keras = model.predict(inpt.reshape(batch, -1))

    # Numpy_net forward output
    numpynet_layer.forward(inpt)
    forward_out_numpynet = numpynet_layer.output

    # Forward output Test
    assert np.allclose(forward_out_numpynet, forward_out_keras, atol = 1e-8)

    # BACKWARD

    # Output derivative in respect to input
    grad      = K.gradients(model.output, [model.input])

    # Output derivative respect to trainable_weights(Weights and Biases)
    gradients = K.gradients(model.output, model.trainable_weights)

    # Definning functions to compute those gradients
    func  = K.function(model.inputs + [model.output], grad)
    func2 = K.function(model.inputs + model.trainable_weights + [model.output], gradients)

    # Evaluation of Delta, weights_updates and bias_updates for Keras
    delta_keras = func( [inpt.reshape(batch, -1)])
    updates     = func2([inpt.reshape(batch, -1)])

    # Initialization of numpy_net starting delta to ones
    numpynet_layer.delta = np.ones(shape=(batch, outputs))

    # Initialization of global delta
    delta = np.zeros(shape=(batch, w, h, c))

    # Computation of delta, weights_update and bias updates for numpy_net
    numpynet_layer.backward(inpt, delta=delta)

    # Now the global variable delta is updated

    assert np.allclose(delta_keras[0].reshape(batch, w, h, c), delta, atol = 1e-8)
    assert np.allclose(updates[0], numpynet_layer.weights_update, atol = 1e-8)
    assert np.allclose(updates[1], numpynet_layer.bias_update,    atol = 1e-8)

    # all passed

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_maxpool_layer():
  '''
  Tests:
    if the numpy_net maxpool layer forward is consistent with Keras
    if the numpy_net maxpool layer backward is the same as Keras
    for differentsizes and strides
  to be:
  '''
  from NumPyNet.layers.maxpool_layer import Maxpool_layer
  from keras.layers import MaxPool2D

  sizes   = [(1,1), (3,3), (30,30)]
  strides = [(1,1), (2,2), (20,20)]

  for size in sizes:
    for stride in strides:
      for pad in [False, True]:

        inpt = np.random.uniform(0.,1.,(5, 50, 51, 3))
        batch, w, h, c = inpt.shape

        # numpynet layer initialization, FALSE (numpynet) == VALID (Keras)
        numpynet = Maxpool_layer(size, stride, padding = pad)

        if pad:
          keras_pad = 'same'
        else :
          keras_pad = 'valid'

        # Keras model initialization.
        inp = Input(shape = (w, h, c), batch_shape=inpt.shape)
        x = MaxPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
        model = Model(inputs=[inp], outputs=x)

        # Keras Output
        forward_out_keras = model.predict(inpt)

        # numpynet forward and output
        numpynet.forward(inpt)
        forward_out_numpynet = numpynet.output

        # Test for dimension and allclose of all output
        assert forward_out_numpynet.shape == forward_out_keras.shape
        assert np.allclose(forward_out_numpynet, forward_out_keras,   atol  = 1e-8)

        # BACKWARD

        # Compute the gradient of output w.r.t input
        gradient = K.gradients(model.output, [model.input])

        # Define a function to evaluate the gradient
        func = K.function(model.inputs + [model.output], gradient)

        # Compute delta for Keras
        delta_keras = func([inpt])[0]

        # Definition of starting delta for numpynet
        delta = np.zeros(inpt.shape)
        numpynet.delta = np.ones(numpynet.out_shape())

        # numpynet Backward
        numpynet.backward(delta=delta)

        assert delta.shape == delta_keras.shape
        assert delta.shape == inpt.shape
        assert np.allclose(delta, delta_keras, atol = 1e-8)

        # ok all passed

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_activation_layer():
  '''
  Tests:
    if the forward and the backward of Numpy_net are consistent with keras.
    if all the possible activation functions works with different batch_size
  to be:
  '''

  from NumPyNet.activations import Relu, Logistic, Linear, Tanh
  from NumPyNet.layers.activation_layer import Activation_layer
  from keras.layers import Activation

  keras_activ = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activ = [Relu, Logistic, Tanh, Linear]

  batch_sizes = [1,5,10]


  for batch in batch_sizes:
      # negative value for Relu testing
    inpt = np.random.uniform(-1., 1., size=(batch, 100, 100, 3))
    b,w,h,c = inpt.shape

    for act_fun in range(0,4):
      # numpynet model init
      numpynet = Activation_layer(activation=numpynet_activ[act_fun])

      # Keras Model init
      inp = Input(shape = inpt.shape[1:], batch_shape = (b,w,h,c))
      x = Activation(activation = keras_activ[act_fun])(inp)
      model = Model(inputs=[inp], outputs=x)

      # FORWARD

      # Keras Forward
      forward_out_keras = model.predict(inpt)

      # numpynet forwrd
      numpynet.forward(inpt)
      forward_out_numpynet = numpynet.output

      # Forward check (Shape and Values)
      assert forward_out_keras.shape == forward_out_numpynet.shape
      assert np.allclose(forward_out_keras, forward_out_numpynet)

      # BACKWARD

      # Gradient computation (Analytical)
      grad = K.gradients(model.output, [model.input])

      # Define a function to compute the gradient numerically
      func = K.function(model.inputs + [model.output], grad)

      # Keras delta
      keras_delta = func([inpt])[0] # It returns a list with one array inside.

      # numpynet delta init. (Multiplication with gradients)
      numpynet.delta = np.ones(shape=inpt.shape)

      # Global delta init.
      delta = np.empty(shape=inpt.shape)

      # numpynet Backward
      numpynet.backward(delta)

      # Check dimension and delta
      assert keras_delta.shape == delta.shape
      assert np.allclose(keras_delta, delta)
      # all passed

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_dropout_layer():
  '''
  Tests:
    Properties of the output, since the seed in tensorflow behaves differently
    from numpy.

    I'm not sure what I should test here.
  '''

  from NumPyNet.layers.dropout_layer import Dropout_layer

  # Set of probabilities
  probabilities = np.round(np.linspace(0.,1.,20),2)

  for prob in probabilities:

    batch = np.random.randint(low=1, high=10)

    # Random input
    inpt     = np.random.uniform(0.,1., size=(batch, 200, 201, 3))
    _, w, h, c = inpt.shape

    # Initialize the numpy_net model
    numpynet = Dropout_layer(prob)

    # Tensor Flow dropout, just to see if it works
    # forward_out_keras = K.eval(tf.nn.dropout(inpt, seed = None, keep_prob=prob))

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    zeros_out = np.count_nonzero(forward_out_numpynet == 0)

    if   prob == 1.:
      assert zeros_out ==  batch * w * h * c
    elif prob == 0.:
      assert zeros_out == 0
    else:
      assert forward_out_numpynet.shape == inpt.shape
      assert zeros_out != 0

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_shuffle_layer():
  '''
  Tests:
    if the forward out of the shuffle layer is the same as tensorflow
    if the backward out of the shuffle layer give the same output
  to be:
  '''
  from NumPyNet.layers.shuffler_layer import Shuffler_layer

  couples = [(2,12),(4,32),(4,48),(6,108)]

  for scale, channels in couples:

    # input initialization
    batch = 2
    inpt = np.random.uniform(0.,1., size=(batch, 100, 101, channels))

    # numpynet model
    numpynet = Shuffler_layer(scale)

    # FORWARD

    # Keras operation
    forward_out_keras = K.eval(tf.depth_to_space(inpt, block_size=scale))

    numpynet.forward(inpt)
    forward_out_numpynet = numpynet.output

    assert forward_out_numpynet.shape == forward_out_keras.shape
    assert np.allclose(forward_out_numpynet, forward_out_keras)

    # BACKWARD

    delta = np.random.uniform(0.,1., size=forward_out_keras.shape)

    delta_keras = K.eval(tf.space_to_depth(delta, block_size = scale))

    numpynet.delta = delta
    delta = delta.reshape(inpt.shape)

    numpynet.backward(delta)

    assert delta_keras.shape == delta.shape
    assert np.allclose(delta_keras, delta)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_shortcut_layer():
  '''
  Tests:
    shortcut layer forward
    shortcut layer backward
  to be:
  '''
  from NumPyNet.layers.shortcut_layer import Shortcut_layer
  from NumPyNet.activations import Relu, Logistic, Linear, Tanh
  from keras.layers import Add

  alphas  = np.round(np.linspace(0,1,3), 2)
  betas   = np.round(np.linspace(0,1,3), 2)
  batches = [1,5,10]

  keras_activations = [ 'tanh','linear','relu', 'sigmoid']
  numpynet_activations = [Tanh, Linear, Relu, Logistic]

  for keras_activ, numpynet_activ in zip(keras_activations, numpynet_activations):
    for alpha in alphas:
      for beta in betas:
        for batch in batches:

          inpt1      = np.random.uniform(-1., 1.,(batch, 100,100,3))
          inpt2      = np.random.uniform(-1., 1.,(batch, 100,100,3))
          b, w, h, c = inpt1.shape

          # numpynet model
          numpynet = Shortcut_layer(inpt1.shape, inpt2.shape,
                                 activation=numpynet_activ,
                                 alpha=alpha, beta=beta)

          # Keras Model, double input
          inp1  = Input(shape = (w, h, c), batch_shape = inpt1.shape)
          inp2  = Input(shape = (w, h, c), batch_shape = inpt2.shape)
          x     = Add()([inp1,inp2])
          out   = Activation(activation = keras_activ)(x)
          model = Model(inputs = [inp1,inp2], outputs = out)

          # FORWARD

          # Perform Add() for alpha*inpt and beta*inpt
          forward_out_keras = model.predict([alpha*inpt1, beta*inpt2])

          numpynet.forward(inpt1,inpt2)
          forward_out_numpynet = numpynet.output

          assert forward_out_keras.shape == forward_out_numpynet.shape
          assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-7)

          # BACKWARD

          grad = K.gradients(model.output, model.inputs)
          func = K.function(model.inputs + [model.output],grad)

          delta1, delta2 = func([alpha*inpt1, beta*inpt2])

          delta1 *= alpha
          delta2 *= beta

          delta      = np.zeros(inpt1.shape)
          prev_delta = np.zeros(inpt2.shape)

          numpynet.delta = np.ones(shape=(batch, w, h, c))
          numpynet.backward(delta, prev_delta)

          assert np.allclose(delta1, delta)
          assert np.allclose(delta2, prev_delta, atol = 1e-8)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_avgpool_layer():
  '''
  Tests:
    if the average pool layer forwards and backward are consistent with keras

  to be:
  '''
  from NumPyNet.layers.avgpool_layer import Avgpool_layer
  from keras.layers import AvgPool2D

  sizes   = [(1,1), (3,3), (30,30)]
  strides = [(1,1), (2,2), (20,20)]


  for size, stride in zip(sizes, strides):
    for pad in [False,True]:

      batch = np.random.randint(low=1, high=10)
      w, h, c = (100, 201, 3)
      inpt = np.random.uniform(0.,1., size=(batch, w, h, c))

      # Numpy_net model
      numpynet = Avgpool_layer(size=size, stride=stride, padding=pad)

      if pad:
        keras_pad = 'same'
      else :
        keras_pad = 'valid'

      # Keras model initialization.
      inp = Input(shape = (w, h, c), batch_shape=inpt.shape)
      x = AvgPool2D(pool_size=size, strides=stride, padding=keras_pad)(inp)
      model = Model(inputs=[inp], outputs=x)

      # Keras Output
      forward_out_keras = model.predict(inpt)

      # numpynet forward and output
      numpynet.forward(inpt)
      forward_out_numpynet = numpynet.output

      # Test for dimension and allclose of all output
      assert forward_out_numpynet.shape == forward_out_keras.shape
      assert np.allclose(forward_out_numpynet, forward_out_keras,   atol  = 1e-8)

      # BACKWARD

      # Compute the gradient of output w.r.t input
      gradient = K.gradients(model.output, [model.input])

      # Define a function to evaluate the gradient
      func = K.function(model.inputs + [model.output], gradient)

      # Compute delta for Keras
      delta_keras = func([inpt])[0]

      # Definition of starting delta for numpynet
      delta = np.zeros(inpt.shape)

      # numpynet Backward
      numpynet.backward(delta)

      # Back tests
      assert delta.shape == delta_keras.shape
      assert delta.shape == inpt.shape
      assert np.allclose(delta, delta_keras, atol = 1e-8)

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_convolutional_layer():
  '''
  Tests:
    if the convolutional layer forward is consistent with keras
    if the convolutional layer backward is consistent with keras

  to be:
    update
  '''
  from NumPyNet.activations import Relu, Logistic, Linear, Tanh
  from NumPyNet.layers.convolutional_layer import Convolutional_layer

  from keras.layers import Conv2D

  keras_activations = ['relu', 'sigmoid', 'tanh','linear']
  numpynet_activations = [Relu, Logistic, Tanh, Linear]
  channels_in = [1,3,5]
  channels_out= [5,10,20]

  sizes   = [(3,3),(20,20),(30,30)]
  strides = [(2,2), (10,10),(20,20)]

  padding = [False, True]

  for c_in in channels_in:
    for c_out in channels_out:
      for keras_activ, numpynet_activ in zip(keras_activations,numpynet_activations):
        for size,stride in zip(sizes,strides):
          for pad in padding:

            batch = np.random.randint(low=1, high=10)
            batch = 1
            if pad:
              keras_pad = 'same'
            else :
              keras_pad = 'valid'

            inpt       = np.random.uniform(-1., 1., size = (batch, 100, 100, c_in))
            b, w, h, c = inpt.shape
            # Shape (size1,size2,c_in, c_out), reshape inside numpynet.forward.
            filters    = np.random.uniform(-1., 1., size = size + (c,c_out))
            bias       = np.random.uniform(-1., 1., size = (c_out,))


            # Numpy_net model
            numpynet = Convolutional_layer(channels_out=c_out , inputs=inpt.shape,
                                        weights=filters, bias=bias,
                                        activation = numpynet_activ,
                                        size = size, stride = stride,
                                        padding = pad)

            # Keras model
            inp  = Input(shape = inpt.shape[1:], batch_shape = inpt.shape)
            Conv2d = Conv2D(filters=c_out,
                              kernel_size=size, strides=stride,
                              padding=keras_pad,
                              activation=keras_activ,
                              data_format='channels_last',
                              use_bias=True , bias_initializer='zeros',
                              dilation_rate=1)(inp)     # dilation rate = 1 is no dilation (I think)
            model = Model(inputs=[inp], outputs=[Conv2d])

            model.set_weights([filters, bias])

            # FORWARD

            print(c_in, c_out, keras_activ, size, stride, pad, keras_pad, '\n', sep = '\n')

            global forward_out_keras, forward_out_numpynet

            forward_out_keras = model.predict(inpt)

            numpynet.forward(inpt)
            forward_out_numpynet = numpynet.output

            assert forward_out_keras.shape == forward_out_numpynet.shape
            assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-04, rtol = 1e-3)

            # BACKWARD
            grad1 = K.gradients(model.output, [model.input])
            grad2 = K.gradients(model.output, model.trainable_weights)

            func1 = K.function(model.inputs + model.outputs, grad1 )
            func2 = K.function(model.inputs + model.trainable_weights + model.outputs, grad2)

            delta_keras = func1([inpt])[0]
            updates     = func2([inpt])      # it does something at least

            delta_numpynet = np.zeros(shape = inpt.shape)
            numpynet.backward(delta_numpynet, inpt)

            assert np.allclose(delta_numpynet, delta_keras)
            assert np.allclose(numpynet.weights_updates, updates[0], atol = 1e-4, rtol = 1e-3) # for a lot of operations, atol is lower
            assert np.allclose(numpynet.bias_updates, updates[1])

# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #

def test_softmax_layer():

  from NumPyNet.layers.softmax_layer import Softmax_layer

  from keras.losses import categorical_crossentropy
  from keras.layers import Softmax
  from keras import backend as K

  spatials = [False, True]

  for spatial in spatials:

    if spatial:
      axis = -1
    else :
      axis = (1,2,3)

    np.random.seed(123)
    inpt = np.random.uniform(low = 0., high = 1., size = (2,10,10,3))

    batch, w, h, c = inpt.shape

    truth = np.random.choice([0., 1.], p = [.5,.5], size=(batch,w,h,c))

    numpynet = Softmax_layer(groups = 1, temperature = 1., spatial = spatial)

    inp = Input(shape=(w,h,c), batch_shape = inpt.shape)
    x = Softmax(axis = axis)(inp)
    model = Model(inputs=[inp], outputs=x)

    forward_out_keras = model.predict(inpt)

    # definition of tensorflow variable
    truth_tf             = K.variable(truth.ravel())
    forward_out_keras_tf = K.variable(forward_out_keras.ravel())

    loss = categorical_crossentropy( truth_tf, forward_out_keras_tf)

    keras_loss = K.eval(loss)
    numpynet.forward(inpt, truth)
    numpynet_loss = numpynet.cost

    assert np.allclose(numpynet_loss, keras_loss)

    forward_out_numpynet = numpynet.output

    assert np.allclose(forward_out_keras, forward_out_numpynet, atol = 1e-8)

    # Forward passed, the backward is different though



# # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # # #


if __name__ == '__main__':
  test_shortcut_layer()
