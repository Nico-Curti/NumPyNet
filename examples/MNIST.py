#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Little example on how to use the Network class to create a model and perform
a basic classification of the MNIST dataset
'''

from time import time

#from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.cost_layer import cost_type
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from NumPyNet.network import Network
from NumPyNet.optimizer import Adam, SGD, Momentum
from NumPyNet.utils import to_categorical
from NumPyNet.metrics import accuracy_score

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score as sk_ac
import matplotlib.pyplot as plt

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Example MNIST'

if __name__ == '__main__':

  np.random.seed(123)

  digits = datasets.load_digits()
  X, y = digits.images, digits.target

  # del digits

  # add channels to images
  X = np.asarray([[x, x, x] for x in X])
  X = X.transpose(0, 2, 3, 1)

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=.33,
                                                      random_state=42)

  batch = 128
  num_classes = len(set(y))

  # del X, y

  # normalization to [0, 1]
  X_train *= 1. / 255.
  X_test  *= 1. / 255.

  # reduce the size of the data set for testing
  ############################################

  train_size = 512
  test_size  = 300

  X_train = X_train[:train_size, ...]
  y_train = y_train[:train_size]
  X_test  = X_test[ :test_size,  ...]
  y_test  = y_test[ :test_size]

  ############################################

  n_train = X_train.shape[0]
  n_test  = X_test.shape[0]

  # transform y to array of dimension 10 and in 4 dimension
  y_train = to_categorical(y_train).reshape(n_train, 1, 1, -1)
  y_test  = to_categorical(y_test).reshape(n_test, 1, 1, -1)

  # Create the model and training
  model = Network(batch=batch, input_shape=X_train.shape[1:])

  model.add(Convolutional_layer(size=3, filters=32, stride=1, pad=True, activation='Relu'))

  model.add(BatchNorm_layer())

  model.add(Maxpool_layer(size=2, stride=1, padding=True))

  model.add(Connected_layer(outputs=100, activation='Relu'))

  model.add(BatchNorm_layer())

  model.add(Connected_layer(outputs=num_classes, activation='Linear'))

  model.add(Softmax_layer(spatial=True, groups=1, temperature=1.))
  # model.add(Cost_layer(cost_type=cost_type.mse))

  print('*************************************')
  print('\n Total input dimension: {}'.format(X_train.shape), '\n')
  print('*************************************')

  model.compile(optimizer=SGD(lr=0.01, decay=0., lr_min=0., lr_max=np.inf))
  model.summary()

  truth = y_test.argmax(axis=3).ravel()

  model._fitted = True # for testing purpose

  out1       = model.predict(X=X_test)
  predicted1 = out1.argmax(axis=3).ravel()
  accuracy1  = accuracy_score(truth, predicted1)

  # accuracy test
  print('\nAccuracy Score      : {:.3f}'.format(accuracy1))

  print('\n***********START TRAINING***********\n')

  # Fit the model on the training set with timing

  tic = time()
  model.fit(X=X_train, y=y_train, max_iter=10)
  toc = time()
  train_time = toc - tic

  print('\n***********END TRAINING**************\n')

  # Test the prediction with timing
  tic  = time()
  out2 = model.predict(X=X_test)
  toc  = time()
  test_time = toc - tic

  predicted2 = out2.argmax(axis=3).ravel()
  accuracy2  = accuracy_score(truth, predicted2)

  print('Accuracy Score      : {:.3f}'.format(accuracy2))
  print('And it tooks {:.1f}s for training and {:.1f}s for predict\n'.format(train_time, test_time))
  # best score I could obtain was 94% with 10 epochs, lr = 0.01 %
