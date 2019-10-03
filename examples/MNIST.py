#!/usr/bin/env python3
# -*- coding: utf-8 -*-

'''
Little example on how to use the Network class to create a model and perform
a basic classification of the MNIST dataset
'''

from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.network import Network
from NumPyNet.utils import to_categorical

import numpy as np
from sklearn import datasets
from sklearn.model_selection import train_test_split

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Example MNIST'


if __name__ == '__main__':

  digits = datasets.load_digits()
  X, y = digits.images, digits.target

  del digits

  # add channels to images
  X = np.asarray([[x, x, x] for x in X])
  X = X.transpose(0, 2, 3, 1)

  X_train, X_test, y_train, y_test = train_test_split(X, y,
                                                      test_size=.33,
                                                      random_state=42)

  batch = 16
  num_classes = len(set(y))

  del X, y

  # normalization to [0, 1]
  X_train *= 1. / 255.
  X_test  *= 1. / 255.

  n_train = X_train.shape[0]
  n_test  = X_test.shape[0]

  # transform y to array of dimension 10
  y_train = to_categorical(y_train)
  #y_test  = to_categorical(y_test)

  # Create the model

  model = Network(batch=batch)

  model.add(Input_layer(input_shape=(batch, *X_train[0].shape)))
  model.add(Convolutional_layer(input_shape=(batch, 32, 32, 3),
                                size=3, filters=32, stride=1, pad=False,
                                activation='Relu'))
  model.add(Maxpool_layer(size=2, stride=1, padding=False))
  model.add(Dropout_layer(prob=0.3))

  model.add(Convolutional_layer(input_shape=(batch, 16, 16, 32),
                                filters=64, activation='Relu',
                                size=3, stride=1, pad=False))
  model.add(Maxpool_layer(size=2, stride=1))
  model.add(Dropout_layer(prob=0.3))

  model.add(Convolutional_layer(input_shape=(batch, 8, 8, 64),
                                filters=64, activation='Relu',
                                size=2, stride=1, pad=False))
  model.add(Maxpool_layer(size=2, stride=1))
  model.add(Dropout_layer(prob=0.4))

  model.add(Connected_layer(input_shape=(batch, 4, 4, 128),
                            outputs=80, activation='Relu'))
  model.add(Dropout_layer(prob=0.3))
  model.add(Connected_layer(input_shape=(batch,80), outputs=num_classes))
  model.add(Softmax_layer(spatial=True))

  model.summary()

  # Fit the model on the training set

  model.fit(X=X_train, y=y_train, max_iter=1)

  # Test the prediction

  out = model.predict(X=X_test[0])

  print('True      label: {:d}'.format(y_test[0]))
  print('Predicted label: {:d}'.format(out.argmax()))

