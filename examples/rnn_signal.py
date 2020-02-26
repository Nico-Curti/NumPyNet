#!/usr/bin/env python
# -*- coding: utf-8 -*-

'''
Little example on how to use a recurrent neural network to predict a math function

Reference: https://www.datatechnotes.com/2018/12/rnn-example-with-keras-simplernn-in.html
'''

# from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.rnn_layer import RNN_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.cost_layer import Cost_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.network import Network
from NumPyNet.optimizer import RMSprop
from NumPyNet.metrics import mean_absolute_error

import numpy as np
import pylab as plt

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

np.random.seed(42)


if __name__ == '__main__':

  Npoints = 1000
  train_size = 800

  time = np.arange(0, Npoints)
  noisy_signal = np.sin(0.02 * time) + 2 * np.random.rand(Npoints)

  window_size = 4
  shift = 1

  X = np.lib.stride_tricks.as_strided(noisy_signal,
                                      shape=(Npoints - window_size + 1, window_size),
                                      strides=(shift * 8, 8)) # 8 is the sizeof(float64)

  y = X[window_size:, 0].copy()
  X = X[: -window_size ].copy()

  # Reshape the data according to a 4D tensor
  num_samples, size = X.shape

  if size != window_size:
    raise ValueError('Something goes wrong with the stride trick!')

  if X.max() > noisy_signal.max() or X.min() < noisy_signal.min():
    raise ValueError('Something goes wrong with the stride trick!')


  X = X.reshape(num_samples, 1, 1, size)

  X_train, X_test = X[:train_size, ...], X[train_size:, ...]
  y_train, y_test = y[:train_size, ...], y[train_size:, ...]

  batch = 16
  step = 4

  # Create the model and training
  model = Network(batch=batch, input_shape=X_train.shape[1:])

  model.add(RNN_layer(outputs=32, steps=step, activation='Relu'))
  model.add(Connected_layer(outputs=8, activation='Relu'))
  model.add(Connected_layer(outputs=1, activation='Linear'))
  model.add(Cost_layer(cost_type='mse'))

  model.compile(optimizer=RMSprop(), metrics=[mean_absolute_error])

  print('*************************************')
  print('\n Total input dimension: {}'.format(X_train.shape), '\n')
  print('**************MODEL SUMMARY***********')

  model.summary()

  print('\n***********START TRAINING***********\n')

  # Fit the model on the training set
  model.fit(X=X_train, y=y_train, max_iter=20)

  print('\n***********START TESTING**************\n')

  # Test the prediction with timing
  loss, out = model.evaluate(X=X_test, truth=y_test, verbose=True)

  mae = mean_absolute_error(y_test, out)

  print('\n')
  print('Loss Score: {:.3f}'.format(loss))
  print('MAE Score: {:.3f}'.format(mae))

  # concatenate the prediction

  train_predicted = model.predict(X=X_train, verbose=False)
  test_predicted  = model.predict(X=X_test, verbose=False)

  predicted = np.concatenate((train_predicted, test_predicted), axis=0)

  fig, ax = plt.subplots(nrows=1, ncols=1, figsize=(8, 6))
  ax.plot(time[:- window_size*2], noisy_signal[:- window_size*2], 'b-', alpha=.75, label='true noisy signal')
  ax.plot(time[:predicted.shape[0]], predicted[:, 0, 0, 0], 'r-', alpha=.5, label='predicted signal')

  ax.vlines(time[train_predicted.shape[0]], noisy_signal.min(), noisy_signal.max(), colors='k', linestyle='dashed')

  ax.set_xlabel('Time', fontsize=14)
  ax.set_ylabel('Signal', fontsize=14)

  fig.tight_layout()

  plt.show()
