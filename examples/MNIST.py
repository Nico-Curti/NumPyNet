#!/usr/bin/env python3
# -*- coding: utf-8 -*-

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'Example MNIST'

'''
Little example on how to use the Network class to create a model and perform
a basic classification of the CIFAR100 dataset 
'''

from NumPyNet.layers.input_layer import Input_layer
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.dropout_layer import Dropout_layer
from NumPyNet.network import Network

import numpy as np 

from keras.datasets import cifar10 # keras comes in handy for datasets loading

(x_train, y_train), (x_test, y_test) = cifar10.load_data()

# reduce the dataset to 1/5 of the original size
x_train = x_train[:100,:,:,:]
y_train = y_train[:100,:]
x_test  = x_test [:20, :,:,:]
y_test  = y_test [:20, :]


x_train = x_train / 255.   # normalization to [0, 1]
x_test  = x_test  / 255.

n_train = len(x_train)
n_test  = len(x_test)


# transform y to array of dimension 10
_y_train = np.zeros(shape=(n_train, 10))
_y_test  = np.zeros(shape=(n_test , 10))

for i in range(n_train):
  _y_train[i][y_train[i]] = 1.

for i in range(n_test):
  _y_test[i][y_test[i]] = 1.  

y_train = _y_train
y_test  = _y_test

#%%

batch=16
num_classes = 10

model = Network(batch=batch, input_shape=(32, 32, 3))

model.add(Input_layer(input_shape=(batch, 32, 32, 3)))
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
                              size=3, stride=1, pad=False))
model.add(Maxpool_layer(size=2, stride=1))
model.add(Dropout_layer(prob=0.4))

model.add(Connected_layer(input_shape=(batch, 4, 4, 128),
                          outputs=80, activation='Relu'))
model.add(Dropout_layer(prob=0.3))
model.add(Connected_layer(input_shape=(batch,80), outputs=num_classes))
model.add(Softmax_layer(spatial=True))


model.fit(X=x_train, y=y_train, max_iter=1)






