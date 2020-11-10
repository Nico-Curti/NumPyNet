.. NumPyNet package documentation master file, created by
   sphinx-quickstart on Fri Oct  2 12:42:24 2020.
   You can adapt this file completely to your liking, but it should at least
   contain the root `toctree` directive.

Welcome to NumPyNet package's documentation!
==================================================================

Implementation in **pure** `Numpy` of neural networks models.
`NumPyNet` supports a syntax very close to the `Keras` one but it is written using **only** `Numpy` functions: in this way it is very light and fast to install and use/modify.

Overview
========

`NumPyNet` is born as educational framework for the study of Neural Network models.
It is written trying to balance code readability and computational performances and it is enriched with a large documentation to better understand the functionality of each script.
The library is written in pure `Python` and the only external library used is `Numpy`_ (a base package for the scientific research).

Despite all common libraries are correlated by a wide documentation is often difficult for novel users to move around the many hyper-links and papers cited in them.
`NumPyNet` tries to overcome this problem with a minimal mathematical documentation associated to each script and a wide range of comments inside the code.

An other "problem" to take in count is related to performances.
Libraries like `Tensorflow`_ are certainly efficient from a computational point-of-view and the numerous wrappers (like *Keras* library) guarantee an extremely simple user interface.
On the other hand, the deeper functionalities of the code and the implementation strategies used are unavoidably hidden behind tons of code lines.
In this way the user can perform complex computational tasks using the library as black-box package.
`NumPyNet` wants to overcome this problem using simple `Python` codes, with extremely readability also for novel users, to better understand the symmetry between mathematical formulas and code.

Usage example
=============

First of all we have to import the main modules of the `NumPyNet` package as

.. code-block:: python

  from NumPyNet.network import Network
  from NumPyNet.layers.connected_layer import Connected_layer
  from NumPyNet.layers.convolutional_layer import Convolutional_layer
  from NumPyNet.layers.maxpool_layer import Maxpool_layer
  from NumPyNet.layers.softmax_layer import Softmax_layer
  from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
  from NumPyNet.optimizer import Adam

Now we can try to create a very simple model able to classify the well known MNIST-digit dataset.
The MNIST dataset can be extracted from the `sklearn` library as

.. code-block:: python

  from sklearn import datasets
  from sklearn.model_selection import train_test_split

  digits = datasets.load_digits()
  X, y = digits.images, digits.target

  X = np.asarray([np.dstack((x, x, x)) for x in X])
  X = X.transpose(0, 2, 3, 1)

  X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)

Now we have to create our model.
We can use a syntax very close to the *Keras* one and simply define a model object adding a series of layers

.. code-block:: python

  model = Network(batch=batch, input_shape=X_train.shape[1:])

  model.add(Convolutional_layer(size=3, filters=32, stride=1, pad=True, activation='Relu'))
  model.add(BatchNorm_layer())
  model.add(Maxpool_layer(size=2, stride=1, padding=True))
  model.add(Connected_layer(outputs=100, activation='Relu'))
  model.add(BatchNorm_layer())
  model.add(Connected_layer(outputs=num_classes, activation='Linear'))
  model.add(Softmax_layer(spatial=True, groups=1, temperature=1.))

  model.compile(optimizer=Adam(), metrics=[accuracy])

  model.summary()

The model automatically creates an `InputLayer` if it is not explicitly provided, but pay attention to the right `input_shape` values!

Before feeding our model we have to convert the image dataset into categorical variables.
To this purpose we can use the simple *utilities* of the `NumPyNet` package.

.. code-block:: python

  from NumPyNet.utils import to_categorical
  from NumPyNet.utils import from_categorical
  from NumPyNet.metrics import mean_accuracy_score

  # normalization to [0, 1]
  X_train *= 1. / 255.
  X_test  *= 1. / 255.

  n_train = X_train.shape[0]
  n_test  = X_test.shape[0]

  # transform y to array of dimension 10 and in 4 dimension
  y_train = to_categorical(y_train).reshape(n_train, 1, 1, -1)
  y_test  = to_categorical(y_test).reshape(n_test, 1, 1, -1)

Now you can run your `fit` function to train the model as

.. code-block:: python

  model.fit(X=X_train, y=y_train, max_iter=100)

and evaluate the results on the testing set

.. code-block:: python

  loss, out = model.evaluate(X=X_test, truth=y_test, verbose=True)

  truth = from_categorical(y_test)
  predicted = from_categorical(out)
  accuracy  = mean_accuracy_score(truth, predicted)

  print('\nLoss Score: {:.3f}'.format(loss))
  print('Accuracy Score: {:.3f}'.format(accuracy))

You should see something like this

.. code-block:: bash

  layer       filters  size              input                output
     0 input                   128 x   8 x   3 x   8   ->   128 x   8 x   3 x   8
     1 conv     32 3 x 3 / 1   128 x   8 x   3 x   8   ->   128 x   8 x   3 x  32  0.000 BFLOPs
     2 batchnorm                       8 x   3 x  32 image
     3 max         2 x 2 / 1   128 x   8 x   3 x  32   ->   128 x   7 x   2 x  32
     4 connected               128 x   7 x   2 x  32   ->   128 x 100
     5 batchnorm                       1 x   1 x 100 image
     6 connected               128 x   1 x   1 x 100   ->   128 x  10
     7 softmax x entropy                                    128 x   1 x   1 x  10

  Epoch 1/10
  512/512 |██████████████████████████████████████████████████| (0.7 sec/iter) loss: 26.676 accuracy: 0.826

  Epoch 2/10
  512/512 |██████████████████████████████████████████████████| (0.6 sec/iter) loss: 22.547 accuracy: 0.914

  Epoch 3/10
  512/512 |██████████████████████████████████████████████████| (0.7 sec/iter) loss: 21.333 accuracy: 0.943

  Epoch 4/10
  512/512 |██████████████████████████████████████████████████| (0.6 sec/iter) loss: 20.832 accuracy: 0.963

  Epoch 5/10
  512/512 |██████████████████████████████████████████████████| (0.5 sec/iter) loss: 20.529 accuracy: 0.975

  Epoch 6/10
  512/512 |██████████████████████████████████████████████████| (0.3 sec/iter) loss: 20.322 accuracy: 0.977

  Epoch 7/10
  512/512 |██████████████████████████████████████████████████| (0.3 sec/iter) loss: 20.164 accuracy: 0.986

  Epoch 8/10
  512/512 |██████████████████████████████████████████████████| (0.3 sec/iter) loss: 20.050 accuracy: 0.992

  Epoch 9/10
  512/512 |██████████████████████████████████████████████████| (0.3 sec/iter) loss: 19.955 accuracy: 0.994

  Epoch 10/10
  512/512 |██████████████████████████████████████████████████| (0.3 sec/iter) loss: 19.875 accuracy: 0.996

  Training on 10 epochs took 21.6 sec

  300/300 |██████████████████████████████████████████████████| (0.0 sec/iter) loss: 10.472
  Prediction on 300 samples took 0.1 sec

  Loss Score: 2.610
  Accuracy Score: 0.937

Obviously the execution time can vary according to your available resources!
You can find a full list of example scripts here_


.. _`Numpy`: http://www.numpy.org/
.. _`Tensorflow`: http://tensorflow.org/
.. _here: https://github.com/Nico-Curti/NumPyNet/tree/master/examples

.. toctree::
   :maxdepth: 2
   :caption: Contents:

   API/modules
   references