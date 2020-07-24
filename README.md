| **Author**   | **Project** | **Documentation** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:-----------------:|:----------------:|:----------------:|:------------:|
|   [**N. Curti**](https://github.com/Nico-Curti) <br/> [**M. Ceccarelli**](https://github.com/Mat092) | **NumPyNet**  | [![docs](https://img.shields.io/badge/documentation-latest-blue.svg?style=plastic)](https://nico-curti.github.io/NumPyNet/) | **Linux/MacOS** : [![travis](https://travis-ci.com/Nico-Curti/NumPyNet.svg?branch=master)](https://travis-ci.com/Nico-Curti/NumPyNet) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/qbn3ml2q04j9rbat?svg=true)](https://ci.appveyor.com/project/Nico-Curti/numpynet) | **Codacy** : [![Codacy Badge](https://api.codacy.com/project/badge/Grade/bc07a2bf6ba84555a7b9647891cc309d)](https://www.codacy.com/manual/Nico-Curti/NumPyNet?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Nico-Curti/NumPyNet&amp;utm_campaign=Badge_Grade) <br/> **Codebeat** : [![Codebeat](https://codebeat.co/badges/3ac26bf3-44ae-47ff-9b93-2f9785c4a7d6)](https://codebeat.co/projects/github-com-nico-curti-numpynet-master) | [![codecov](https://codecov.io/gh/Nico-Curti/NumPyNet/branch/master/graph/badge.svg)](https://codecov.io/gh/Nico-Curti/NumPyNet) |

![NumPyNet CI](https://github.com/Nico-Curti/NumPyNet/workflows/NumPyNet%20CI/badge.svg)

[![GitHub pull-requests](https://img.shields.io/github/issues-pr/Nico-Curti/NumPyNet.svg?style=plastic)](https://github.com/Nico-Curti/NumPyNet/pulls)
[![GitHub issues](https://img.shields.io/github/issues/Nico-Curti/NumPyNet.svg?style=plastic)](https://github.com/Nico-Curti/NumPyNet/issues)

[![GitHub stars](https://img.shields.io/github/stars/Nico-Curti/NumPyNet.svg?label=Stars&style=social)](https://github.com/Nico-Curti/NumPyNet/stargazers)
[![GitHub watchers](https://img.shields.io/github/watchers/Nico-Curti/NumPyNet.svg?label=Watch&style=social)](https://github.com/Nico-Curti/NumPyNet/watchers)

<a href="https://github.com/UniboDIFABiophysics">
<div class="image">
<img src="https://cdn.rawgit.com/physycom/templates/697b327d/logo_unibo.png" width="90" height="90">
</div>
</a>

# Neural Networks in Pure NumPy - NumPyNet

Implementation in **pure** `Numpy` of neural networks models.
`NumPyNet` supports a syntax very close to the `Keras` one but it is written using **only** `Numpy` functions: in this way it is very light and fast to install and use/modify.

* [Overview](#overview)
* [Theory](#theory)
* [Prerequisites](#prerequisites)
* [Installation](#installation)
* [Efficiency](#efficiency)
* [Usage](#usage)
* [Contribution](#contribution)
* [References](#references)
* [Authors](#authors)
* [License](#license)
* [Acknowledgments](#acknowledgments)
* [Citation](#citation)

## Overview

`NumPyNet` is born as educational framework for the study of Neural Network models.
It is written trying to balance code readability and computational performances and it is enriched with a large documentation to better understand the functionality of each script.
The library is written in pure `Python` and the only external library used is [`Numpy`](http://www.numpy.org/) (a base package for the scientific research).

Despite all common libraries are correlated by a wide documentation is often difficult for novel users to move around the many hyper-links and papers cited in them.
`NumPyNet` tries to overcome this problem with a minimal mathematical documentation associated to each script and a wide range of comments inside the code.

An other "problem" to take in count is related to performances.
Libraries like [`Tensorflow`](http://tensorflow.org/) are certainly efficient from a computational point-of-view and the numerous wrappers (like *Keras* library) guarantee an extremely simple user interface.
On the other hand, the deeper functionalities of the code and the implementation strategies used are unavoidably hidden behind tons of code lines.
In this way the user can perform complex computational tasks using the library as black-box package.
`NumPyNet` wants to overcome this problem using simple `Python` codes, with extremely readability also for novel users, to better understand the symmetry between mathematical formulas and code.

## Theory

We propose a full list of mathematical instructions about each layer model into our [online](https://nico-curti.github.io/NumPyNet) documentation.
Each script is also combined with a very simple usage into its `__main__` section: in this way we can easily visualize the results produced by each function into a test image.

The full list of available layers is the following:

- [Activation Layer](./docs/NumPyNet/layers/activation_layer.md)
- [Avgpool Layer](./docs/NumPyNet/layers/avgpool_layer.md)
- [BatchNorm Layer](./docs/NumPyNet/layers/batchnorm_layer.md)
- [Connected Layer](./docs/NumPyNet/layers/connected_layer.md)
- [Convolutional Layer](./docs/NumPyNet/layers/convolutional_layer.md)
- [Cost Layer](./docs/NumPyNet/layers/cost_layer.md)
- [DropOut Layer](./docs/NumPyNet/layers/dropout_layer.md)
- [Input Layer](./docs/NumPyNet/layers/input_layer.md)
- [L1norm Layer](./docs/NumPyNet/layers/l1norm_layer.md)
- [L2norm Layer](./docs/NumPyNet/layers/l2norm_layer.md)
- [Logistic Layer](./docs/NumPyNet/layers/logistic_layer.md)
- [LSTM Layer](./docs/NumPyNet/layers/lstm_layer.md) : **TODO**
- [MaxPool Layer](./docs/NumPyNet/layers/maxpool_layer.md)
- [RNN Layer](./docs/NumPyNet/layers/rnn_layer.md) : **TODO**
- [Route Layer](./docs/NumPyNet/layers/route_layer.md)
- [PixelShuffle Layer](./docs/NumPyNet/layers/pixelshuffle_layer.md)
- [Shortcut Layer](./docs/NumPyNet/layers/shortcut_layer.md)
- [UpSample Layer](./docs/NumPyNet/layers/upsample-layer.md)
- [YOLO Layer](./docs/NumPyNet/layers/yolo_layer.md)

## Prerequisites

Python version supported : ![Python version](https://img.shields.io/badge/python-2.7|3.4|3.5|3.6|3.7|3.8-blue.svg)

First of all ensure that a right `Python` version is installed (`Python` >= 2.7 is required).
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recommended.

**Note:** some utilities (e.g image and video objects) required `OpenCV` library.
`OpenCV` does not support `Python2.6` and `Python3.3`.
If you are working with these two versions, please consider to remove the utilities objects or simply convert the `OpenCV` dependencies with other packages (like [Pillow](https://pypi.org/project/Pillow) or [scikit-image](https://pypi.org/project/scikit-image)).

## Installation

Download the project or the latest release:

```bash
git clone https://github.com/Nico-Curti/NumPyNet
cd NumPyNet
```

The principal `NumPyNet` requirements are `numpy`, `matplotlib`, `enum34` and `configparser`.
For layer visualizations we use the `Pillow` package while the `OpenCV` library is used to wrap some useful image processing objects.
You can simply install the full list of requirements with the command:

```bash
pip install -r ./requirements.txt
```

The testing procedure of this library is performed using `PyTest` and `Hypothesis` packages.
Please consider to install also these libraries if you want a complete installation of `NumPyNet`.

In the `NumPyNet` directory execute:

```bash
python setup.py install
```

or for installing in development mode:

```bash
python setup.py develop --user
```

### Testing

A full set of testing functions is provided in the [testing](https://github.com/Nico-Curti/NumPyNet/tree/master/testing) directory.
The tests are performed against the `Keras` implementation of the same functions (we tested only the `Tensorflow` backend in our simulations).
You can run the full list of tests with:

```bash
cd NumPyNet/testing
pytest
```

The continuous integration using `Travis` and `Appveyor` tests each function in every commit, thus pay attention to the status badges before use this package or use the latest stable version available.

## Efficiency

**TODO**

## Usage

First of all we have to import the main modules of the `NumPyNet` package as


```python
from NumPyNet.network import Network
from NumPyNet.layers.connected_layer import Connected_layer
from NumPyNet.layers.convolutional_layer import Convolutional_layer
from NumPyNet.layers.maxpool_layer import Maxpool_layer
from NumPyNet.layers.softmax_layer import Softmax_layer
from NumPyNet.layers.batchnorm_layer import BatchNorm_layer
from NumPyNet.optimizer import Adam
```

Now we can try to create a very simple model able to classify the well known MNIST-digit dataset.
The MNIST dataset can be extracted from the `sklearn` library as

```python
from sklearn import datasets
from sklearn.model_selection import train_test_split

digits = datasets.load_digits()
X, y = digits.images, digits.target

X = np.asarray([np.dstack((x, x, x)) for x in X])
X = X.transpose(0, 2, 3, 1)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=.33, random_state=42)
```

Now we have to create our model.
We can use a syntax very close to the *Keras* one and simply define a model object adding a series of layers

```python
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
```

The model automatically creates an `InputLayer` if it is not explicitly provided, but pay attention to the right `input_shape` values!

Before feeding our model we have to convert the image dataset into categorical variables.
To this purpose we can use the simple *utilities* of the `NumPyNet` package.

```python
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
```

Now you can run your `fit` function to train the model as

```python
model.fit(X=X_train, y=y_train, max_iter=100)
```

and evaluate the results on the testing set

```python
loss, out = model.evaluate(X=X_test, truth=y_test, verbose=True)

truth = from_categorical(y_test)
predicted = from_categorical(out)
accuracy  = mean_accuracy_score(truth, predicted)

print('\nLoss Score: {:.3f}'.format(loss))
print('Accuracy Score: {:.3f}'.format(accuracy))
```

You should see something like this

```bash
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
```

Obviously the execution time can vary according to your available resources!

You can find a full list of example scripts [here](https://github.com/Nico-Curti/NumPyNet/tree/master/examples)

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/NumPyNet/blob/master/ISSUE_TEMPLATE.md) or a [pull request](https://github.com/Nico-Curti/NumPyNet/blob/master/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/Nico-Curti/NumPyNet/blob/master/CONTRIBUTING.md) for further informations about how to contribute with this project.

## References

<blockquote>1- Travis Oliphant. "NumPy: A guide to NumPy", USA: Trelgol Publishing, 2006. </blockquote>

<blockquote>2- Bradski, G. "The OpenCV Library", Dr. Dobb's Journal of Software Tools, 2000. </blockquote>

**TODO**

## Authors

* <img src="https://avatars0.githubusercontent.com/u/24650975?s=400&v=4" width="25px"> **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)

* <img src="https://avatars0.githubusercontent.com/u/41483077?s=400&v=4" width="25px;"/> **Mattia Ceccarelli** [git](https://github.com/Mat092)

See also the list of [contributors](https://github.com/Nico-Curti/NumPyNet/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/NumPyNet.svg?style=plastic)](https://github.com/Nico-Curti/NumPyNet/graphs/contributors/) who participated in this project.

## License

The `NumPyNet` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/NumPyNet/blob/master/LICENSE.md)

## Acknowledgment

Thanks goes to all contributors of this project.

## Citation

If you have found `NumPyNet` helpful in your research, please consider citing this project repository

```BibTex
@misc{NumPyNet,
  author = {Curti, Nico and Ceccarelli, Mattia},
  title = {NumPyNet},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Nico-Curti/NumPyNet}},
}
```
