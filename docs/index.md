| **Author**   | **Project** | **Documentation** | **Build Status** | **Code Quality** | **Coverage** |
|:------------:|:-----------:|:-----------------:|:----------------:|:----------------:|:------------:|
|   [**N. Curti**](https://github.com/Nico-Curti) <br/> [**M. Ceccarelli**](https://github.com/Mat092) | **NumPyNet**  | [![docs](https://img.shields.io/badge/documentation-latest-blue.svg?style=plastic)](https://nico-curti.github.io/NumPyNet/) | **Linux/MacOS** : [![travis](https://travis-ci.com/Nico-Curti/NumPyNet.svg?branch=master)](https://travis-ci.com/Nico-Curti/NumPyNet) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/qbn3ml2q04j9rbat?svg=true)](https://ci.appveyor.com/project/Nico-Curti/numpynet) | **Codacy** : [![Codacy Badge](https://api.codacy.com/project/badge/Grade/bc07a2bf6ba84555a7b9647891cc309d)](https://www.codacy.com/manual/Nico-Curti/NumPyNet?utm_source=github.com&amp;utm_medium=referral&amp;utm_content=Nico-Curti/NumPyNet&amp;utm_campaign=Badge_Grade) <br/> **Codebeat** : [![Codebeat](https://codebeat.co/badges/3ac26bf3-44ae-47ff-9b93-2f9785c4a7d6)](https://codebeat.co/projects/github-com-nico-curti-numpynet-master) | [![codecov](https://codecov.io/gh/Nico-Curti/NumPyNet/branch/master/graph/badge.svg)](https://codecov.io/gh/Nico-Curti/NumPyNet) |

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
This library is useful as testing framework for optimized codes.

## Table of Contents

1. [Introduction](#introduction)
2. [Layers](#layers)
3. [Installation](#installation)
4. [Contributions](#contributions)
5. [References](#references)
6. [Authors](#authors)
7. [License](#license)
8. [Acknowledgment](#acknowledgment)
9. [Citation](#citation)

## Introduction

**TODO:** Insert here introduction about neural network models starting from the Simple Perceptron model provided [here](https://gist.github.com/Nico-Curti/358b7a2ffed1abbb57ee87a5338ca073)

Layers are the main core of a Neural Network model: every one of them performs, on its respective input, a different operation.
The concatenation of multiple Layer form a CNN, where for concatenation we mean that the output of a layer become the input of the next one.

In the `NumPyNet` framework a Layer is a python `class`, this allow the users to instantiate an object of the chosen type and call one of its methods.

Main Method:

* **forward** : this function is defined for every Layer and perform the so-called *forward pass*, that is the implementation of the transformation the Layer performs on the Input.
It usually receives as argument just the output of the previous Layer

* **backward** : this function is defined for every Layer and perform the so-called *backward pass*, that is an implementation of the BackPropagation algorithm for the error.
It computes the delta to be back-propagated during *training* and, eventually, the updates to trainable weights
It usually receives as input only the global delta of the network, on which it performs a transformation, depending on the layer.

* **update** : this function is defined only for layers with trainable weights.

## Layers

Some texts for introduction:

* [Activation Layer](./NumPyNet/layers/activation_layer.md)
* [Avgpool Layer](./NumPyNet/layers/avgpool_layer.md)
* [BatchNorm Layer](./NumPyNet/layers/batchnorm_layer.md)
* [Connected Layer](./NumPyNet/layers/connected_layer.md)
* [Convolutional Layer](./NumPyNet/layers/convolutional_layer.md)
* [Cost Layer](./NumPyNet/layers/cost_layer.md)
* [DropOut Layer](./NumPyNet/layers/dropout_layer.md)
* [Input Layer](./NumPyNet/layers/input_layer.md)
* [L1norm Layer](./NumPyNet/layers/l1norm_layer.md)
* [L2norm Layer](./NumPyNet/layers/l2norm_layer.md)
* [Logistic Layer](./NumPyNet/layers/logistic_layer.md)
* [MaxPool Layer](./NumPyNet/layers/maxpool_layer.md)
* [Route Layer](./NumPyNet/layers/route_layer.md)
* [PixelShuffle Layer](./NumPyNet/layers/pixelshuffle_layer.md)
* [Shortcut Layer](./NumPyNet/layers/shortcut_layer.md)
* [UpSample Layer](./NumPyNet/layers/upsample-layer.md)
* [YOLO Layer](./NumPyNet/layers/yolo_layer.md)

## Installation

Python version supported :

![Python version](https://img.shields.io/badge/python-2.7|3.4|3.5|3.6|3.7|3.8-blue.svg)

First of all ensure that a right Python version is installed (Python >= 2.7 is required).
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recommended.

**Note:** some utilities (e.g image and video objects) required OpenCV library.
OpenCV does not support `Python2.6` and `Python3.3`.
If you are working with these two versions, please consider to remove the utilities objects or simply convert the OpenCV dependencies with other packages (like [Pillow](https://pypi.org/project/Pillow) or [scikit-image](https://pypi.org/project/scikit-image)).

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

## Contribution

Any contribution is more than welcome :heart:. Just fill an [issue](https://github.com/Nico-Curti/NumPyNet/blob/master/ISSUE_TEMPLATE.md) or a [pull request](https://github.com/Nico-Curti/NumPyNet/blob/master/PULL_REQUEST_TEMPLATE.md) and we will check ASAP!

See [here](https://github.com/Nico-Curti/NumPyNet/blob/master/CONTRIBUTING.md) for further informations about how to contribute with this project.

## References

**TODO**

## Authors

* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)
* **Mattia Ceccarelli** [git](https://github.com/Mat092)

See also the list of [contributors](https://github.com/Nico-Curti/NumPyNet/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/NumPyNet.svg?style=plastic)](https://github.com/Nico-Curti/NumPyNet/graphs/contributors/) who participated in this project.

## License

The `NumPyNet` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/NumPyNet/blob/master/LICENSE.md)

## Acknowledgment

Thanks goes to all contributors of this project.

## Citation

If you have found `NumPyNet` helpful in your research, please consider citing this project repository

```tex
@misc{NumPyNet,
  author = {Curti, Nico and Ceccarelli, Mattia},
  title = {NumPyNet},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Nico-Curti/NumPyNet}},
}
```
