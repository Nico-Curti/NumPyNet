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

Implementation in **pure** numpy of neural networks models.
This library is usefull as testing framework for optimized codes.

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

**TODO**

## Theory

**TODO**

## Prerequisites

Python version supported : ![Python version](https://img.shields.io/badge/python-2.7|3.4|3.5|3.6|3.7|3.8-blue.svg)

First of all ensure that a right Python version is installed (Python >= 2.7 is required).
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recomended.

**Note:** some utilities (e.g image and video objects) required OpenCV library.
OpenCV does not support `Python2.6` and `Python3.3`.
If you are working with these two versions, please consider to remove the utilities objects or simply convert the OpenCV dependencies with other packages (like [Pillow](https://pypi.org/project/Pillow) or [scikit-image](https://pypi.org/project/scikit-image)).

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

**TODO**

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
