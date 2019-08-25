| **Author**   | **Project** | **Documentation**                                                                   | **Build Status**              |
|:------------:|:-----------:|:-----------------------------------------------------------------------------------:|:-----------------------------:|
|   [**N. Curti**](https://github.com/Nico-Curti) <br/> [**M. Ceccarelli**](https://github.com/Mat092)   |  **NumPyNet**  | [![docs](https://img.shields.io/badge/documentation-latest-blue.svg?style=plastic)](https://nico-curti.github.io/NumPyNet/) | **Linux/MacOS** : [![travis](https://travis-ci.com/Nico-Curti/NumPyNet.svg?branch=master)](https://travis-ci.com/Nico-Curti/NumPyNet) <br/> **Windows** : [![appveyor](https://ci.appveyor.com/api/projects/status/qbn3ml2q04j9rbat?svg=true)](https://ci.appveyor.com/project/Nico-Curti/numpynet) |

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

Implementation in **pure** numpy of neural networks models. This library is usefull as testing framework for optimized codes.

1. [Installation](#installation)
2. [Authors](#authors)
3. [License](#license)
4. [Contributions](#contributions)
5. [Acknowledgments](#acknowledgments)
6. [Citation](#citation)


## Installation

First of all ensure that a right Python version is installed (Python >= 3.4 is required). <!-- to check -->
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recomended.

Download the project or the latest release:

```
git clone https://github.com/Nico-Curti/NumPyNet
cd NumPyNet
```

To install the prerequisites type:

```
pip install -r ./requirements.txt
```

In the `NumPyNet` directory execute:

```
python setup.py install
```

or for installing in development mode:

```
python setup.py develop --user
```


## Authors

* **Nico Curti** [git](https://github.com/Nico-Curti), [unibo](https://www.unibo.it/sitoweb/nico.curti2)
* **Mattia Ceccarelli** [git](https://github.com/Mat092)

See also the list of [contributors](https://github.com/Nico-Curti/NumPyNet/contributors) [![GitHub contributors](https://img.shields.io/github/contributors/Nico-Curti/NumPyNet.svg?style=plastic)](https://github.com/Nico-Curti/NumPyNet/graphs/contributors/) who participated in this project.

## License

The `NumPyNet` package is licensed under the MIT "Expat" License. [![License](https://img.shields.io/github/license/mashape/apistatus.svg)](https://github.com/Nico-Curti/NumPyNet/blob/master/LICENSE.md)


### Contributions

Any contribution is more than welcome. Just fill an issue or a pull request and I will check ASAP!

If you want update the list of layer objects please pay attention to the syntax of the layer class and to the names of member functions/variables used to prevent the compatibility with other layers and utility functions.


### Acknowledgment

Thanks goes to all contributors of this project.

### Citation

Please cite `NumPyNet` if you use it in your research.

```tex
@misc{NumPyNet,
  author = {Nico Curti and Mattia Ceccarelli},
  title = {{N}um{P}y{N}et},
  year = {2019},
  publisher = {GitHub},
  journal = {GitHub repository},
  howpublished = {\url{https://github.com/Nico-Curti/NumPyNet}},
}
```

