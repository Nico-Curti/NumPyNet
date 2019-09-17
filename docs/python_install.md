## Python Installation

First of all ensure that a right Python version is installed (Python >= 2.7 is required).
The [Anaconda/Miniconda](https://www.anaconda.com/) python version is recomended.

**Note:** some utilities (e.g image and video objects) required OpenCV library.
OpenCV does not support `Python2.6` and `Python3.3`.
If you are working with these two versions, please consider to remove the utilities objects or simply convert the OpenCV dependencies with other packages (like [Pillow](https://pypi.org/project/Pillow) or [Scikit-Image](https://pypi.org/project/scikit-image)).

First of all download the project or the latest release:

```bash
git clone https://github.com/Nico-Curti/NumPyNet
cd NumPyNet
```

### Installing prerequisites

The principal NumPyNet requirements are `numpy`, `matplotlib`, `enum34` and `configparser`.
For layer visualizations we use the `Pillow` package while the `OpenCV` library is used to wrap some useful image processing objects.
You can simply install the full list of requirements with the command:

```bash
pip install -r ./requirements.txt
```

The testing of this library is performed using `PyTest` and `Hypothesis` packages.
Please consider to install also these libraries if you want a complete installation of NumPyNet.

### Installation from sources

In the `NumPyNet` directory execute:

```bash
python setup.py install
```

for a permanent installation, or simply:

```bash
python setup.py develop --user
```

for an installation in development mode.

### Testing

A full set of testing functions is provided in the [testing]() directory.
The tests are performed against the `Keras` implementation of the same functions (we tested only the Tensorflow backend in our simulations).
You can run the full list of tests with:

```bash
cd NumPyNet/testing
pytest
```

The continuous integration using `Travis` and `Appveyor` tests each function in every commit, thus pay attention to the status badges before use this package or use the latest stable version available.
