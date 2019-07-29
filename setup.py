#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import print_function

import os

try:
  from setuptools import setup
  from setuptools import find_packages

except ImportError:
  from distutils.core import setup
  from distutils.core import find_packages


from NumPyNet.build import get_requires
from NumPyNet.build import read_description

here = os.path.abspath(os.path.dirname(__file__))

# Package meta-data.
NAME = 'NumPyNet'
DESCRIPTION = 'Neural Networks Library in pure Numpy'
URL = 'https://github.com/Nico-Curti/NumPyNet'
EMAIL = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
AUTHOR = ['Mattia Ceccarelli', 'Nico Curti']
REQUIRES_PYTHON = '>=3.4'
VERSION = None
KEYWORDS = 'neural-networks deep-neural-networks deep-learning image-classification super-resolution'

README_FILENAME = os.path.join(here, 'README.md')
REQUIREMENTS_FILENAME = os.path.join(here, 'requirements.txt')
VERSION_FILENAME = os.path.join(here, 'Pyron', '__version__.py')

# Import the README and use it as the long-description.
# Note: this will only work if 'README.md' is present in your MANIFEST.in file!
try:
  LONG_DESCRIPTION = read_description(README_FILENAME)

except FileNotFoundError:
  LONG_DESCRIPTION = DESCRIPTION


# Load the package's __version__.py module as a dictionary.
about = {}
if not VERSION:
  with open(VERSION_FILENAME) as fp:
    exec(fp.read(), about)

else:
  about['__version__'] = VERSION

# parse version variables and add them to command line as definitions
Version = about['__version__'].split('.')


setup(
  name                          = NAME,
  version                       = about['__version__'],
  description                   = DESCRIPTION,
  long_description              = LONG_DESCRIPTION,
  long_description_content_type = 'text/markdown',
  author                        = AUTHOR,
  author_email                  = EMAIL,
  maintainer                    = AUTHOR,
  maintainer_email              = EMAIL,
  python_requires               = REQUIRES_PYTHON,
  install_requires              = get_requires(REQUIREMENTS_FILENAME),
  url                           = URL,
  download_url                  = URL,
  keywords                      = KEYWORDS,
  packages                      = find_packages(include=['NumPyNet', 'NumPyNet.*'], exclude=('test', 'testing')),
  #include_package_data          = True, # no absolute paths are allowed
  platforms                     = 'any',
  classifiers                   =[
                                   #'License :: OSI Approved :: GPL License',
                                   'Programming Language :: Python',
                                   'Programming Language :: Python :: 3',
                                   'Programming Language :: Python :: 3.6',
                                   'Programming Language :: Python :: Implementation :: CPython',
                                   'Programming Language :: Python :: Implementation :: PyPy'
                                 ],
  license                       = 'MIT'
)
