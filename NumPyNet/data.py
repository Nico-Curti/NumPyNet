#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import os
import time
import numpy as np
from glob import glob
from threading import Thread
from functools import partial

from NumPyNet.image import Image

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']


class DataGenerator (object):

  def __init__ (self, load_func, batch_size, source_path=None, source_file=None, label_path=None, label_file=None,
                      source_extension='', label_extension='', seed=123,
                      **load_func_kwargs):
    '''
    Data generator in detached thread.

    Parameters
    ----------
      load_func : function or lambda of preprocessing on a single data/label pair

      source_path :
    '''
    np.random.seed(seed)

    if source_path is None and source_file is None:
      raise ValueError('Source path and Source file can not be both null. Please give one of them')

    if source_path is not None:

      if not os.path.exists(source_path):
        raise ValueError('Source path does not exist')

      source_files = sorted(glob(source_path + '/*{}'.format(source_extension)))

    else:

      with open(source_file) as fp:
        source_files = fp.read().splitlines()

    source_files = np.asarray(source_files)

    if label_path is not None:

      if not os.path.exists(label_path):
        raise ValueError('Labels path does not exist')

      label_files = sorted(glob(label_path  + '/*{}'.format(label_extension)))
      label_files = np.asarray(label_files)

    elif label_file is not None:

      with open(label_file) as fp:
        label_files = fp.read().splitlines()

      # convert to unique numbers
      _, label_files = np.unique(label_files, return_inverse=True)
      label_files = label_files.astype(float)

    else:
      label_files = None

    self._num_data = source_files.size

    source_files, label_files = self._randomize(source_files, label_files)

    load_func = partial(load_func, **load_func_kwargs)
    self.load_func = load_func
    self._batch = batch_size

    self._thread = Thread(target=self._update, args=(source_files, label_files))
    self._thread.daemon = True

    self._current_batch = 0

    self._stopped = False
    self._data, self._label = (None, None)

  @property
  def num_data(self):
    return self._num_data

  def _randomize (self, source, label=None):
    '''
    Randomize the source and labels arrays

    Parameters
    ----------
      source : array-like
        List of source files

      label : array-like (default = None)
        List of label files

    Return
    ------
      source : array of source shuffled

      label : array of labels shuffled
    '''

    if label is not None:
      random_index = np.arange(0, self._num_data)
      np.random.shuffle(random_index)
      source = source[random_index]
      label  = label[random_index]

    else:
      np.random.shuffle(source)

    return (source, label)

  def _load (self, sources, labels):
    '''
    Map the loading function over the sources and labels
    '''

    try:
      self._data, self._label = zip(*map(self.load_func, sources, labels))

    except Exception as e:

      self._stopped = True
      raise e

    return (self._data, self._label)


  def _update (self, source_files, label_files):
    '''
    Infinite loop of batch reading.
    Each batch is read only if necessary (the previous is already used).
    '''

    start_time = time.time()
    elapsed = 1.

    while not self._stopped:

      if self._data is None:

        # we reach the end of batch
        if self._current_batch + self._batch >= self._num_data:

          source_files, label_files = self._randomize(source_files, label_files)

          self._current_batch = 0

        if label_files is not None:
          self._data, self._label = self._load(source_files[self._current_batch : self._current_batch + self._batch],
                                               label_files[self._current_batch : self._current_batch + self._batch])

        else:
          # TODO: add label loads
          self._data, self._label = self._load(source_files[self._current_batch : self._current_batch + self._batch])

        self._current_batch += self._batch

        elapsed = time.time() - start_time
        print('Elapsed {:.3f} sec.'.format(elapsed))

      else:

        time.sleep(elapsed + .05)
        start_time = time.time()


  def start (self):
    '''
    Start the thread
    '''
    self._thread.start()
    time.sleep(1.)
    return self

  def stop (self):
    '''
    Stop the thread
    '''
    self._stopped = True
    self._thread.join()

  def load_data (self):
    '''
    Get a batch of images and labels
    '''
    data, label = (self._data, self._label)
    self._data = None

    return (data, label, not self._stopped)



def load_super_resolution (hr_image_filename, patch_size=(48, 48), scale=4):
  '''
  Load Super resolution data.

  Parameters
  ----------
    hr_image_filename : string filename of the high resolution image

    patch_size : tuple of path dimension to cut

    scale : downsampling scale factor

  Notes
  -----
  In SR models the labels are given by the HR image while the input data are obtained
  from the same image after a downsampling/resizing.
  The upsample scale factor learned by the SR model will be the same used inside this
  function.
  '''

  hr_image = Image(hr_image_filename)
  w, h, _ = hr_image.shape

  patch_x, patch_y = patch_size

  dx = np.random.uniform(low=0, high=w - patch_x - 1)
  dy = np.random.uniform(low=0, high=h - patch_y - 1)

  hr_image = hr_image.crop(dsize=(dx, dy), size=patch_size)

  random_flip = np.random.uniform(low=0, high=1.)

  if random_flip >= .66:
    hr_image = hr_image.transpose()
    hr_image = hr_image.flip()

  elif random_flip >= .33:
    hr_image = hr_image.flip()

  else:
    pass

  label = hr_image
  data  = hr_image.resize(scale_factor=(scale, scale))

  return (data, label)



def load_segmentation (source_image_filename, mask_image_filename):
  '''
  Load Segmentation data.

  Parameters
  ----------
    source_image_filename : string filename of the source image

    mask_image_filename : string filename of the corresponding mask image
                          in binary format

  Notes
  -----
  In Segmentation model we have to feed the model with a simple image and
  the labels will be given by the mask (binary) of the same image in which
  the segmentation parts are highlight
  No checks are performed on the compatibility between source image and
  corresponding mask file.
  The only checks are given on the image size (channels are excluded)
  '''

  src_image  = Image(source_image_filename)
  mask_image = Image(mask_image_filename)

  if src_image.shape[:2] != mask_image.shape[:2]:
    raise ValueError('Incorrect shapes found. The source image and the corresponding mask have different sizes')

  return (src_image, mask_image)



if __name__ == '__main__':

  import pylab as plt

  train_gen = DataGenerator(load_func=load_segmentation, batch_size=2,
                            source_path='/path/to/train/images',
                            label_path='/path/to/mask/images',
                            source_extension='.png',
                            label_extension='.png'
                            )

  train_gen.start()

  fig, ((ax00, ax01), (ax10, ax11)) = plt.subplots(nrows=2, ncols=2)

  for i in range(10):

    grabbed = False

    while not grabbed:

      (data1, data2), (label1, label2), grabbed = train_gen.load_data()

    ax00.imshow(data1.get(), cmap='gray')
    ax00.axis('off')

    ax01.imshow(label1.get(), cmap='gray')
    ax01.axis('off')

    ax10.imshow(data2.get(), cmap='gray')
    ax10.axis('off')

    ax11.imshow(label2.get(), cmap='gray')
    ax11.axis('off')

    plt.pause(1e-2)

  plt.show()

  train_gen.stop()

