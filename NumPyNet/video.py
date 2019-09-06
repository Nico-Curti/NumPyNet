#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import cv2
import time
from threading import Thread

try:

  from queue import Queue

except ImportError:

  from Queue import Queue

from NumPyNet.image import Image
from NumPyNet.exception import VideoError

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']
__package__ = 'videocapture_class'


class VideoCapture (object):

  def __init__ (self, cam_index=0, queue_size=128):
    '''
    OpenCV VideoCapture wrap in detached thread.

    Parameters
    ----------
      cam_index : integer or string filename of movie

      queue_size : integer of maximum number of frame to store
                   into the queue

    Notes
    -----
    The object is inspired to the ImUtils implementation
    provided in https://github.com/jrosebr1/imutils
    '''

    self._stream = cv2.VideoCapture(cam_index)

    if self._stream is None or not self._stream.isOpened():
      raise VideoError('Can not open or find camera. Given: {}'.format(cam_index))

    self._queue = Queue(maxsize=queue_size)
    self._thread = Thread(target=self._update, args=())
    self._thread.daemon = True

    self._num_frames = 0
    self._start = None
    self._end = None

    self._stopped = False

  def start (self):
    '''
    Start the video capture in thread
    '''
    self._thread.start()
    return self

  def _update (self):
    '''
    Infinite loop of frame reading.
    Each frame is inserted into the private queue.
    '''

    self._start = time.time()

    while not self._stopped:

      if not self._queue.full():
        (grabbed, frame) = self._stream.read()

        if not grabbed:
          self._stopped = True

        else:
          self._num_frames += 1

          self._queue.put(frame)

      else:

        time.sleep(.1)

    self._stream.release()

  def read (self):
    '''
    Get a frame as Image object
    '''
    im = Image()
    return im.from_frame(self._queue.get())

  def running (self):
    '''
    Check if new frames are available
    '''

    tries = 0

    while self._queue.qsize() == 0 and not self._stopped and tries < 5:
      time.sleep(.1)
      tries += 1

    return self._queue.qsize() > 0


  def stop (self):
    '''
    Stop the thread
    '''

    self._stopped = True
    self._thread.join()
    self._end = time.time()

  @property
  def elapsed (self):
    '''
    Elapsed time from start to up to now
    '''
    return time.time() - self._start

  @property
  def fps (self):
    '''
    Frame per seconds
    '''
    return self._num_frames / self.elapsed



if __name__ == '__main__':

  cap = VideoCapture()
  time.sleep(.1)

  cv2.namedWindow('Camera', cv2.WINDOW_NORMAL)

  cap.start()

  while cap.running():

    frame = cap.read()
    frame.show('Camera', ms=1)
    print('FPS: {:.3f}'.format(cap.fps))

  cap.stop()

  cv2.destroyAllWindows()

