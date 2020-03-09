# !/usr/bin/env python3
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import tensorflow as tf
from tensorflow.keras.layers import Layer
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input
import tensorflow.keras.backend as K

from NumPyNet.exception import LayerError
from NumPyNet.exception import NotFittedError
from NumPyNet.layers.yolo_layer import Yolo_layer

import numpy as np
import pytest
from hypothesis import strategies as st
from hypothesis import given
from hypothesis import settings

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

# Tensorflow version of Yolo layer
# Reference: https://github.com/experiencor/keras-yolo3/blob/master/yolo.py

class YoloLayer(Layer):

    def __init__(self, anchors, max_grid, batch_size, warmup_batches, ignore_thresh,
                    grid_scale, obj_scale, noobj_scale, xywh_scale, class_scale,
                    **kwargs):
        # make the model settings persistent

        self.ignore_thresh  = ignore_thresh
        self.warmup_batches = warmup_batches
        self.anchors        = tf.constant(anchors, dtype='float', shape=[1,1,1,3,2])
        self.grid_scale     = grid_scale
        self.obj_scale      = obj_scale
        self.noobj_scale    = noobj_scale
        self.xywh_scale     = xywh_scale
        self.class_scale    = class_scale

        # make a persistent mesh grid
        max_grid_h, max_grid_w = max_grid

        cell_x = tf.to_float(tf.reshape(tf.tile(tf.range(max_grid_w), [max_grid_h]), (1, max_grid_h, max_grid_w, 1, 1)))
        cell_y = tf.transpose(cell_x, (0, 2, 1, 3, 4))
        self.cell_grid = tf.tile(tf.concat([cell_x, cell_y], axis=-1), [batch_size, 1, 1, 3, 1])

        super(YoloLayer, self).__init__(**kwargs)

    def build(self, input_shape):
      super(YoloLayer, self).build(input_shape)

    def call(self, x):

      input_image, y_pred, y_true, true_boxes = x

      # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
      # y_pred = tf.reshape(y_pred, tf.concat([tf.shape(y_pred)[:3], tf.constant([3, -1])], axis=0))

      # initialize the masks
      object_mask     = tf.expand_dims(y_true[..., 4], 4)

      # the variable to keep track of number of batches processed
      batch_seen = tf.Variable(0.)

      # compute grid factor and net factor
      grid_h      = tf.shape(y_true)[1]
      grid_w      = tf.shape(y_true)[2]
      grid_factor = tf.reshape(tf.cast([grid_w, grid_h], tf.float32), [1,1,1,1,2])

      net_h       = tf.shape(input_image)[1]
      net_w       = tf.shape(input_image)[2]
      net_factor  = tf.reshape(tf.cast([net_w, net_h], tf.float32), [1,1,1,1,2])

      """
      Adjust prediction
      """
      pred_box_xy    = (self.cell_grid[:,:grid_h,:grid_w,:,:] + tf.sigmoid(y_pred[..., :2]))  # sigma(t_xy) + c_xy
      pred_box_wh    = y_pred[..., 2:4]                                                       # t_wh
      pred_box_conf  = tf.expand_dims(tf.sigmoid(y_pred[..., 4]), 4)                          # adjust confidence
      pred_box_class = y_pred[..., 5:]

      """
      Adjust ground truth
      """
      true_box_xy    = y_true[..., 0:2] # (sigma(t_xy) + c_xy)
      true_box_wh    = y_true[..., 2:4] # t_wh
      true_box_conf  = tf.expand_dims(y_true[..., 4], 4)
      true_box_class = tf.argmax(y_true[..., 5:], -1)

      """
      Compare each predicted box to all true boxes
      """
      # initially, drag all objectness of all boxes to 0
      conf_delta  = pred_box_conf - 0

      # then, ignore the boxes which have good overlap with some true box
      true_xy = true_boxes[..., 0:2] / grid_factor
      true_wh = true_boxes[..., 2:4] / net_factor

      true_wh_half = true_wh / 2.
      true_mins    = true_xy - true_wh_half
      true_maxes   = true_xy + true_wh_half

      pred_xy = tf.expand_dims(pred_box_xy / grid_factor, 4)
      pred_wh = tf.expand_dims(tf.exp(pred_box_wh) * self.anchors / net_factor, 4)

      pred_wh_half = pred_wh / 2.
      pred_mins    = pred_xy - pred_wh_half
      pred_maxes   = pred_xy + pred_wh_half

      intersect_mins  = tf.maximum(pred_mins,  true_mins)
      intersect_maxes = tf.minimum(pred_maxes, true_maxes)

      intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
      intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

      true_areas = true_wh[..., 0] * true_wh[..., 1]
      pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

      union_areas = pred_areas + true_areas - intersect_areas
      iou_scores  = tf.truediv(intersect_areas, union_areas)

      best_ious   = tf.reduce_max(iou_scores, axis=4)
      conf_delta *= tf.expand_dims(tf.to_float(best_ious < self.ignore_thresh), 4)

      """
      Compute some online statistics
      """
      true_xy = true_box_xy / grid_factor
      true_wh = tf.exp(true_box_wh) * self.anchors / net_factor

      true_wh_half = true_wh / 2.
      true_mins    = true_xy - true_wh_half
      true_maxes   = true_xy + true_wh_half

      pred_xy = pred_box_xy / grid_factor
      pred_wh = tf.exp(pred_box_wh) * self.anchors / net_factor

      pred_wh_half = pred_wh / 2.
      pred_mins    = pred_xy - pred_wh_half
      pred_maxes   = pred_xy + pred_wh_half

      intersect_mins  = tf.maximum(pred_mins,  true_mins)
      intersect_maxes = tf.minimum(pred_maxes, true_maxes)
      intersect_wh    = tf.maximum(intersect_maxes - intersect_mins, 0.)
      intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

      true_areas = true_wh[..., 0] * true_wh[..., 1]
      pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

      union_areas = pred_areas + true_areas - intersect_areas
      iou_scores  = tf.truediv(intersect_areas, union_areas)
      iou_scores  = object_mask * tf.expand_dims(iou_scores, 4)

      count       = tf.reduce_sum(object_mask)
      count_noobj = tf.reduce_sum(1 - object_mask)
      detect_mask = tf.to_float((pred_box_conf*object_mask) >= 0.5)
      class_mask  = tf.expand_dims(tf.to_float(tf.equal(tf.argmax(pred_box_class, -1), true_box_class)), 4)
      recall50    = tf.reduce_sum(tf.to_float(iou_scores >= 0.5 ) * detect_mask  * class_mask) / (count + 1e-3)
      recall75    = tf.reduce_sum(tf.to_float(iou_scores >= 0.75) * detect_mask  * class_mask) / (count + 1e-3)
      avg_iou     = tf.reduce_sum(iou_scores) / (count + 1e-3)
      avg_obj     = tf.reduce_sum(pred_box_conf  * object_mask)  / (count + 1e-3)
      avg_noobj   = tf.reduce_sum(pred_box_conf  * (1-object_mask))  / (count_noobj + 1e-3)
      avg_cat     = tf.reduce_sum(object_mask * class_mask) / (count + 1e-3)

      """
      Warm-up training
      """
      batch_seen = tf.assign_add(batch_seen, 1.)

      true_box_xy, true_box_wh, xywh_mask = tf.cond(tf.less(batch_seen, self.warmup_batches+1),
                            lambda: [true_box_xy + (0.5 + self.cell_grid[:,:grid_h,:grid_w,:,:]) * (1-object_mask),
                                     true_box_wh + tf.zeros_like(true_box_wh) * (1-object_mask),
                                     tf.ones_like(object_mask)],
                            lambda: [true_box_xy,
                                     true_box_wh,
                                     object_mask])

      """
      Compare each true box to all anchor boxes
      """
      wh_scale = tf.exp(true_box_wh) * self.anchors / net_factor
      wh_scale = tf.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

      xy_delta    = xywh_mask   * (pred_box_xy-true_box_xy) * wh_scale * self.xywh_scale
      wh_delta    = xywh_mask   * (pred_box_wh-true_box_wh) * wh_scale * self.xywh_scale
      conf_delta  = object_mask * (pred_box_conf-true_box_conf) * self.obj_scale + (1-object_mask) * conf_delta * self.noobj_scale
      class_delta = object_mask * \
                    tf.expand_dims(tf.nn.sparse_softmax_cross_entropy_with_logits(labels=true_box_class, logits=pred_box_class), 4) * \
                    self.class_scale

      loss_xy    = tf.reduce_sum(tf.square(xy_delta),       list(range(1,5)))
      loss_wh    = tf.reduce_sum(tf.square(wh_delta),       list(range(1,5)))
      loss_conf  = tf.reduce_sum(tf.square(conf_delta),     list(range(1,5)))
      loss_class = tf.reduce_sum(class_delta,               list(range(1,5)))

      loss = loss_xy + loss_wh + loss_conf + loss_class

      # if debug:
      if False:
          loss = tf.Print(loss, [grid_h, avg_obj], message='avg_obj \t\t', summarize=1000)
          loss = tf.Print(loss, [grid_h, avg_noobj], message='avg_noobj \t\t', summarize=1000)
          loss = tf.Print(loss, [grid_h, avg_iou], message='avg_iou \t\t', summarize=1000)
          loss = tf.Print(loss, [grid_h, avg_cat], message='avg_cat \t\t', summarize=1000)
          loss = tf.Print(loss, [grid_h, recall50], message='recall50 \t', summarize=1000)
          loss = tf.Print(loss, [grid_h, recall75], message='recall75 \t', summarize=1000)
          loss = tf.Print(loss, [grid_h, count], message='count \t', summarize=1000)
          loss = tf.Print(loss, [grid_h, tf.reduce_sum(loss_xy),
                                      tf.reduce_sum(loss_wh),
                                      tf.reduce_sum(loss_conf),
                                      tf.reduce_sum(loss_class)],  message='loss xy, wh, conf, class: \t',   summarize=1000)


      return loss*self.grid_scale

class TestYoloLayer:


  def test_forward (self):

    sess = tf.compat.v1.Session()

    classes = 20
    anchors = [1,2,3,4,5,6]
    ignore_thresh = 0.9
    warmup_batches = 3
    grid_scale = 0.3
    obj_scale = 0.7
    noobj_scale = 0.5
    xywh_scale = 3.1
    class_scale = 1.3
    max_grid = (11,11)
    batch_size = 5
    max_box = 15

    grid_h = 7
    grid_w = 7

    n_anchors = 1 # If changed broadcast problems

    input_image = np.random.uniform(size=(batch_size, 100, 101, 3)).astype(np.float32)
    y_pred      = np.random.uniform(size=(batch_size, grid_h, grid_w, n_anchors, 5*2 + classes)).astype(np.float32)
    y_true      = np.random.uniform(size=(batch_size, grid_h, grid_w, n_anchors, 5*2 + classes)).astype(np.float32)
    true_boxes  = np.random.uniform(size=(1, 1, 1, max_box, 4)).astype(np.float32)

    x = [input_image, y_pred, y_true, true_boxes]

    tf_yolo = YoloLayer(anchors=anchors, max_grid=max_grid, batch_size=batch_size, warmup_batches=warmup_batches, ignore_thresh=ignore_thresh,
                    grid_scale=grid_scale, obj_scale=obj_scale, noobj_scale=noobj_scale, xywh_scale=xywh_scale, class_scale=class_scale)
    tf_yolo.build((batch_size, 100, 101, 3))

    np_yolo = Yolo_layer(input_shape=(batch_size, 100, 101, 3), anchors=anchors, max_grid=max_grid, warmup_batches=warmup_batches, ignore_thresh=ignore_thresh,
                    grid_scale=grid_scale, obj_scale=obj_scale, noobj_scale=noobj_scale, xywh_scale=xywh_scale, class_scale=class_scale)

    out_np = np_yolo.forward(y_pred, y_true, (100, 101), true_boxes)
    out_tf = tf_yolo.call(x)

    init = tf.global_variables_initializer()
    sess.run(init)

    with sess.as_default():
      outtf = out_tf.eval()

    assert np.allclose(out_np.cost, outtf, rtol=1e-1)

  def test_backwad (self):
    pass


if __name__ == '__main__':

  test = TestYoloLayer()

  test.test_forward()
