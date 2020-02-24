#!/usr/bin/env python
# -*- coding: utf-8 -*-

from __future__ import division
from __future__ import print_function

import numpy as np
from NumPyNet.activations import Logistic
from NumPyNet.exception import LayerError
from NumPyNet.utils import check_is_fitted

__author__ = ['Mattia Ceccarelli', 'Nico Curti']
__email__ = ['mattia.ceccarelli3@studio.unibo.it', 'nico.curti2@unibo.it']

# Reference: https://github.com/experiencor/keras-yolo3/blob/master/yolo.py

class Yolo_layer(object):

  def __init__(self, input_shape, anchors, max_grid,
                     warmup_batches,
                     ignore_thresh,
                     grid_scale,
                     obj_scale, noobj_scale,
                     xywh_scale, class_scale,
                     **kwargs):
    '''
    '''
    # (grid_w, grid_h) is the equivalent of (w, h)
    # nb_box is the equivalent of N

    self.batch, self.grid_w, self.grid_h, self.c = input_shape

    self.ignore_thresh  = ignore_thresh
    self.warmup_batches = warmup_batches
    self.anchors        = np.asarray(anchors, dtype=float).reshape(1, 1, 1, 3, 2)
    self.grid_scale     = grid_scale
    self.obj_scale      = obj_scale
    self.noobj_scale    = noobj_scale
    self.xywh_scale     = xywh_scale
    self.class_scale    = class_scale

    # It seems to work only with equal grid dims
    max_grid_h, max_grid_w = max_grid

    cell_x = np.broadcast_to(np.arange(max_grid_w), shape=(max_grid_h, max_grid_w)).T.reshape(1, max_grid_h, max_grid_w, 1, 1)
    cell_y = cell_x.transpose(0, 2, 1, 3, 4)
    self.cell_grid = np.tile(np.concatenate((cell_x, cell_y), axis=-1), (batch, 1, 1, 3, 1))

    self.cost = 0.
    self.output, self.delta = (None, None)


  def __str__(self):
    return 'yolo'

  def __call__(self, previous_layer):

    if previous_layer.out_shape is None:
      class_name = self.__class__.__name__
      prev_name  = layer.__class__.__name__
      raise LayerError('Incorrect shapes found. Layer {} cannot be connected to the previous {} layer.'.format(class_name, prev_name))

    self.batch, self.grid_w, self.grid_h, self.c = previous_layer.out_shape
    return self

  @property
  def out_shape(self):
    '''
    '''
    return (self.batch, self.grid_w, self.grid_h, self.c)

  def forward(self, inpt, truth, net_shape, index):
    '''
    '''
    # NOTE: The input (probably) should be given in (c, h, w) fmt

    # y_pred is the previous output thus the input

    # adjust the shape of the y_predict [batch, grid_h, grid_w, 3, 4+1+nb_class]
    inpt = inpt.reshape(inpt.shape[:3] + (3, -1))

    # initialize the masks
    object_mask = np.expand_dims(truth[..., 4], axis=4)

    # the variable to keep track of number of batches processed
    batch_seen = 0.

    # compute grid factor and net factor
    self.grid_w, self.grid_h = truth.shape[:2]
    grid_factor = np.array([self.grid_w, self.grid_h], dtype=float).reshape((1, 1, 1, 1, 2))

    net_w, net_h = net_shape
    net_factor   = np.array([net_w, net_h], dtype=float).reshape((1, 1, 1, 1, 2))

    # Adjust prediction
    pred_box_xy    = (self.cell_grid[:, :grid_h, :grid_w, :, :] + Logistic.activate(inpt[..., :2]))
    pred_box_wh    = inpt[..., 2 : 4]
    pred_box_conf  = np.expand_dims(Logistic.activate(inpt[..., 4]), axis=4)
    pred_box_class = inpt[..., 5:]

    # Adjust ground truth
    true_box_xy    = truth[..., 0 : 2]
    true_box_wh    = truth[..., 2 : 4]
    true_box_conf  = np.expand_dims(truth[..., 4], axis=4)
    true_box_class = np.argmax(truth[..., 5:], axis=-1)

    # Compare each predicted box to all true boxes

    # initially, drag all objectness of all boxes to 0
    conf_delta = pred_box_conf - 0

    # then, ignore the boxes which have good overlap with some true box
    true_xy = true_boxes[..., 0 : 2] / grid_factor
    true_wh = true_boxes[..., 2 : 4] / net_factor

    # NOTE: probably in our configuration (x, y, w, h) we do not need to center boxes in the following way
    true_wh_half = true_wh * .5
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_xy = np.expand_dims(pred_box_xy / grid_factor, axis=4)
    pred_wh = np.expand_dims(np.exp(pred_box_wh) * self.anchors / net_factor, axis=4)

    pred_wh_half = pred_wh * .5
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half

    intersect_mins  = np.maximum(pred_mins,  true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)

    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas  = true_wh[..., 0] * true_wh[..., 1]
    pred_areas  = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores  = np.divide(intersect_areas, union_areas)

    best_ious   = np.max(iou_scores, axis=4)
    conf_delta *= np.expand_dims(best_ious < self.ignore_thresh, axis=4)

    # Compute some online statistics

    true_xy = true_box_xy / grid_factor
    true_wh = np.exp(true_box_wh) * self.anchors / net_factor

    # the same possible troubles with the centering of boxes
    true_wh_half = true_wh * .5
    true_mins    = true_xy - true_wh_half
    true_maxes   = true_xy + true_wh_half

    pred_xy = pred_box_xy / grid_factor
    pred_wh = np.exp(pred_box_wh) * self.anchors / net_factor

    pred_wh_half = pred_wh * .5
    pred_mins    = pred_xy - pred_wh_half
    pred_maxes   = pred_xy + pred_wh_half

    intersect_mins  = np.maximum(pred_mins, true_mins)
    intersect_maxes = np.minimum(pred_maxes, true_maxes)
    intersect_wh    = np.maximum(intersect_maxes - intersect_mins, 0.)
    intersect_areas = intersect_wh[..., 0] * intersect_wh[..., 1]

    true_areas = true_wh[..., 0] * true_wh[..., 1]
    pred_areas = pred_wh[..., 0] * pred_wh[..., 1]

    union_areas = pred_areas + true_areas - intersect_areas
    iou_scores = np.divide(intersect_areas, union_areas)
    iou_scores = object_mask * np.expand_dims(iou_scores, axis=4)

    count = np.sum(object_mask)
    count_noobj = np.sum(1. - object_mask)
    detect_mask = pred_box_conf * object_mask >= .5
    class_mask = np.expand_dims(pred_box_class.argmax(axis=-1) == true_box_class, axis=4)

    recall50 = np.sum( ((iou_scores >= .5)  * detect_mask * class_mask) / (count + 1e-3) )
    recall50 = np.sum( ((iou_scores >= .75) * detect_mask * class_mask) / (count + 1e-3) )
    avg_iou = np.sum(iou_scores) / (count + 1e-3)
    avg_obj = np.sum(pred_box_conf * object_mask) / (count + 1e-3)
    avg_noobj = np.sum(pred_box_conf * (1. - object_mask)) / (count_noobj + 1e-3)
    avg_cat = np.sum(object_mask * class_mask) / (count + 1e-3)

    #  Warm-up training
    batch_seen += 1

    if batch_seen < self.warmup_batches + 1:

      true_box_xy = true_box_xy + (.5 + self.cell_grid[: , :grid_h, :grid_w, :, :]) * (1 - object_mask)
      true_box_wh = true_box_wh + no.zeros_like(true_box_wh) * (1 - object_mask)
      xywh_mask   = np.ones_like(object_mask)

    else:
      # true_box_xy = true_box_xy
      # true_box_wh = true_box_wh
      xywh_mask   = object_mask

    # Compare each true box to all anchor boxes
    wh_scale = np.exp(true_box_wh) * self.anchors / net_factor
    wh_scale = np.expand_dims(2 - wh_scale[..., 0] * wh_scale[..., 1], axis=4) # the smaller the box, the bigger the scale

    xy_delta    = xywh_mask   * (pred_box_xy   - true_box_xy) * wh_scale * self.xywh_scale
    wh_delta    = xywh_mask   * (pred_box_wh   - true_box_wh) * wh_scale * self.xywh_scale
    conf_delta  = object_mask * (pred_box_conf - true_box_conf) * self.obj_scale + (1 - object_mask) * conf_delta * self.noobj_scale
    class_delta = object_mask# *   * self.class_scale # MISS (line 168)

    loss_xy    = np.sum(xy_delta * xy_delta,     axis=tuple(range(1, 5)))
    loss_wh    = np.sum(wh_delta * wh_delta,     axis=tuple(range(1, 5)))
    loss_conf  = np.sum(conf_delta * conf_delta, axis=tuple(range(1, 5)))
    loss_class = np.sum(class_delta,             axis=tuple(range(1, 5)))

    loss = loss_xy + loss_wh + loss_conf + loss_class

    print('Yolo {:d} Avg IOU: {:.3f}, Class: {:.3f}, Obj: {:.3f}, No Obj: {:.3f}, .5R: {:.3f}, .75R: {:.3f}, count: {:d}'.format(
          index, avg_iou, avg_cat, avg_obj, avg_noobj, recall50, recall75, count))

    self.cost = loss * self.grid_scale
    self.delta = np.zeros(shape=self.out_shape, dtype=float)

    return self

  def backward(self, delta):
    '''
    '''
    check_is_fitted(self, 'delta')

    delta[:] += self.delta

    return self

  def num_detections(self, thresh):
    '''
    '''
    # TODO: check it
    output = self.output.reshape((self.grid_h, self.grid_w, self.n, -1))

    return np.sum(output[..., 5:] > thresh)

  def get_detections(self, net_shape, thresh, relative):
    '''
    '''
    # it probably works only with n == 3
    output = self.output.reshape((self.grid_h, self.grid_w, self.n, -1))

    detections = []

    for i in range(self.grid_h):
      for j in range(self.grid_w):
        for k in range(self.n):

          objectness = output[i, j, k, 4]

          if objectness <= thresh: continue

          x, y, w, h = output[i, j, k, :4]

          x = (j + x) / self.grid_w
          y = (i + y) / self.grid_h
          w = np.exp(w) * self.anchors[2 * k    ] / net_w
          h = np.exp(h) * self.anchors[2 * k + 1] / net_h

          probability = self.output[i, j, k, 5:]
          probs = objectness * probability
          probability[prob <= thresh] = 0.

          dets = Detection()
          dets._objectness = objectness
          dets._box = Box(coords=(x, y, w, h))
          dets._prob = probability

          detections.append(dets)

    return detections


  def correct_boxes(self, detections, img_shape, net_shape, relative):
    '''
    '''
    net_w, net_h = net_shape
    img_w, img_h = img_shape

    if ( net_w / img_w ) < ( net_h / img_h ):
      new_w = net_w
      new_h = (img_h * net_w) / img_w
    else:
      new_h = net_h
      new_w = (img_w * net_h) / img_h

    nw_new = net_w / new_w
    nh_new = net_h / new_h


    for d in detections:

      if relative:
        d.bbox.x =      d.bbox.x - ((net_w - new_w) * .5 / net_w ) * nw_new
        d.bbox.y =      d.bbox.y - ((net_h - new_h) * .5 / net_w ) * nw_new
        d.bbox.w *= nw_new
        d.bbox.h *= nh_new

      else:
        d.bbox.x = w * (d.bbox.x - ((net_w - new_w) * .5 / net_w ) * nw_new )
        d.bbox.y = h * (d.bbox.y - ((net_h - new_h) * .5 / net_w ) * nw_new )
        d.bbox.w *= w * nw_new
        d.bbox.h *= h * nh_new


if __name__ == '__main__':

  print('Insert testing here')
