# Copyright 2020 The TensorFlow Authors
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#    https://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
"""Evaluator computing metrics over given pairs of predictions and labels."""

import os
import pickle
from absl import logging
import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.math.interpolation import trilinear
from tensorflow_graphics.projects.points_to_3Dobjects.models import centernet_utils
from tensorflow_graphics.projects.points_to_3Dobjects.utils import tf_utils
from google3.pyglib import gfile
from google3.third_party.google_research.google_research.tf3d.object_detection.box_utils import np_box_ops


class ShapeAccuracyMetric:
  """Computes the accuracy of shpe prediction."""

  def __init__(self, k=1):
    self.metric = tf.keras.metrics.SparseTopKCategoricalAccuracy(k)

  def update(self, sparse_labels, predicted_probabilities, sample_weights=None):
    self.metric.update_state(sparse_labels, predicted_probabilities,
                             sample_weights)

  def evaluate(self):
    return self.metric.result().numpy()

  def reset(self):
    self.metric.reset_states()


def get_2d_bounding_box_iou(box1, box2):
  """Compute IoU between two 2D bounding boxes.

  Args:
    box1: Input tensor with shape [4] [x_min, y_min, x_max, y_max]
    box2: Input tensor with shape [4] [x_min, y_min, x_max, y_max]

  Returns:
    The intersection over union as a float.
  """
  x_min1, y_min1, x_max1, y_max1 = box1
  x_min2, y_min2, x_max2, y_max2 = box2
  ma = np.maximum
  mi = np.minimum
  intersection = ma(0, mi(x_max1, x_max2) - ma(x_min1, x_min2)) * \
                 ma(0, mi(y_max1, y_max2) - ma(y_min1, y_min2))
  area1 = (x_max1 - x_min1) * (y_max1 - y_min1)
  area2 = (x_max2 - x_min2) * (y_max2 - y_min2)
  union = area1 + area2 - intersection
  print(intersection / union)
  return intersection / (union + 1e-5)


def get_3d_bounding_box_iou(box1, box2):
  """Computes intersection between two given 3d bounding boxes.

  Args:
    box1: Input tensor with shape [B, 7] where the inner dimensions are as
          follows:[x, y, z, length, width, height, yaw].
    box2: Input tensor with shape [B, 7] where the inner dimensions are as
          follows:[x, y, z, length, width, height, yaw].

  Returns:
    The IoU between the two bounding boxes.
  """
  box1 = box1.numpy() if isinstance(box1, tf.Tensor) else box1
  box2 = box2.numpy() if isinstance(box2, tf.Tensor) else box2

  box1 = box1.astype(np.float32)
  box2 = box2.astype(np.float32)

  # rotates around z, while we rotate around y so need to swap
  center_1 = tf.reshape(box1[0:3][[0, 2, 1]], [1, 3])
  center_2 = tf.reshape(box2[0:3][[0, 2, 1]], [1, 3])

  rotation_z_1 = tf.reshape(box1[-1], [1])
  rotation_z_2 = tf.reshape(box2[-1], [1])

  length_1 = tf.reshape(box1[3 + 0], [1])
  height_1 = tf.reshape(box1[3 + 2], [1])
  width_1 = tf.reshape(box1[3 + 1], [1])

  length_2 = tf.reshape(box2[3 + 0], [1])
  height_2 = tf.reshape(box2[3 + 2], [1])
  width_2 = tf.reshape(box2[3 + 1], [1])

  iou = np.squeeze(np_box_ops.iou3d_7dof_box(
      length_1, height_1, width_1, center_1, rotation_z_1,
      length_2, height_2, width_2, center_2, rotation_z_2))

  return iou


class IoUMetric:
  """IoU metric."""

  def __init__(self, max_num_classes=6, resolution=128, tol=0.05, slave=False,
               path=None):
    self.max_num_classes = max_num_classes
    self.iou_per_class = {i: [] for i in range(self.max_num_classes)}
    self.resolution = resolution
    self.slave = slave
    self.path = path
    self.tol = tol

  def update(self, labeled_sdfs, labeled_classes, labeled_poses,
             predicted_sdfs, predicted_classes, predicted_poses):
    """Update."""
    labeled_rotations = labeled_poses[0]
    labeled_translations = labeled_poses[1]
    labeled_sizes = labeled_poses[2]

    status = True
    if status:
      box_limits_x = [100, -100]
      # box_limits_y = [100, -100]
      box_limits_z = [100, -100]
      for i in range(labeled_translations.shape[0]):
        rot = tf.reshape(tf.gather(labeled_rotations[i], [0, 2, 6, 8]), [2, 2])

        min_x = tf.cast(0.0 - labeled_sizes[i][0] / 2.0, dtype=tf.float32)
        max_x = tf.cast(0.0 + labeled_sizes[i][0] / 2.0, dtype=tf.float32)
        # min_y = tf.cast(0.0 - labeled_sizes[i][1] / 2.0, dtype=tf.float32)
        # max_y = tf.cast(0.0 + labeled_sizes[i][1] / 2.0, dtype=tf.float32)
        min_z = tf.cast(0.0 - labeled_sizes[i][2] / 2.0, dtype=tf.float32)
        max_z = tf.cast(0.0 + labeled_sizes[i][2] / 2.0, dtype=tf.float32)

        translation = tf.reshape([labeled_translations[i][0],
                                  labeled_translations[i][2]], [2, 1])

        pt_0 = rot @ tf.reshape([min_x, min_z], [2, 1]) + translation
        pt_1 = rot @ tf.reshape([min_x, max_z], [2, 1]) + translation
        pt_2 = rot @ tf.reshape([max_x, min_z], [2, 1]) + translation
        pt_3 = rot @ tf.reshape([max_x, max_z], [2, 1]) + translation

        for pt in [pt_0, pt_1, pt_2, pt_3]:
          if pt[0] < box_limits_x[0]:
            box_limits_x[0] = pt[0]

          if pt[0] > box_limits_x[1]:
            box_limits_x[1] = pt[0]

          if pt[1] < box_limits_z[0]:
            box_limits_z[0] = pt[1]

          if pt[1] > box_limits_z[1]:
            box_limits_z[1] = pt[1]
      mean_x = tf.reduce_mean(box_limits_x)
      mean_z = tf.reduce_mean(box_limits_z)
    else:
      mean_x = tf.reduce_mean(labeled_translations[:, 0])
      mean_z = tf.reduce_mean(labeled_translations[:, 2])
    samples_world = grid.generate(
        (mean_x - 0.5, 0.0, mean_z - 0.5), (mean_x + 0.5, 1.0, mean_z + 0.5),
        [self.resolution, self.resolution, self.resolution])
    # samples_world = grid.generate(
    #     (box_limits_x[0][0], box_limits_y[0], box_limits_z[0][0]),
    #     (box_limits_x[1][0], box_limits_y[1], box_limits_z[1][0]),
    #     [self.resolution, self.resolution, self.resolution])
    # samples_world = grid.generate(
    #     (-5.0, -5.0, -5.0),
    #     (5.0, 5.0, 5.0),
    #     [self.resolution, self.resolution, self.resolution])
    samples_world = tf.reshape(samples_world, [-1, 3])
    ious = []

    status = False
    if status:
      _, axs = plt.subplots(labeled_translations.shape[0], 5)
      fig_obj_count = 0
    for class_id in range(self.max_num_classes):
      # Do the same for the ground truth and predictions
      sdf_values = tf.zeros_like(samples_world)[:, 0:1]
      for mtype, (classes, sdfs, poses) in enumerate([
          (labeled_classes, labeled_sdfs, labeled_poses),
          (predicted_classes, predicted_sdfs, predicted_poses)]):
        for i in range(classes.shape[0]):
          if class_id == classes[i]:
            sdf = tf.expand_dims(sdfs[i], -1)
            sdf = sdf * -1.0  # inside positive, outside zero
            samples_object = centernet_utils.transform_pointcloud(
                tf.reshape(samples_world, [1, 1, -1, 3]),
                tf.reshape(poses[2][i], [1, 1, 3]),
                tf.reshape(poses[0][i], [1, 1, 3, 3]),
                tf.reshape(poses[1][i], [1, 1, 3]), inverse=True) * 2.0
            samples_object = \
                (samples_object * (29.0/32.0) / 2.0 + 0.5) * 32.0 - 0.5
            samples = tf.squeeze(samples_object)
            interpolated = trilinear.interpolate(sdf, samples)

            sdf_values += tf.math.sign(tf.nn.relu(interpolated + self.tol))
            status2 = False
            if status2:
              a = 2
              values = interpolated
              inter = tf.reshape(values, [self.resolution,
                                          self.resolution,
                                          self.resolution])
              inter = tf.transpose(tf.reduce_max(inter, axis=a))
              im = axs[fig_obj_count, mtype * 2 + 0].matshow(inter.numpy())
              plt.colorbar(im, ax=axs[fig_obj_count, mtype * 2 + 0])
              print(mtype, fig_obj_count, 0)

              values = tf.math.sign(tf.nn.relu(interpolated + self.tol))
              inter = tf.reshape(values, [self.resolution,
                                          self.resolution,
                                          self.resolution])
              inter = tf.transpose(tf.reduce_max(inter, axis=a))
              im = axs[fig_obj_count, mtype * 2 + 1].matshow(inter.numpy())
              plt.colorbar(im, ax=axs[fig_obj_count, mtype * 2 + 1])
              print(mtype, fig_obj_count, 1)

              if mtype == 1:
                values = sdf_values
                inter = tf.reshape(values, [self.resolution,
                                            self.resolution,
                                            self.resolution])
                inter = tf.transpose(tf.reduce_max(inter, axis=a))
                im = axs[fig_obj_count, 4].matshow(inter.numpy())
                plt.colorbar(im, ax=axs[fig_obj_count, 4])
                print(mtype, fig_obj_count, 2)
                fig_obj_count += 1

      intersection = tf.reduce_sum(tf.math.sign(tf.nn.relu(sdf_values - 1)))
      union = tf.reduce_sum(tf.math.sign(sdf_values))
      iou = intersection / union
      if not tf.math.is_nan(iou):
        ious.append(iou)
      status3 = False
      if status3:
        _ = plt.figure(figsize=(5, 5))
        plt.clf()
        # mask = (sdf_values.numpy() > 0)[:, 0]
        # plt.scatter(samples_world.numpy()[mask, 0],
        #             samples_world.numpy()[mask, 1],
        #             marker='.', c=sdf_values.numpy()[mask, 0])

        plt.scatter(samples_world.numpy()[:, 0],
                    samples_world.numpy()[:, 1],
                    marker='.', c=sdf_values.numpy()[:, 0])
        plt.colorbar()
      if not tf.math.is_nan(iou):
        self.iou_per_class[class_id].append(iou)
    if ious:
      ious = [0]
    return np.mean(ious), np.min(ious)

  def evaluate(self):
    """Evaluate."""
    if self.slave:
      data = self.iou_per_class
      with gfile.Open(self.path, 'wb') as file:
        pickle.dump(data, file)
      logging.info(file)
      return
    else:
      iou_per_class_means = []
      for _, v in self.iou_per_class.items():
        if v:
          iou_per_class_means.append(np.mean(v))
      return np.mean(iou_per_class_means)

  def reset(self):
    self.iou_per_class = {i: [] for i in range(self.max_num_classes)}


class CollisionMetric:
  """Collision."""

  def __init__(self, max_num_classes=6, resolution=128,
               tol=0.04, slave=False, path=None):
    self.max_num_classes = max_num_classes
    self.collisions = []
    self.intersections = []
    self.ious = []
    self.resolution = resolution
    self.slave = slave
    self.path = path
    self.tol = tol

  def update(self, labeled_sdfs, labeled_classes, labeled_poses,
             predicted_sdfs, predicted_classes, predicted_poses):
    """Update."""
    if labeled_sdfs or labeled_classes:
      print(labeled_sdfs)
    mean_x = tf.reduce_mean(labeled_poses[1][:, 0])
    mean_z = tf.reduce_mean(labeled_poses[1][:, 2])
    samples_world = grid.generate(
        (mean_x - 0.5, 0.0, mean_z - 0.5), (mean_x + 0.5, 1.0, mean_z + 0.5),
        [self.resolution, self.resolution, self.resolution])
    samples_world = tf.reshape(samples_world, [-1, 3])

    status = False
    if status:
      _, axs = plt.subplots(3, 3)
      fig_obj_count = 0

    # Do the same for the ground truth and predictions
    num_collisions = 0
    prev_intersection = 0
    sdf_values = tf.zeros_like(samples_world)[:, 0:1]
    for classes, sdfs, poses in [(predicted_classes,
                                  predicted_sdfs,
                                  predicted_poses)]:
      for i in range(classes.shape[0]):
        sdf = tf.expand_dims(sdfs[i], -1)
        sdf = sdf * -1.0  # inside positive, outside zero
        samples_object = centernet_utils.transform_pointcloud(
            tf.reshape(samples_world, [1, 1, -1, 3]),
            tf.reshape(poses[2][i], [1, 1, 3]),
            tf.reshape(poses[0][i], [1, 1, 3, 3]),
            tf.reshape(poses[1][i], [1, 1, 3]), inverse=True) * 2.0
        samples_object = (samples_object * (29.0/32.0) / 2.0 + 0.5) * 32.0 - 0.5
        samples = tf.squeeze(samples_object)
        interpolated = trilinear.interpolate(sdf, samples)
        occupancy_value = tf.math.sign(tf.nn.relu(interpolated + self.tol))
        sdf_values += occupancy_value
        intersection = tf.reduce_sum(tf.math.sign(tf.nn.relu(sdf_values - 1)))
        if intersection > prev_intersection:
          prev_intersection = intersection
          num_collisions += 1
        status2 = False
        if status2:
          a = 1
          values = interpolated
          inter = tf.reshape(values, [self.resolution,
                                      self.resolution,
                                      self.resolution])
          inter = tf.transpose(tf.reduce_max(inter, axis=a))
          im = axs[fig_obj_count, 0].matshow(inter.numpy())
          plt.colorbar(im, ax=axs[fig_obj_count, 0])

          values = tf.math.sign(tf.nn.relu(interpolated + self.tol))
          inter = tf.reshape(values, [self.resolution,
                                      self.resolution,
                                      self.resolution])
          inter = tf.transpose(tf.reduce_max(inter, axis=a))
          im = axs[fig_obj_count, 1].matshow(inter.numpy())
          plt.colorbar(im, ax=axs[fig_obj_count, 1])

          values = sdf_values
          inter = tf.reshape(values, [self.resolution,
                                      self.resolution,
                                      self.resolution])
          inter = tf.transpose(tf.reduce_max(inter, axis=a))
          im = axs[fig_obj_count, 2].matshow(inter.numpy())
          plt.colorbar(im, ax=axs[fig_obj_count, 2])

          fig_obj_count += 1

    intersection = tf.reduce_sum(tf.math.sign(tf.nn.relu(sdf_values - 1)))
    union = tf.reduce_sum(tf.math.sign(sdf_values))
    iou = intersection / union
    self.collisions.append(num_collisions)
    self.intersections.append(intersection)
    self.ious.append(iou)
    return num_collisions, intersection, iou

  def evaluate(self):
    """Evaluate."""
    if self.slave:
      data = {'collisions': self.collisions,
              'intersections': self.intersections,
              'ious': self.ious}
      with gfile.Open(self.path, 'wb') as file:
        pickle.dump(data, file)
      logging.info(file)
      return
    else:
      # self.collisions = []
      # for k, v in self.iou_per_class.items():
      #   if len(v) > 0:
      #     iou_per_class_means.append(np.mean(v))
      return np.sum(self.collisions)

  def reset(self):
    self.intersections = []
    self.ious = []
    self.collisions = []


class BoxIoUMetric:
  """BoxIOU."""

  def __init__(self, t=0.5, threed=False):
    self.labeled_boxes = {}
    self.predicted_boxes = {}
    self.threshold = t
    self.threed = threed
    self.get_iou_func = get_2d_bounding_box_iou
    if self.threed:
      self.get_iou_func = get_3d_bounding_box_iou

  def update(self, scene_id, labeled_boxes, labeled_classes, predicted_boxes,
             predicted_classes, confidences):
    """For one scene, provide all ground-truth and all predicted detections."""
    self.labeled_boxes[scene_id] = (labeled_boxes, labeled_classes)
    self.predicted_boxes[scene_id] = (predicted_boxes, predicted_classes,
                                      confidences)

  def evaluate(self):
    """Eval."""
    predictions_per_class = {}  # map {classname: pred}
    labels_per_class = {}  # map {classname: gt}

    for scene_id in self.predicted_boxes:
      bboxes, classnames, scores = self.predicted_boxes[scene_id]
      classnames = classnames.numpy()
      bboxes = bboxes.numpy()
      scores = scores.numpy()
      for i in range(classnames.shape[0]):
        classname = classnames[i]
        bbox = bboxes[i]
        score = scores[i]
        # for classname, bbox, score in self.predicted_boxes[scene_id]:
        if classname not in predictions_per_class:
          predictions_per_class[classname] = {}
        if scene_id not in predictions_per_class[classname]:
          predictions_per_class[classname][scene_id] = []
        if classname not in labels_per_class:
          labels_per_class[classname] = {}
        if scene_id not in labels_per_class[classname]:
          labels_per_class[classname][scene_id] = []
        predictions_per_class[classname][scene_id].append((bbox, score))

    for scene_id in self.labeled_boxes:
      bboxes, classnames = self.labeled_boxes[scene_id]
      classnames = classnames.numpy()
      bboxes = bboxes.numpy()
      for i in range(classnames.shape[0]):
        classname = classnames[i]
        bbox = bboxes[i]
        if classname not in labels_per_class:
          labels_per_class[classname] = {}
        if scene_id not in labels_per_class[classname]:
          labels_per_class[classname][scene_id] = []
        labels_per_class[classname][scene_id].append(bbox)

    recall_per_class = {}
    precision_per_class = {}
    ap_per_class = {}
    for classname in labels_per_class:
      print('Computing AP for class: ', classname)
      if classname in predictions_per_class:
        recall, precision, ap = self._eval_detections_per_class(
            # this does not work when class was never predicted
            predictions_per_class[classname],
            labels_per_class[classname],
            self.threshold)
      else:
        recall, precision, ap = 0.0, 0.0, 0.0
      recall_per_class[classname] = recall
      precision_per_class[classname] = precision
      ap_per_class[classname] = ap
      print(classname, ap)
    # return recall_per_class, precision_per_class, ap_per_class
    mean = np.mean(np.array([v for k, v in ap_per_class.items()]))
    print(mean)
    return mean

  def _get_iou_main(self, get_iou_func, args):
    return get_iou_func(*args)

  def _voc_ap(self, rec, prec):
    """Compute VOC AP given precision and recall."""
    mrec = np.concatenate(([0.], rec, [1.]))
    mpre = np.concatenate(([0.], prec, [0.]))
    # compute the precision envelope
    for i in range(mpre.size - 1, 0, -1):
      mpre[i - 1] = np.maximum(mpre[i - 1], mpre[i])
    # to calculate area under PR curve, look for points
    # where X axis (recall) changes value
    i = np.where(mrec[1:] != mrec[:-1])[0]
    # and sum (\Delta recall) * prec
    return np.sum((mrec[i + 1] - mrec[i]) * mpre[i + 1])

  def _eval_detections_per_class(self, pred, gt, ovthresh=0.25):
    """Generic functions to compute precision/recall for object detection."""

    # construct gt objects
    class_recs = {}  # {img_id: {'bbox': bbox list, 'det': matched list}}
    npos = 0
    for img_id in gt.keys():
      bbox = np.array(gt[img_id])
      det = [False] * len(bbox)
      npos += len(bbox)
      class_recs[img_id] = {'bbox': bbox, 'det': det}
    # pad empty list to all other imgids
    for img_id in pred:
      if img_id not in gt:
        class_recs[img_id] = {'bbox': np.array([]), 'det': []}

    # construct dets
    image_ids = []
    confidence = []
    bb = []
    for img_id in pred:
      for box, score in pred[img_id]:
        image_ids.append(img_id)
        confidence.append(score)
        bb.append(box)
    confidence = np.array(confidence)
    bb = np.array(bb)  # (nd,4 or 8,3 or 6)

    # sort by confidence
    sorted_ind = np.argsort(-confidence)
    bb = bb[sorted_ind, ...]
    image_ids = [image_ids[x] for x in sorted_ind]

    # go down dets and mark TPs and FPs
    nd = len(image_ids)
    tp = np.zeros(nd)
    fp = np.zeros(nd)
    for d in range(nd):
      r = class_recs[image_ids[d]]
      bb = bb[d, ...].astype(float)
      ovmax = -np.inf
      bbgt = r['bbox'].astype(float)

      if bbgt.size > 0:
        # compute overlaps
        for j in range(bbgt.shape[0]):
          iou = self._get_iou_main(self.get_iou_func, (bb, bbgt[j, ...]))
          if iou > ovmax:
            ovmax = iou
            jmax = j

      if ovmax > ovthresh:
        if not r['det'][jmax]:
          tp[d] = 1.
          r['det'][jmax] = 1
        else:
          fp[d] = 1.
      else:
        fp[d] = 1.

    # compute precision recall
    fp = np.cumsum(fp)
    tp = np.cumsum(tp)
    rec = tp / float(npos + 1e-5)
    # avoid divide by zero in case the first detection matches a difficult
    # ground truth
    prec = tp / np.maximum(tp + fp, np.finfo(np.float64).eps)
    ap = self._voc_ap(rec, prec)

    return rec, prec, ap

  def reset(self):
    self.labeled_boxes = {}
    self.predicted_boxes = {}


class Evaluator:
  """Evaluator for specified metrics."""

  def __init__(self, metrics, split, shapenet_dir):
    self.metrics = metrics
    self.split = split
    self.shapenet_dir = shapenet_dir

  def add_detections(self, sample, detections):
    """Add detections to evaluation.

    Args:
      sample: the ground truth information
      detections: the predicted detections

    Returns:
      dict of intermediate results.

    """
    result_dict = {'iou_mean': -1, 'iou_min': -1, 'collisions': 0,
                   'collision_intersection': 0, 'collision_iou': 0}
    num_boxes = sample['num_boxes'].numpy()

    for _, metric in self.metrics.items():
      if isinstance(metric, ShapeAccuracyMetric):
        labels = sample['shapes']
        weights = tf.math.sign(labels + 1)  # -1 is mapped to zero, else 1
        metric.update(labels, detections['shapes_logits'], weights)
      elif isinstance(metric, BoxIoUMetric):
        scene_id = str(sample['scene_filename'].numpy(), 'utf-8')

        # Get ground truth boxes
        labeled_boxes = tf.gather(
            sample['groundtruth_boxes'], axis=1, indices=[1, 0, 3, 2]) * 256.0
        if metric.threed:
          rotations_y = tf.concat([tf_utils.euler_from_rotation_matrix(
              tf.reshape(detections['rotations_3d'][i], [3, 3]),
              1) for i in range(num_boxes)], axis=0)
          rotations_y = tf.reshape(rotations_y, [-1, 1])
          labeled_boxes = tf.concat([sample['translations_3d'],
                                     sample['sizes_3d'],
                                     rotations_y], axis=1)

        # Get predicted boxes
        predicted_boxes = detections['detection_boxes']
        if metric.threed:
          rotations_y = tf.concat([tf_utils.euler_from_rotation_matrix(
              tf.reshape(detections['rotations_3d'][i], [3, 3]),
              1) for i in range(num_boxes)], axis=0)
          rotations_y = tf.reshape(rotations_y, [-1, 1])
          predicted_boxes = tf.concat([detections['translations_3d'],
                                       detections['sizes_3d'],
                                       rotations_y], axis=1)

        labeled_classes = tf.cast(sample['groundtruth_valid_classes'], tf.int64)
        predicted_classes = tf.cast(detections['detection_classes'], tf.int64)
        confidences = detections['detection_scores']
        metric.update(scene_id, labeled_boxes, labeled_classes, predicted_boxes,
                      predicted_classes, confidences)
      elif isinstance(metric, IoUMetric):
        classes = sample['classes']
        mesh_names = sample['mesh_names']
        labeled_sdfs = []
        for i in range(num_boxes):
          class_id = str(classes[i].numpy()).zfill(8)
          model_name = str(mesh_names[i].numpy(), 'utf-8')
          path_prefix = os.path.join(self.shapenet_dir, class_id, model_name)
          file_sdf = os.path.join(path_prefix, 'model_normalized_sdf.npy')
          with gfile.Open(file_sdf, 'rb') as f:
            labeled_sdfs.append(tf.expand_dims(np.load(f).astype(np.float32),
                                               0))
        labeled_sdfs = tf.concat(labeled_sdfs, axis=0)

        labeled_classes = tf.cast(sample['groundtruth_valid_classes'], tf.int64)
        labeled_permutation = np.argsort(labeled_classes)

        labeled_sdfs = labeled_sdfs.numpy()[labeled_permutation]
        labeled_classes = labeled_classes.numpy()[labeled_permutation]
        labeled_rotations_3d = sample['rotations_3d'].numpy()
        labeled_rotations_3d = labeled_rotations_3d[labeled_permutation]
        labeled_translations_3d = sample['translations_3d'].numpy()
        labeled_translations_3d = labeled_translations_3d[labeled_permutation]
        labeled_sizes_3d = sample['sizes_3d'].numpy()[labeled_permutation]
        labeled_poses = (labeled_rotations_3d, labeled_translations_3d,
                         labeled_sizes_3d)

        # Predictions
        predicted_classes = tf.cast(detections['detection_classes'], tf.int64)
        predicted_permutation = np.argsort(predicted_classes)
        predicted_classes = predicted_classes.numpy()[predicted_permutation]

        predicted_sdfs = \
          detections['predicted_sdfs'].numpy()[predicted_permutation]
        predicted_rotations_3d = \
          detections['rotations_3d'].numpy()[predicted_permutation]
        predicted_translations_3d = \
          detections['translations_3d'].numpy()[predicted_permutation]
        predicted_sizes_3d = \
          detections['sizes_3d'].numpy()[predicted_permutation]
        predicted_poses = (predicted_rotations_3d, predicted_translations_3d,
                           predicted_sizes_3d)

        full_oracle = False
        if full_oracle:
          predicted_sdfs = detections['groundtruth_sdfs'].numpy()
          predicted_sdfs = predicted_sdfs[labeled_permutation]
          predicted_classes = labeled_classes
          predicted_poses = labeled_poses

        print('----------------------------')
        print(predicted_sdfs.shape)
        print(predicted_classes.shape)
        print(predicted_poses[0].shape)
        print(predicted_poses[1].shape)
        print(predicted_poses[2].shape)

        pose_oracle = False
        if pose_oracle:
          predicted_sdfs = detections['predicted_sdfs'].numpy()
          predicted_sdfs = predicted_sdfs[predicted_permutation]
          predicted_poses = (labeled_rotations_3d, labeled_translations_3d,
                             labeled_sizes_3d)

        class_oracle = True
        if class_oracle:
          predicted_classes *= 0
          labeled_classes *= 0

        iou_mean, iou_min = metric.update(
            labeled_sdfs, labeled_classes, labeled_poses, predicted_sdfs,
            predicted_classes, predicted_poses, sample['dot'])
        result_dict['iou_mean'] = iou_mean
        result_dict['iou_min'] = iou_min
      elif isinstance(metric, CollisionMetric):

        labeled_sdfs = detections['groundtruth_sdfs']
        labeled_classes = tf.cast(sample['groundtruth_valid_classes'], tf.int64)
        labeled_poses = (sample['rotations_3d'],
                         sample['translations_3d'],
                         sample['sizes_3d'])

        predicted_classes = tf.cast(detections['detection_classes'], tf.int64)
        predicted_sdfs = detections['predicted_sdfs']
        predicted_poses = (detections['rotations_3d'],
                           detections['translations_3d'],
                           detections['sizes_3d'])

        full_oracle = False
        if full_oracle:
          predicted_sdfs = detections['groundtruth_sdfs'].numpy()
          predicted_classes = labeled_classes
          predicted_poses = labeled_poses

        num_collisions, intersection, iou = metric.update(
            labeled_sdfs, labeled_classes, labeled_poses,
            predicted_sdfs, predicted_classes, predicted_poses)
        result_dict['collisions'] = num_collisions
        result_dict['collision_intersection'] = intersection
        result_dict['collision_iou'] = iou

    return result_dict

  def evaluate(self):
    """Runs metrics over provided pairs and returns metric dict."""
    metrics = {}
    for name, metric in self.metrics.items():
      metrics[name] = metric.evaluate()
    return metrics

  def reset_metrics(self):
    for _, metric in self.metrics.items():
      metric.reset()
