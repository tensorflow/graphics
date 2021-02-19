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
# python3
"""Targets to use in the loss."""

# Functions: _smallest_positive_root, max_distance_for_overlap and,
# assign_center_targets taken from
# /google3/third_party/tensorflow_models/object_detection/core/target_assigner.py
# """
import tensorflow as tf

from google3.third_party.tensorflow_models.object_detection.core import box_list
from google3.third_party.tensorflow_models.object_detection.core import box_list_ops


def _smallest_positive_root(a, b, c):
  """Returns the smallest positive root of a quadratic equation."""

  determinant = tf.sqrt(b**2 - 4 * a * c)

  # CenterNet implementation. The commented lines implement the fixed version
  # in https://github.com/princeton-vl/CornerNet. Change the implementation
  # after verifying it has no negative impact.
  # root1 = (-b - determinant) / (2 * a)
  # root2 = (-b + determinant) / (2 * a)

  # return tf.where(tf.less(root1, 0), root2, root1)

  return (-b + determinant) / 2.0


def _max_distance_for_overlap(height, width, min_iou=0.7):
  """Computes how far apart bbox corners can lie while maintaining the iou.

  Given a bounding box size, this function returns a lower bound on how far
  apart the corners of another box can lie while still maintaining the given
  IoU. The implementation is based on the `gaussian_radius` function in the
  Objects as Points github repo: https://github.com/xingyizhou/CenterNet

  Args:
    height: A 1-D float Tensor representing height of the ground truth boxes.
    width: A 1-D float Tensor representing width of the ground truth boxes.
    min_iou: A float representing the minimum IoU desired.

  Returns:
   distance: A 1-D Tensor of distances, of the same length as the input
     height and width tensors.
  """
  height, width = tf.math.ceil(height), tf.math.ceil(width)

  distance_detection_offset = _smallest_positive_root(
      a=1,
      b=-(height + width),
      c=width * height * ((1 - min_iou) / (1 + min_iou)))

  distance_detection_in_gt = _smallest_positive_root(
      a=4, b=-2 * (height + width), c=(1 - min_iou) * width * height)

  distance_gt_in_detection = _smallest_positive_root(
      a=4 * min_iou,
      b=(2 * min_iou) * (width + height),
      c=(min_iou - 1) * width * height)

  return tf.reduce_min([
      distance_detection_offset, distance_gt_in_detection,
      distance_detection_in_gt
  ],
                       axis=0)


def valid_bounding_boxes_mask(boxes, height, width, tolerance_pixels=1.0):
  boxes_height = (boxes[:, 2] - boxes[:, 0]) * height
  boxes_width = (boxes[:, 3] - boxes[:, 1]) * width
  invalid_boxes = tf.math.logical_or(
      tf.math.less_equal(boxes_height, tolerance_pixels),
      tf.math.less_equal(boxes_width, tolerance_pixels))
  valid_boxes_mask = tf.cast(tf.math.logical_not(invalid_boxes), tf.float32)
  return valid_boxes_mask


def assign_center_targets(gt_box_batch,
                          gt_class_batch,
                          gt_num_boxes_batch,
                          image_size,
                          stride,
                          num_classes,
                          gt_weights_batch=None,
                          min_overlap=0.7):
  """Returns the center heatmap target for the CenterNet model.

  Args:
    gt_box_batch: A list of BoxList objects representing the ground truth
      detection bounding boxes for each sample in the batch.
    gt_class_batch: A tensors representing with the class label for each box in
      `gt_box_batch`.
    gt_num_boxes_batch: A tensors with the number of valid boxes in
      `gt_box_batch`.
    image_size: int tuple (height, width) of input to the CenterNet model. This
      is used to determine the height of the output.
    stride: int, ratio between the input and output size in the network.
    num_classes: int, total number of classes.
    gt_weights_batch: A list of tensors corresponding to the weight of each
      ground truth detection box.
    min_overlap: A float representing the minimum IoU desired.

  Returns:
    heatmap: A Tensor of size [batch_size, output_height, output_width,
      num_classes] representing the per class center heatmap. output_height
      and output_width are computed by dividing the input height and width by
      the stride.
  """
  height, width = image_size

  if gt_weights_batch:
    print(gt_weights_batch)

  x_range = tf.range(width // stride, dtype=tf.float32)
  y_range = tf.range(height // stride, dtype=tf.float32)
  x_grid, y_grid = tf.meshgrid(x_range, y_range, indexing='xy')

  x_grid = tf.expand_dims(x_grid, 2)
  y_grid = tf.expand_dims(y_grid, 2)

  def _create_heatmap(elems):
    gt_boxes = elems[0]
    gt_num_boxes = elems[1]
    gt_classes = elems[2]

    num_boxes = tf.cast(gt_num_boxes, tf.int32)
    boxes = box_list.BoxList(gt_boxes[:num_boxes, ...])
    class_targets = tf.one_hot(gt_classes[:num_boxes, ...], num_classes)
    boxes = box_list_ops.to_absolute_coordinates(boxes, height // stride,
                                                 width // stride)

    (y_center, x_center, boxes_height, boxes_width) = \
      boxes.get_center_coordinates_and_sizes()

    # The raw center coordinates in the output space.
    x_diff = x_grid - tf.math.floor(x_center)
    y_diff = y_grid - tf.math.floor(y_center)
    squared_distance = x_diff**2 + y_diff**2

    # We are dividing by 3 so that points closer than the computed
    # distance have a >99% CDF. In Pytorch, sigma = (2 * sigma + 1) / 6
    sigma = _max_distance_for_overlap(boxes_height, boxes_width, min_overlap)
    sigma = (2 * tf.math.maximum(tf.math.floor(sigma), 0.0) + 1) / 6

    gaussian_map = tf.exp(-squared_distance / (2 * sigma * sigma))

    valid_boxes_mask = valid_bounding_boxes_mask(gt_boxes[:num_boxes, ...],
                                                 height // stride,
                                                 width // stride)
    gaussian_map = gaussian_map * valid_boxes_mask

    output_height = tf.shape(gaussian_map)[0]
    output_width = tf.shape(gaussian_map)[1]
    num_boxes = tf.shape(gaussian_map)[2]

    reshaped_gaussian_map = tf.reshape(
        gaussian_map, (output_height, output_width, num_boxes, 1))
    reshaped_class_targets = \
        tf.reshape(class_targets, (1, 1, num_boxes, num_classes))
    gaussian_per_box_per_class_map = (
        reshaped_gaussian_map * reshaped_class_targets)
    heatmap = tf.reduce_max(gaussian_per_box_per_class_map, axis=2)
    return heatmap

  return tf.map_fn(
      _create_heatmap, (gt_box_batch, gt_num_boxes_batch, gt_class_batch),
      dtype=tf.float32)


def assign_offset_targets(gt_box_batch, gt_num_boxes_batch, stride, image_size):
  """Assign offset."""
  # Returns offsets x,y
  height, width = image_size

  def _create_offsets(elems):
    gt_boxes = elems[0]
    gt_num_boxes = elems[1]

    num_boxes = tf.cast(gt_num_boxes, tf.int32)
    if tf.math.equal(num_boxes, 0):
      return tf.zeros((tf.shape(gt_boxes)[0], 2), dtype=tf.float32)
    boxes = box_list.BoxList(gt_boxes[:num_boxes, ...])
    boxes = box_list_ops.to_absolute_coordinates(boxes, height // stride,
                                                 width // stride)
    (y_center, x_center, _, _) = boxes.get_center_coordinates_and_sizes()
    center = tf.stack([x_center, y_center], axis=1)
    center_floor = tf.math.floor(center)
    num_padding_boxes = tf.shape(gt_boxes)[0] - tf.shape(center_floor)[0]
    return tf.pad(center - center_floor, ((0, num_padding_boxes), (0, 0)))

  return tf.map_fn(
      _create_offsets, (gt_box_batch, gt_num_boxes_batch), dtype=tf.float32)


def assign_width_height_targets(gt_box_batch, gt_num_boxes_batch, stride,
                                image_size):
  """Assign width height."""
  # Returns width, height
  height, width = image_size

  def _create_width_height(elems):
    gt_boxes = elems[0]
    gt_num_boxes = elems[1]

    num_boxes = tf.cast(gt_num_boxes, tf.int32)
    if tf.math.equal(num_boxes, 0):
      return tf.zeros((tf.shape(gt_boxes)[0], 2), dtype=tf.float32)
    boxes = box_list.BoxList(gt_boxes[:num_boxes, ...])
    boxes = box_list_ops.to_absolute_coordinates(boxes, height // stride,
                                                 width // stride)
    (_, _, boxes_height, boxes_width) = boxes.get_center_coordinates_and_sizes()
    width_height = tf.stack([boxes_width, boxes_height], axis=1)
    num_padding_boxes = tf.shape(gt_boxes)[0] - tf.shape(width_height)[0]
    return tf.pad(width_height, ((0, num_padding_boxes), (0, 0)))

  return tf.map_fn(
      _create_width_height, (gt_box_batch, gt_num_boxes_batch),
      dtype=tf.float32)


def assign_center_indices_targets(gt_box_batch, gt_num_boxes_batch, stride,
                                  image_size):
  """Assing center."""
  height, width = image_size

  def _create_center_indices(elems):
    gt_boxes = elems[0]
    gt_num_boxes = elems[1]

    num_boxes = tf.cast(gt_num_boxes, tf.int32)
    if tf.math.equal(num_boxes, 0):
      return tf.zeros((tf.shape(gt_boxes)[0]), dtype=tf.int64)
    boxes = box_list.BoxList(gt_boxes[:num_boxes, ...])
    boxes = box_list_ops.to_absolute_coordinates(boxes, height // stride,
                                                 width // stride)
    (y_center, x_center, _, _) = boxes.get_center_coordinates_and_sizes()
    center = tf.stack([y_center, x_center], axis=1)
    center_floor = tf.math.floor(center)
    indices = center_floor[:, 1] + center_floor[:, 0] * (width // stride)
    num_padding_boxes = tf.shape(gt_boxes)[0] - tf.shape(center_floor)[0]
    return tf.cast(tf.pad(indices, ((0, num_padding_boxes),)), tf.int64)

  return tf.map_fn(
      _create_center_indices, (gt_box_batch, gt_num_boxes_batch),
      dtype=tf.int64)


def assign_valid_boxes_mask_targets(gt_box_batch, gt_num_boxes_batch,
                                    image_size, stride):
  """Assign valid boxes."""
  height, width = image_size

  def _create_mask(elems):

    gt_boxes = elems[0]
    gt_num_boxes = elems[1]

    num_boxes = tf.cast(gt_num_boxes, tf.int32)

    valid_boxes = valid_bounding_boxes_mask(gt_boxes[:num_boxes, ...],
                                            height // stride, width // stride)
    mask = tf.concat(
        [valid_boxes, tf.zeros(tf.shape(gt_boxes)[0] - num_boxes)], axis=0)
    return mask

  # _create_mask((gt_box_batch[0, ...], gt_num_boxes_batch[0, ...]))
  return tf.map_fn(
      _create_mask, (gt_box_batch, gt_num_boxes_batch), dtype=tf.float32)
