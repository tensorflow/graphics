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
"""Functions for plotting."""

import os
import pickle
import re

import matplotlib.pyplot as plt
import numpy as np
import tensorflow as tf
from tensorflow_graphics.projects.points_to_3Dobjects.utils import tf_utils

from google3.pyglib import gfile


def plot_to_image(figure):
  """Converts a matplotlib figure into a TF image e.g. for TensorBoard."""
  figure.canvas.draw()
  width, height = figure.canvas.get_width_height()
  data_np = np.frombuffer(figure.canvas.tostring_rgb(), dtype='uint8')
  data_np = data_np.reshape([width, height, 3])
  image = tf.expand_dims(data_np, 0)
  return image


def resize_heatmap(centers, color=(1, 0, 0), stride=4):
  assert len(centers.shape) == 2
  centers = np.repeat(np.repeat(centers, stride, axis=0), stride, axis=1)
  centers = np.expand_dims(centers, axis=-1)
  cmin, cmax = np.min(centers), np.max(centers)
  centers = np.concatenate([np.ones(centers.shape) * color[0],
                            np.ones(centers.shape) * color[1],
                            np.ones(centers.shape) * color[2],
                            (centers-cmin)/cmax], axis=-1)
  return centers


def plot_heatmaps(image, detections, figsize=5):
  """Plot."""
  figure = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  width, height = image.shape[1], image.shape[0]
  if width != height:
    image = tf.image.pad_to_bounding_box(image, 0, 0, 640, 640)
    image = image.numpy()

  width, height = image.shape[1], image.shape[0]

  plt.imshow(np.concatenate([image.astype(float)/255.0,
                             1.0 * np.ones([height, width, 1])], axis=-1))
  num_predicted_objects = detections['detection_classes'].numpy().shape[0]
  for object_id in range(num_predicted_objects):
    for k, color in [['centers', [0, 1, 0]]
                     ]:
      class_id = int(detections['detection_classes'][object_id].numpy())
      centers = detections[k][:, :, class_id].numpy()
      color = [[1, 0, 0], [0, 1, 0], [0, 0, 1]][object_id]
      plt.imshow(resize_heatmap(centers, color=color))
  plt.axis('off')
  plt.tight_layout()
  return figure


def draw_coordinate_frame(camera_intrinsic, pose_world2camera, dot):
  """Draw coordinate system frame."""
  print(dot)

  width = camera_intrinsic[0, 2] * 2.0
  height = camera_intrinsic[1, 2] * 2.0

  plt.plot([0, width], [height / 4.0 * 3.0, height / 4.0 * 3.0], 'g--')
  plt.plot([width / 2.0, width / 2.0], [0.0, height], 'g--')

  camera_intrinsic = np.reshape(camera_intrinsic, [3, 3])
  pose_world2camera = np.reshape(pose_world2camera, [3, 4])
  frame = np.array([[0, 0, 0, 1],
                    [0.1, 0, 0, 1],
                    [0, 0.1, 0, 1],
                    [0, 0, 0.1, 1]], dtype=np.float32).T  # Shape: (4, 4)
  projected_frame = camera_intrinsic @ pose_world2camera @ frame
  projected_frame = projected_frame[0:2, :] / projected_frame[2, :]
  plt.plot(projected_frame[0, [0, 1]], projected_frame[1, [0, 1]], 'r-')
  plt.plot(projected_frame[0, [0, 2]], projected_frame[1, [0, 2]], 'g-')
  plt.plot(projected_frame[0, [0, 3]], projected_frame[1, [0, 3]], 'b-')

  dot_proj = camera_intrinsic @ \
      pose_world2camera @ [dot[0, 0], dot[0, 1], dot[0, 2], 1.0]
  dot_proj /= dot_proj[2]
  print(dot_proj)
  plt.plot(dot_proj[0], dot_proj[1], 'y*')


def plot_gt_boxes_2d(sample, shape_pointclouds, figsize=5):
  """Plot."""
  _ = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  plt.imshow(sample['image'])

  # Plot ground truth boxes
  sample['detection_boxes'] = sample['groundtruth_boxes'].numpy()
  colors = ['r.', 'g.', 'b.']
  for i in range(sample['num_boxes'].numpy()):
    shape_id = sample['shapes'][i]
    pointcloud = tf.transpose(shape_pointclouds[shape_id])
    translation = sample['translations_3d'][i]
    rotation = tf.reshape(sample['rotations_3d'][i], [3, 3])
    size = np.diag(sample['sizes_3d'][i])

    trafo_pc = \
        rotation @ size @ (pointcloud / 2.0) + tf.expand_dims(translation, 1)
    trafo_pc = tf.concat([trafo_pc, tf.ones([1, 512])], axis=0)
    projected_pointcloud = \
        tf.reshape(sample['k'], [3, 3]) @ sample['rt'] @ trafo_pc
    projected_pointcloud /= projected_pointcloud[2, :]
    plt.plot(projected_pointcloud[0, :],
             projected_pointcloud[1, :], colors[i % 3])

    y_min, x_min, y_max, x_max = sample['detection_boxes'][i]
    y_min *= sample['original_image_spatial_shape'][1].numpy()
    y_max *= sample['original_image_spatial_shape'][1].numpy()
    x_min *= sample['original_image_spatial_shape'][0].numpy()
    x_max *= sample['original_image_spatial_shape'][0].numpy()
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             linestyle='dashed')


def show_sdf(sdf, figsize=5, resolution=32):
  _, axis = plt.subplots(1, 3, figsize=(3*figsize, figsize))
  sdf = tf.reshape(sdf, [resolution, resolution, resolution])
  for a in range(3):
    proj_sdf = tf.transpose(tf.reduce_min(sdf, axis=a))
    c = axis[a].matshow(proj_sdf.numpy())
    plt.colorbar(c, ax=axis[a])


def plot_gt_boxes_3d(sample, shape_pointclouds, figsize=5):
  """Plot."""
  intrinsics = sample['k'].numpy()
  pose_world2camera = sample['rt'].numpy()

  _ = plt.figure(figsize=(figsize, figsize))
  plt.clf()

  intrinsics = np.reshape(intrinsics, [3, 3])
  pose_world2camera = np.reshape(pose_world2camera, [3, 4])

  # Plot ground truth boxes
  # num_boxes = sample['groundtruth_boxes'].shape[0]
  colors = ['r', 'g', 'b', 'c', 'm', 'y']
  colors2 = ['r.', 'g.', 'b.']
  for i in [2, 1, 0]:
    shape_id = sample['shapes'][i]
    pointcloud = tf.transpose(shape_pointclouds[shape_id])
    translation = sample['translations_3d'][i]
    rotation = tf.reshape(sample['rotations_3d'][i], [3, 3])
    size = np.diag(sample['sizes_3d'][i])

    trafo_pc = \
        rotation @ size @ (pointcloud / 2.0) + tf.expand_dims(translation, 1)
    trafo_pc = tf.concat([trafo_pc, tf.ones([1, 512])], axis=0)
    projected_pointcloud = \
        tf.reshape(sample['k'], [3, 3]) @ sample['rt'] @ trafo_pc
    projected_pointcloud /= projected_pointcloud[2, :]
    plt.plot(projected_pointcloud[0, :],
             projected_pointcloud[1, :], 'w.', markersize=5)
    plt.plot(projected_pointcloud[0, :],
             projected_pointcloud[1, :], colors2[i], markersize=3)

    predicted_pose_obj2world = np.eye(4)
    predicted_pose_obj2world[0:3, 0:3] = \
        tf.reshape(sample['rotations_3d'][i], [3, 3]).numpy()
    predicted_pose_obj2world[0:3, 3] = sample['translations_3d'][i].numpy()
    draw_bounding_box_3d(sample['sizes_3d'][i].numpy(),
                         predicted_pose_obj2world,
                         intrinsics, pose_world2camera,
                         linestyle='solid', color='w', linewidth=3)
    draw_bounding_box_3d(sample['sizes_3d'][i].numpy(),
                         predicted_pose_obj2world,
                         intrinsics, pose_world2camera,
                         linestyle='solid', color=colors[i], linewidth=1)
  # draw_coordinate_frame(intrinsics, pose_world2camera, sample['dot'])


CLASSES = ('chair', 'sofa', 'table', 'bottle', 'bowl', 'mug', 'bowl', 'mug')


def plot_boxes_2d(image, sample, predictions, projection=True, groundtruth=True,
                  figsize=5,
                  class_id_to_name=CLASSES):
  """Plot."""
  batch_id = 0

  figure = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  plt.imshow(image)

  if projection:
    points = predictions['projected_pointclouds'].numpy()
    colors = ['r.', 'g.', 'b.', 'c.', 'm.', 'y.']
    # print('HERE:', points.shape)
    for i in range(points.shape[0]):
      # print(i, points.shape)
      plt.plot(points[i, :, 0], points[i, :, 1],
               colors[int(predictions['detection_classes'][i])])

  # Plot ground truth boxes
  if groundtruth:
    sample['detection_boxes'] = sample['groundtruth_boxes'][batch_id].numpy()
    for i in range(sample['detection_boxes'].shape[0]):
      y_min, x_min, y_max, x_max = sample['detection_boxes'][i]
      y_min *= sample['original_image_spatial_shape'][batch_id][1].numpy()
      y_max *= sample['original_image_spatial_shape'][batch_id][1].numpy()
      x_min *= sample['original_image_spatial_shape'][batch_id][0].numpy()
      x_max *= sample['original_image_spatial_shape'][batch_id][0].numpy()
      plt.plot([x_min, x_max, x_max, x_min, x_min],
               [y_min, y_min, y_max, y_max, y_min],
               linestyle='dashed')

  # Plot predicted boxes
  colors = ['r', 'g', 'b', 'c', 'm', 'y']
  for i in range(predictions['detection_boxes'].shape[0]):
    x_min, y_min, x_max, y_max = predictions['detection_boxes'][i]
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             linestyle='solid',
             color=colors[int(predictions['detection_classes'][i])])
    plt.text(x_min, y_min, str(i) + '_' +
             class_id_to_name[int(predictions['detection_classes'][i])] +
             str(int(predictions['detection_scores'][i]*1000) / 1000.0))
  plt.axis('off')
  plt.tight_layout()
  return figure


def plot_boxes_3d(image, sample, predictions, figsize=5, groundtruth=True,
                  class_id_to_name=CLASSES):
  """Plot."""
  batch_id = 0

  intrinsics = sample['k'][batch_id].numpy()
  pose_world2camera = sample['rt'][batch_id].numpy()

  figure = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  plt.imshow(image)

  intrinsics = np.reshape(intrinsics, [3, 3])
  pose_world2camera = np.reshape(pose_world2camera, [3, 4])

  # Plot ground truth boxes
  if groundtruth:
    num_boxes = sample['groundtruth_boxes'][batch_id].shape[0]
    sample['detection_boxes'] = sample['groundtruth_boxes'][batch_id].numpy()
    colors = ['c', 'm', 'y']
    for i in range(num_boxes):
      predicted_pose_obj2world = np.eye(4)
      predicted_pose_obj2world[0:3, 0:3] = \
          tf.reshape(sample['rotations_3d'][batch_id][i], [3, 3]).numpy()
      predicted_pose_obj2world[0:3, 3] = \
          sample['translations_3d'][batch_id][i].numpy()
      draw_bounding_box_3d(sample['sizes_3d'][batch_id][i].numpy(),
                           predicted_pose_obj2world,
                           intrinsics, pose_world2camera,
                           linestyle='dashed', color=colors[i % 3])
      y_min, x_min, y_max, x_max = sample['detection_boxes'][i]
      y_min *= sample['original_image_spatial_shape'][batch_id][1].numpy()
      y_max *= sample['original_image_spatial_shape'][batch_id][1].numpy()
      x_min *= sample['original_image_spatial_shape'][batch_id][0].numpy()
      x_max *= sample['original_image_spatial_shape'][batch_id][0].numpy()
      plt.text(x_max, y_min,
               str(i) + '_gt_' + \
               class_id_to_name[int(sample['groundtruth_valid_classes'][batch_id][i])])

  # Plot predicted boxes
  colors = ['r', 'g', 'b', 'c', 'm', 'y', 'c', 'm', 'y', 'c', 'm', 'y']
  num_boxes = predictions['rotations_3d'].shape[0]
  for i in range(num_boxes):
    predicted_pose_obj2world = np.eye(4)
    predicted_pose_obj2world[0:3, 0:3] = predictions['rotations_3d'][i].numpy()
    predicted_pose_obj2world[0:3, 3] = predictions['translations_3d'][i].numpy()
    draw_bounding_box_3d(predictions['sizes_3d'].numpy()[i],
                         predicted_pose_obj2world,
                         intrinsics, pose_world2camera,
                         linestyle='solid',
                         color=colors[int(predictions['detection_classes'][i])])
    x_min, y_min, x_max, y_max = predictions['detection_boxes'][i]
    plt.text(x_min, y_min, str(i) + '_' +
             class_id_to_name[int(predictions['detection_classes'][i])] +
             str(int(predictions['detection_scores'][i] * 1000) / 1000.0))
  plt.axis('off')
  plt.tight_layout()
  return figure


def plot_detections(
    image,
    intrinsics,
    pose_world2camera,
    detections,
    labels,
    figsize=0.1):
  """Plot."""
  figure = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  plt.imshow(np.concatenate([image.astype(float)/255.0,
                             0.2 * np.ones([256, 256, 1])], axis=-1))

  # Plot heatmaps
  num_predicted_objects = detections['detection_classes'].numpy().shape[0]
  for object_id in range(num_predicted_objects):
    for k, color in [['centers_sigmoid', [0, 1, 0]],
                     ['centers_nms', [1, 0, 0]]]:
      class_id = int(detections['detection_classes'][object_id].numpy())
      centers = detections[k][:, :, class_id].numpy()
      plt.imshow(resize_heatmap(centers, color=color))

  intrinsics = np.reshape(intrinsics, [3, 3])
  pose_world2camera = np.reshape(pose_world2camera, [3, 4])

  for j, [boxes, style] in enumerate([[labels, 'dashed'],
                                      [detections, 'solid']]):
    number_of_boxes = boxes['detection_boxes'].shape[0]
    for i in range(number_of_boxes):
      predicted_pose_obj2world = np.eye(4)
      predicted_pose_obj2world[0:3, 0:3] = boxes['rotations_3d'][i].numpy()
      predicted_pose_obj2world[0:3, 3] = boxes['center3d'][i].numpy()
      draw_bounding_box_3d(boxes['size3d'].numpy()[i],
                           predicted_pose_obj2world,
                           intrinsics, pose_world2camera,
                           linestyle=style)
      if j == 0:
        if isinstance(boxes['detection_boxes'], tf.Tensor):
          boxes['detection_boxes'] = boxes['detection_boxes'].numpy()
        # if isinstance(boxes['detection_classes'], tf.Tensor):
        #   boxes['detection_classes'] = boxes['detection_classes'].numpy()

        x_min, y_min, x_max, y_max = boxes['detection_boxes'][i]
        # plt.text(x_min, y_min,
        #          class_id_to_name[int(boxes['detection_classes'][i])])
        plt.plot([x_min, x_max, x_max, x_min, x_min],
                 [y_min, y_min, y_max, y_max, y_min],
                 linestyle=style)

  plt.axis('off')
  plt.tight_layout()
  return figure


def plot_all_heatmaps(image, detections, figsize=0.1, num_classes=6):
  """Plot."""
  if figsize:
    print(figsize)
  figure, axis = plt.subplots(1, num_classes, figsize=(num_classes * 5, 5))
  for class_id in range(num_classes):
    for k, color in [['centers_sigmoid', [0, 1, 0]],
                     ['centers_nms', [1, 0, 0]]]:
      axis[class_id].imshow(np.concatenate(
          [image.astype(float)/255.0, 0.5 * np.ones([256, 256, 1])], axis=-1))
      centers = detections[k][:, :, class_id].numpy()
      axis[class_id].imshow(resize_heatmap(centers, color=color))
  return figure


def plot_gt_heatmaps(image, heatmaps, num_classes=6):
  figure, axis = plt.subplots(1, num_classes, figsize=(num_classes * 4, 4))
  for class_id in range(num_classes):
    axis[class_id].imshow(np.concatenate(
        [image, 0.5 * np.ones([image.shape[0], image.shape[1], 1])], axis=-1))
    centers = heatmaps[:, :, class_id].numpy()
    axis[class_id].imshow(resize_heatmap(centers, color=[255, 0, 0]))
  return figure


def draw_bounding_box_3d(size, pose, camera_intrinsic, world2camera,
                         linestyle='solid', color=None, linewidth=1):
  """Draw bounding box."""
  size = size * 0.5

  origin = np.zeros([4, 1])
  origin[3, 0] = 1.0
  bbox3d = np.tile(origin, [1, 10])  # shape: (4, 10)
  bbox3d[0:3, 0] += np.array([-size[0], -size[1], -size[2]])
  bbox3d[0:3, 1] += np.array([size[0], -size[1], -size[2]])
  bbox3d[0:3, 2] += np.array([size[0], -size[1], size[2]])
  bbox3d[0:3, 3] += np.array([-size[0], -size[1], size[2]])

  bbox3d[0:3, 4] += np.array([-size[0], size[1], -size[2]])
  bbox3d[0:3, 5] += np.array([size[0], size[1], -size[2]])
  bbox3d[0:3, 6] += np.array([size[0], size[1], size[2]])
  bbox3d[0:3, 7] += np.array([-size[0], size[1], size[2]])

  bbox3d[0:3, 8] += np.array([0.0, -size[1], 0.0])
  bbox3d[0:3, 9] += np.array([0.0, -size[1], -size[2]])

  projected_bbox3d = camera_intrinsic @ world2camera @ pose @ bbox3d
  projected_bbox3d = projected_bbox3d[0:2, :] / projected_bbox3d[2, :]

  lw = linewidth
  plt.plot(projected_bbox3d[0, [0, 4, 7, 3]],
           projected_bbox3d[1, [0, 4, 7, 3]],
           linewidth=lw, linestyle=linestyle, color=color)
  plt.plot(projected_bbox3d[0, [1, 5, 6, 2]],
           projected_bbox3d[1, [1, 5, 6, 2]],
           linewidth=lw, linestyle=linestyle, color=color)
  plt.plot(projected_bbox3d[0, [0, 1, 2, 3, 0]],
           projected_bbox3d[1, [0, 1, 2, 3, 0]],
           linewidth=lw, linestyle=linestyle, color=color)
  plt.plot(projected_bbox3d[0, [4, 5, 6, 7, 4]],
           projected_bbox3d[1, [4, 5, 6, 7, 4]],
           linewidth=lw, linestyle=linestyle, color=color)
  plt.plot(projected_bbox3d[0, [8, 9]],
           projected_bbox3d[1, [8, 9]],
           linewidth=lw, linestyle=linestyle, color=color)


def plot_prediction(inputs, outputs, figsize=0.1, batch_id=0, plot_2d=False):
  """Plot bounding box predictions along ground truth labels.

  Args:
    inputs: Dict of batched inputs to the network.
    outputs: Dict of batched outputs of the network.
    figsize: The size of the figure.
    batch_id: The batch entry to plot.
    plot_2d: Whether 2D bounding boxes should be shown or not.

  Returns:
    A matplotlib figure.

  """
  image = inputs['image'][batch_id].numpy()
  size2d = inputs['box_dim2d'][batch_id].numpy()
  size3d = inputs['box_dim3d'][batch_id].numpy()[[0, 2, 1]]
  center2d = inputs['center2d'][batch_id].numpy()
  center3d = inputs['center3d'][batch_id].numpy()
  predicted_center2d = outputs['center2d'][batch_id].numpy()
  predicted_size2d = outputs['size2d'][batch_id].numpy()
  predicted_rotation = outputs['rotation'][batch_id].numpy()
  predicted_center3d = outputs['center3d'][batch_id].numpy().T
  predicted_size3d = outputs['size3d'][batch_id].numpy()[[0, 2, 1]]
  # dot = outputs['dot'][batch_id].numpy()
  intrinsics = inputs['k'][batch_id].numpy()
  pose_world2camera = inputs['rt'][batch_id].numpy()
  object_translation = np.squeeze(center3d[0:3])
  object_rotation = inputs['rotation'][batch_id].numpy()

  pose_obj2world = np.eye(4)
  rad = np.deg2rad(object_rotation*-1)
  cos = np.cos(rad)
  sin = np.sin(rad)
  pose_obj2world[0, 0] = cos
  pose_obj2world[0, 1] = sin
  pose_obj2world[1, 1] = cos
  pose_obj2world[1, 0] = -sin
  pose_obj2world[0:3, 3] = object_translation

  predicted_pose_obj2world = np.eye(4)
  predicted_pose_obj2world[0:2, 0:2] = predicted_rotation
  predicted_pose_obj2world[0:3, 3] = predicted_center3d

  figure = plt.figure(figsize=(figsize, figsize))
  plt.clf()
  plt.imshow(image / 255.)
  plt.plot(center2d[0], center2d[1], 'g*')

  def draw_ground_plane(camera_intrinsic, pose_world2camera):
    """Draw ground plane as grid.

    Args:
      camera_intrinsic: Camera intrinsic.
      pose_world2camera: Camera extrinsic.
    """
    line = np.array([[-3, 3, 0, 1], [3, 3, 0, 1]]).T
    projected_line = camera_intrinsic @ pose_world2camera @ line
    projected_line = projected_line[0:2, :] / projected_line[2, :]
    plt.plot(projected_line[0, [0, 1]], projected_line[1, [0, 1]],
             'black',
             linewidth=1)

  def draw_bounding_box_2d(center, size, style='b+-'):
    bbox2d = np.tile(np.reshape(center, [1, 2]), [4, 1])  # shape: (4, 2)
    bbox2d[0, :] += np.array([-size[0], -size[1]])
    bbox2d[1, :] += np.array([size[0], -size[1]])
    bbox2d[2, :] += np.array([size[0], size[1]])
    bbox2d[3, :] += np.array([-size[0], size[1]])
    plt.plot(bbox2d[[0, 1, 2, 3, 0], 0], bbox2d[[0, 1, 2, 3, 0], 1], style)

  draw_bounding_box_3d(size3d, pose_obj2world, intrinsics,
                       pose_world2camera, 'dashed')
  draw_ground_plane(intrinsics, pose_world2camera)
  # draw_coordinate_frame(intrinsics, pose_world2camera)
  draw_bounding_box_3d(predicted_size3d, predicted_pose_obj2world,
                       intrinsics, pose_world2camera)

  if plot_2d:
    draw_bounding_box_2d(center2d, size2d  / 2, 'g-')
    draw_bounding_box_2d(predicted_center2d, predicted_size2d / 2, 'b-')
  return figure


def matrix_from_angle(angle: float, axis: int):
  matrix = np.eye(3)
  if axis == 1:
    matrix[0, 0] = np.cos(angle)
    matrix[2, 2] = np.cos(angle)
    matrix[2, 0] = -np.sin(angle)
    matrix[0, 2] = np.sin(angle)
  return matrix


def save_for_blender(detections,
                     sample,
                     log_dir, dict_clusters, shape_pointclouds,
                     class_id_to_name=CLASSES):
  """Save for blender."""
  # VisualDebugging uses the OpenCV coordinate representation
  # while the dataset uses OpenGL (left-hand) so make sure to convert y and z.

  batch_id = 0
  prefix = '/cns/lu-d/home/giotto3d/datasets/shapenet/raw/'
  sufix = 'models/model_normalized.obj'

  blender_dict = {}
  blender_dict['image'] = \
      tf.io.decode_image(sample['image_data'][batch_id]).numpy()
  blender_dict['world_to_cam'] = sample['rt'].numpy()
  num_predicted_shapes = int(detections['sizes_3d'].shape[0])
  blender_dict['num_predicted_shapes'] = num_predicted_shapes
  blender_dict['predicted_rotations_3d'] = \
      tf.reshape(detections['rotations_3d'], [-1, 3, 3]).numpy()
  blender_dict['predicted_rotations_y'] = [
      tf_utils.euler_from_rotation_matrix(
          tf.reshape(detections['rotations_3d'][i], [3, 3]), 1).numpy()
      for i in range(num_predicted_shapes)]
  blender_dict['predicted_translations_3d'] = \
      detections['translations_3d'].numpy()
  blender_dict['predicted_sizes_3d'] = detections['sizes_3d'].numpy()
  predicted_shapes_path = []
  for i in range(num_predicted_shapes):
    shape = detections['shapes'][i].numpy()
    _, class_str, model_str = dict_clusters[shape]
    filename = os.path.join(prefix, class_str, model_str, sufix)
    predicted_shapes_path.append(filename)
  blender_dict['predicted_shapes_path'] = predicted_shapes_path
  blender_dict['predicted_class'] = [
      class_id_to_name[int(detections['detection_classes'][i].numpy())]
      for i in range(num_predicted_shapes)]

  blender_dict['predicted_pointcloud'] = [
      shape_pointclouds[int(detections['shapes'][i].numpy())]
      for i in range(num_predicted_shapes)]

  num_groundtruth_shapes = int(sample['sizes_3d'][batch_id].shape[0])
  blender_dict['num_groundtruth_shapes'] = num_groundtruth_shapes
  blender_dict['groundtruth_rotations_3d'] = \
      tf.reshape(sample['rotations_3d'][batch_id], [-1, 3, 3]).numpy()
  blender_dict['groundtruth_rotations_y'] = [
      tf_utils.euler_from_rotation_matrix(
          tf.reshape(sample['rotations_3d'][batch_id][i], [3, 3]), 1).numpy()
      for i in range(sample['num_boxes'][batch_id].numpy())]
  blender_dict['groundtruth_translations_3d'] = \
      sample['translations_3d'][batch_id].numpy()
  blender_dict['groundtruth_sizes_3d'] = sample['sizes_3d'][batch_id].numpy()
  groundtruth_shapes_path = []
  for i in range(num_groundtruth_shapes):
    class_str = str(sample['classes'][batch_id, i].numpy()).zfill(8)
    model_str = str(sample['mesh_names'][batch_id, i].numpy())[2:-1]
    filename = os.path.join(prefix, class_str, model_str, sufix)
    groundtruth_shapes_path.append(filename)
  blender_dict['groundtruth_shapes_path'] = groundtruth_shapes_path
  blender_dict['groundtruth_classes'] = \
      sample['groundtruth_valid_classes'].numpy()

  path = log_dir + '.pkl'
  with gfile.Open(path, 'wb') as file:
    pickle.dump(blender_dict, file)


def obj_read_for_gl(filename, texture_size=(32, 32)):
  """Read vertex and part information from OBJ file."""

  if texture_size:
    print(texture_size)
  with gfile.Open(filename, 'r') as f:
    content = f.readlines()

    vertices = []
    texture_coords = []
    vertex_normals = []

    group_name = None
    material_name = None

    faces = []
    faces_tex = []
    faces_normals = []
    face_groups = []
    material_ids = []

    for i in range(len(content)):
      line = content[i]
      parts = re.split(r'\s+', line)

      # if parts[0] == 'mtllib':
      #   material_file = parts[1]

      # Vertex information -----------------------------------------------------
      if parts[0] == 'v':
        vertices.append([float(v) for v in parts[1:4]])
      if parts[0] == 'vt':
        texture_coords.append([float(v) for v in parts[1:4]])
      if parts[0] == 'vn':
        vertex_normals.append([float(v) for v in parts[1:4]])

      if parts[0] == 'g':
        group_name = parts[1]
      if parts[0] == 'usemtl':
        material_name = parts[1]

      # Face information ------------------------------------------------------
      if parts[0] == 'f':
        vertex_index, tex_index, normal_index = 0, 0, 0
        current_face, current_face_tex, current_face_norm = [], [], []
        for j in range(1, 4):
          face_info = parts[j]
          if face_info.count('/') == 2:
            vertex_index, tex_index, normal_index = face_info.split('/')
            if not tex_index:
              tex_index = 0
          elif face_info.count('/') == 1:
            vertex_index, tex_index = face_info.split('/')
          elif face_info.count('/') == 0:
            vertex_index = face_info
          current_face.append(int(vertex_index)-1)
          current_face_tex.append(int(tex_index)-1)
          current_face_norm.append(int(normal_index)-1)
        faces.append(current_face)
        faces_tex.append(current_face_tex)
        faces_normals.append(current_face_norm)
        face_groups.append(group_name)
        material_ids.append(material_name)

    vertices = np.array(vertices)
    texture_coords = np.array(texture_coords)
    vertex_normals = np.array(vertex_normals)
    has_tex_coord, has_normals = True, True
    if texture_coords.shape[0] == 0:
      has_tex_coord = False
    if vertex_normals.shape[0] == 0:
      has_normals = False

    faces = np.array(faces)
    faces_tex = np.array(faces_tex)
    faces_normals = np.array(faces_normals)

    n_faces = faces.shape[0]
    vertex_positions = np.zeros((n_faces, 3, 3), dtype=np.float32)
    tex_coords = np.zeros((n_faces, 3, 2), dtype=np.float32)
    normals = np.zeros((n_faces, 3, 3), dtype=np.float32)
    for i in range(n_faces):
      for j in range(3):
        vertex_positions[i, j, :] = vertices[faces[i, j], :]
        if has_tex_coord:
          tex_coords[i, j, :] = texture_coords[faces_tex[i, j], :2]
        if has_normals:
          normals[i, j, :] = vertex_normals[faces_normals[i, j], :]

  # Material info --------------------------------------------------------------
  return vertex_positions, \
         tex_coords, \
         normals, \
         material_ids, \
         vertices, \
         faces


def plot_labeled_2d_boxes(sample, batch_id=0):
  """Plot."""
  image = tf.io.decode_image(sample['image_data'][batch_id]).numpy()  / 255.
  image = sample['image'][batch_id].numpy()[..., ::-1]
  image2 = np.reshape(image, [-1, 3])
  image2 -= np.min(image2, axis=0)
  image2 /= np.max(image2, axis=0)
  image = np.reshape(image2, [256, 256, 3])
  sample['detection_boxes'] = sample['groundtruth_boxes'][batch_id].numpy()

  figure = plt.figure(figsize=(5, 5))
  plt.clf()
  plt.imshow(image)

  for i in range(sample['groundtruth_boxes'][batch_id].shape[0]):
    y_min, x_min, y_max, x_max = sample['detection_boxes'][i] * 256.0
    plt.plot([x_min, x_max, x_max, x_min, x_min],
             [y_min, y_min, y_max, y_max, y_min],
             linestyle='dashed')
  return figure

