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
"""Collision loss."""

import tensorflow as tf
from tensorflow_graphics.math.interpolation import trilinear
from tensorflow_graphics.projects.points_to_3Dobjects.models import centernet_utils


class CollisionLoss:
  """Collision loss."""

  def __init__(self, reduce_batch=False, return_values=False, tol=0.001):
    self.reduce_batch = reduce_batch
    self.return_values = return_values
    self.tol = tol

  def __call__(self, prediction, gt, sample):
    sdfs, pointclouds, sizes_3d, translations_3d, rotations_3d = prediction
    translations_3d = translations_3d.numpy()
    translations_3d[0, 0, :] = [0.0, 0.0, 0.3]
    translations_3d[0, 1, :] = [0.0, -0.25, 0.0]
    translations_3d = tf.constant(translations_3d)

    rotations_3d = rotations_3d.numpy()
    rotations_3d[0, 0, :] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    rotations_3d[0, 1, :] = [1.0, 0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 1.0]
    rotations_3d = tf.constant(rotations_3d)

    sizes_3d = sizes_3d.numpy()
    sizes_3d[0, 0, :] = [0.5, 1.0, 0.5]
    sizes_3d[0, 1, :] = [0.5, 0.5, 0.5]
    sizes_3d = tf.constant(sizes_3d)

    pointclouds_world = centernet_utils.transform_pointcloud(
        pointclouds / 2.0, sizes_3d, rotations_3d, translations_3d)

    all_sdf_values = [[], [], []]
    total_collision_loss = tf.constant(0.0)
    num_objects = sdfs.shape[1]
    for i in range(num_objects):  # iterate over all objects
      object_loss = 0
      for j in range(num_objects):  # iterate over other objects
        if i == j:
          continue

        # Transform into coordinate system of other object
        size_3d_object_j = sizes_3d[:, j:j + 1]
        translation_3d_object_j = translations_3d[:, j:j + 1]
        rotations_3d_object_j = rotations_3d[:, j:j + 1]
        pointcloud_i_object_j = centernet_utils.transform_pointcloud(
            pointclouds_world[:, i:i + 1], size_3d_object_j,
            rotations_3d_object_j, translation_3d_object_j, inverse=True) * 2.0

        # Map into SDF
        sdf_j = tf.expand_dims(sdfs[:, j:j + 1], -1) * -1.0  # inside positive
        pointcloud_i_sdf_j = (pointcloud_i_object_j *
                              (29.0/32.0) / 2.0 + 0.5) * 32.0 - 0.5
        sdf_values = trilinear.interpolate(sdf_j, pointcloud_i_sdf_j)
        sdf_values = tf.nn.relu(sdf_values + self.tol)
        all_sdf_values[i].append(sdf_values)
        object_loss += tf.reduce_sum(sdf_values, axis=[1, 2, 3])
      robust_loss = 0.5*(object_loss*object_loss)/(1+object_loss*object_loss)
      total_collision_loss += robust_loss

    if self.reduce_batch:
      loss = tf.reduce_mean(total_collision_loss)
    else:
      loss = total_collision_loss
      if self.return_values:
        return loss, all_sdf_values
    return loss
