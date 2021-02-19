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
"""Utility functions for the CenterNet model."""
import tensorflow as tf


def softargmax(y, voxels, beta=50):
  """Softargmax class.

  Args:
    y:      float [batch_size, num_objects, num_shapes]
    voxels: float [num_shapes, 32, 32, 32]
    beta:   float scalar

  Returns:
    Softargmax
  """
  batch_size = y.shape[0]
  num_objects = y.shape[1]
  num_shapes = y.shape[2]
  softmax = tf.nn.softmax(beta * y)
  softmax = tf.reshape(softmax, [batch_size, num_objects, num_shapes, 1])
  voxels = tf.reshape(voxels, [1, num_shapes, -1])
  result = tf.reduce_sum(voxels * softmax, axis=2)
  return result


class SoftArgMax(tf.keras.layers.Layer):
  """Softargmax."""

  def __init__(self, beta, voxels, voxel_size=32):
    super(SoftArgMax, self).__init__()
    self.beta = beta
    self.voxels = voxels
    self.voxel_size = voxel_size

  def call(self, inputs):
    batch_size = inputs.shape[0]
    num_objects = inputs.shape[1]
    num_shapes = inputs.shape[2]
    softmax = tf.nn.softmax(self.beta * inputs, name='sm')
    softmax = tf.reshape(softmax,
                         [batch_size, num_objects, num_shapes, 1], name='rs1')
    voxels = tf.reshape(self.voxels,
                        [1, num_shapes, self.voxel_size**3], name='rs2')
    result = tf.reduce_sum(voxels * softmax, axis=2, name='sum')
    return tf.reshape(result, [batch_size, num_objects, -1], name='rs3')


def get_heatmap_values(heatmap, indices):
  b, _, _, n = heatmap.shape
  values = tf.reshape(heatmap, [b, -1, n])
  values = tf.gather(values, indices, batch_dims=1)
  return values


def assemble_pose(rotations_3d, translations_3d, sizes_3d):
  """Assemble pose."""
  batch_size = rotations_3d.shape[0]
  rotations_3d = tf.reshape(rotations_3d, [batch_size, -1, 3, 3])
  sizes_3d = tf.linalg.diag(sizes_3d)
  rotations_3d = rotations_3d @ sizes_3d
  translations_3d = tf.reshape(translations_3d, [batch_size, -1, 3, 1])
  poses = tf.concat([
      tf.concat([rotations_3d, tf.zeros([batch_size, 3, 1, 3])], axis=-2),
      tf.concat([translations_3d, tf.ones([batch_size, 3, 1, 1])], axis=-2)],
                    axis=-1)
  return poses


def transform_pointcloud(pointclouds, sizes_3d, rotations_3d, translations_3d,
                         inverse=False):
  """Transform pointcloud."""
  batch_size = rotations_3d.shape[0]
  rotations_3d = tf.reshape(rotations_3d, [batch_size, -1, 3, 3])
  if inverse:
    sizes_3d = tf.linalg.diag(1.0 / sizes_3d)
    rotations_3d = tf.transpose(rotations_3d, [0, 1, 3, 2])
    pointclouds -= tf.expand_dims(translations_3d, 2)
    pointclouds = tf.transpose(pointclouds, [0, 1, 3, 2])
    pointclouds = sizes_3d @ rotations_3d @ pointclouds
    pointclouds = tf.transpose(pointclouds, [0, 1, 3, 2])
  else:
    sizes_3d = tf.linalg.diag(sizes_3d)
    pointclouds = tf.transpose(pointclouds, [0, 1, 3, 2])
    pointclouds = rotations_3d @ sizes_3d @ pointclouds
    pointclouds = tf.transpose(pointclouds, [0, 1, 3, 2])
    pointclouds += tf.expand_dims(translations_3d, 2)
  return pointclouds


def decode_box_3d(output, indices, rotation_svd):
  """Return size, translation and rotation."""
  batch_size = output['centers'].shape[0]
  sizes_3d = get_heatmap_values(output['sizes_offset_3d'], indices)
  sizes_3d += 0.25
  translations_3d = get_heatmap_values(output['translations_offset_3d'],
                                       indices)

  rotations_3d = get_heatmap_values(output['rotations_offset_3d'], indices)
  rotations_3d += tf.constant([1.0, 0.0, 0.0,
                               0.0, 1.0, 0.0,
                               0.0, 0.0, 1.0])
  rotations_3d = tf.reshape(rotations_3d, [batch_size, -1, 3, 3])
  if rotation_svd:
    s, u, v = tf.linalg.svd(rotations_3d)
    det = tf.linalg.det(tf.matmul(u, tf.linalg.matrix_transpose(v)))
    det = tf.expand_dims(det, axis=-1)
    a = tf.ones(det.shape)
    s = tf.linalg.diag(tf.concat([a, a, det], axis=-1))
    rotations_3d = tf.matmul(u, tf.matmul(s, v, adjoint_b=True))

  rotations_3d = tf.reshape(rotations_3d, [batch_size, -1, 9])
  return sizes_3d, translations_3d, rotations_3d
