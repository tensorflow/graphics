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
"""Functions that operate on voxels."""
import numpy as np
import tensorflow as tf
from tensorflow_graphics.math.interpolation import trilinear
import tensorflow_graphics.projects.radiance_fields.utils as utils


def get_mask_voxels(shape=(1, 128, 128, 128, 1), dtype=np.float32):
  """Generates a mask for a voxel grid by removing the borders."""
  voxels = np.ones(shape, dtype=dtype)
  voxels[:, [0, -1], :, :, :] = 0
  voxels[:, :, [0, -1], :, :] = 0
  voxels[:, :, :, [0, -1], :] = 0
  return tf.convert_to_tensor(voxels)


@tf.function
def inflate_voxels(voxels, kernel_size=3):
  """Inflates the occupied voxels in a voxel grid."""
  kernel = [kernel_size, kernel_size, kernel_size]
  voxels = tf.nn.pool(voxels, kernel, "MAX", [1, 1, 1], padding="SAME")
  voxels = tf.nn.pool(voxels, kernel, "AVG", [1, 1, 1], padding="SAME")
  return voxels


@tf.function
def ray_sample_voxel_grid(ray_points, voxels, w2v_alpha, w2v_beta):
  """Estimates the voxel value at the ray points using trilinear interpolation.

  Args:
    ray_points: A tensor of shape `[B, M, N, 3]` where B is the batch size,
      M is the number of rays and N the samples along the ray.
    voxels: A tensor of shape `[B, H, W, D, C]`, where B is the batch size, H,
      W, D is height, width and depth of the voxel grid and C is the number
      of channels.
    w2v_alpha: A tensor of shape `[B, 3]` where B the batch size.
    w2v_beta: A tensor of shape `[B, 3]` where B the batch size.

  Returns:
    A tensor of shape `[B, M, N, C]`
  """
  w2v_alpha = utils.match_intermediate_batch_dimensions(w2v_alpha, ray_points)
  w2v_beta = utils.match_intermediate_batch_dimensions(w2v_beta, ray_points)
  rays = w2v_alpha*ray_points + w2v_beta

  batch_size = tf.shape(voxels)[0]
  channels = tf.shape(voxels)[-1]
  target_shape = tf.concat([tf.shape(rays)[:-1], [channels]], axis=-1)
  rays = tf.reshape(rays, [batch_size, -1, 3])
  features_alpha = trilinear.interpolate(voxels, rays)
  return tf.reshape(features_alpha, target_shape)
