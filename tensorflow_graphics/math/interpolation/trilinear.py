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
"""This module implements trilinear interpolation."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def interpolate(grid_3d, sampling_points, name=None):
  """Trilinear interpolation on a 3D regular grid.

  Args:
    grid_3d: A tensor with shape `[A1, ..., An, H, W, D, C]` where H, W, D are
      height, width, depth of the grid and C is the number of channels.
    sampling_points: A tensor with shape `[A1, ..., An, M, 3]` where M is the
    number of sampling points. Sampling points outside the grid are projected
    in the grid borders.
    name:  A name for this op that defaults to "trilinear_interpolate".

  Returns:
    A tensor of shape `[A1, ..., An, M, C]`
  """

  with tf.compat.v1.name_scope(name, "trilinear_interpolate",
                               [grid_3d, sampling_points]):
    grid_3d = tf.convert_to_tensor(value=grid_3d)
    sampling_points = tf.convert_to_tensor(value=sampling_points)

    shape.check_static(
        tensor=grid_3d, tensor_name="grid_3d", has_rank_greater_than=3)
    shape.check_static(tensor=sampling_points,
                       tensor_name="sampling_points",
                       has_dim_equals=(-1, 3),
                       has_rank_greater_than=1)
    shape.compare_batch_dimensions(
        tensors=(grid_3d, sampling_points),
        last_axes=(-5, -3),
        tensor_names=("grid_3d", "sampling_points"),
        broadcast_compatible=True)
    voxel_cube_shape = tf.shape(input=grid_3d)[-4:-1]
    sampling_points.set_shape(sampling_points.shape)
    batch_dims = tf.shape(input=sampling_points)[:-2]
    num_points = tf.shape(input=sampling_points)[-2]

    bottom_left = tf.floor(sampling_points)
    top_right = bottom_left + 1
    bottom_left_index = tf.cast(bottom_left, tf.int32)
    top_right_index = tf.cast(top_right, tf.int32)
    x0_index, y0_index, z0_index = tf.unstack(bottom_left_index, axis=-1)
    x1_index, y1_index, z1_index = tf.unstack(top_right_index, axis=-1)
    index_x = tf.concat([x0_index, x1_index, x0_index, x1_index,
                         x0_index, x1_index, x0_index, x1_index], axis=-1)
    index_y = tf.concat([y0_index, y0_index, y1_index, y1_index,
                         y0_index, y0_index, y1_index, y1_index], axis=-1)
    index_z = tf.concat([z0_index, z0_index, z0_index, z0_index,
                         z1_index, z1_index, z1_index, z1_index], axis=-1)
    indices = tf.stack([index_x, index_y, index_z], axis=-1)
    clip_value = tf.convert_to_tensor(value=[voxel_cube_shape - 1],
                                      dtype=indices.dtype)
    indices = tf.clip_by_value(indices, 0, clip_value)
    content = tf.gather_nd(params=grid_3d,
                           indices=indices,
                           batch_dims=tf.size(input=batch_dims))
    distance_to_bottom_left = sampling_points - bottom_left
    distance_to_top_right = top_right - sampling_points
    x_x0, y_y0, z_z0 = tf.unstack(distance_to_bottom_left, axis=-1)
    x1_x, y1_y, z1_z = tf.unstack(distance_to_top_right, axis=-1)
    weights_x = tf.concat([x1_x, x_x0, x1_x, x_x0,
                           x1_x, x_x0, x1_x, x_x0], axis=-1)
    weights_y = tf.concat([y1_y, y1_y, y_y0, y_y0,
                           y1_y, y1_y, y_y0, y_y0], axis=-1)
    weights_z = tf.concat([z1_z, z1_z, z1_z, z1_z,
                           z_z0, z_z0, z_z0, z_z0], axis=-1)
    weights = tf.expand_dims(weights_x * weights_y * weights_z, axis=-1)

    interpolated_values = weights * content
    return tf.add_n(tf.split(interpolated_values, [num_points] * 8, -2))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
