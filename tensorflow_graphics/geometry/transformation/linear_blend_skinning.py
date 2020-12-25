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
"""This module implements TensorFlow linear blend skinning functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def blend(points,
          skinning_weights,
          bone_rotations,
          bone_translations,
          name=None):
  """Transforms the points using Linear Blend Skinning.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible and allow transforming full 3D shapes at once.
    In the following, B1 to Bm are optional batch dimensions, which allow
    transforming multiple poses at once.

  Args:
    points: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents a 3d point.
    skinning_weights: A tensor of shape `[A1, ..., An, W]`, where the last
      dimension represents the skinning weights of each bone.
    bone_rotations: A tensor of shape `[B1, ..., Bm, W, 3, 3]`, which represents
      the 3d rotations applied to each bone.
    bone_translations: A tensor of shape `[B1, ..., Bm, W, 3]`, which represents
      the 3d translation vectors applied to each bone.
    name: A name for this op that defaults to "linear_blend_skinning_blend".

  Returns:
    A tensor of shape `[B1, ..., Bm, A1, ..., An, 3]`, where the last dimension
    represents a 3d point.

  Raises:
    ValueError: If the shape of the input tensors are not supported.
  """
  with tf.compat.v1.name_scope(
      name, "linear_blend_skinning_blend",
      [points, skinning_weights, bone_rotations, bone_translations]):
    points = tf.convert_to_tensor(value=points)
    skinning_weights = tf.convert_to_tensor(value=skinning_weights)
    bone_rotations = tf.convert_to_tensor(value=bone_rotations)
    bone_translations = tf.convert_to_tensor(value=bone_translations)

    shape.check_static(
        tensor=points, tensor_name="points", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=bone_rotations,
        tensor_name="bone_rotations",
        has_rank_greater_than=2,
        has_dim_equals=((-2, 3), (-1, 3)))
    shape.check_static(
        tensor=bone_translations,
        tensor_name="bone_translations",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 3))
    shape.compare_dimensions(
        tensors=(skinning_weights, bone_rotations),
        tensor_names=("skinning_weights", "bone_rotations"),
        axes=(-1, -3))
    shape.compare_dimensions(
        tensors=(skinning_weights, bone_translations),
        tensor_names=("skinning_weights", "bone_translations"),
        axes=(-1, -2))
    shape.compare_batch_dimensions(
        tensors=(points, skinning_weights),
        tensor_names=("points", "skinning_weights"),
        last_axes=(-2, -2),
        broadcast_compatible=True)
    shape.compare_batch_dimensions(
        tensors=(bone_rotations, bone_translations),
        tensor_names=("bone_rotations", "bone_translations"),
        last_axes=(-3, -2),
        broadcast_compatible=True)

    num_bones = skinning_weights.shape[-1]

    def dim_value(dim):
      return 1 if dim is None else tf.compat.v1.dimension_value(dim)

    # TODO(b/148362025): factorize this block out
    points_batch_shape = shape.get_broadcasted_shape(
        points.shape[:-1], skinning_weights.shape[:-1])
    points_batch_shape = [dim_value(dim) for dim in points_batch_shape]

    points = tf.broadcast_to(points, points_batch_shape + [3])
    skinning_weights = tf.broadcast_to(skinning_weights,
                                       points_batch_shape + [num_bones])

    bones_batch_shape = shape.get_broadcasted_shape(
        bone_rotations.shape[:-3], bone_translations.shape[:-2])
    bones_batch_shape = [dim_value(dim) for dim in bones_batch_shape]

    bone_rotations = tf.broadcast_to(bone_rotations,
                                     bones_batch_shape + [num_bones, 3, 3])
    bone_translations = tf.broadcast_to(bone_translations,
                                        bones_batch_shape + [num_bones, 3])

    points_batch_dims = points.shape.ndims - 1
    bones_batch_dims = bone_rotations.shape.ndims - 3

    points = tf.reshape(points,
                        [1] * bones_batch_dims + points_batch_shape + [1, 3])
    skinning_weights = tf.reshape(skinning_weights, [1] * bones_batch_dims +
                                  points_batch_shape + [num_bones, 1])
    bone_rotations = tf.reshape(
        bone_rotations,
        bones_batch_shape + [1] * points_batch_dims + [num_bones, 3, 3])
    bone_translations = tf.reshape(
        bone_translations,
        bones_batch_shape + [1] * points_batch_dims + [num_bones, 3])

    transformed_points = rotation_matrix_3d.rotate(
        points, bone_rotations) + bone_translations
    weighted_points = tf.multiply(skinning_weights, transformed_points)
    blended_points = tf.reduce_sum(input_tensor=weighted_points, axis=-2)

    return blended_points


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
