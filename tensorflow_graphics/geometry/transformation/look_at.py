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
"""This module implements OpenGL lookAt functionalities."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def right_handed(camera_position, look_at, up_vector, name="right_handed"):
  """Builds a right handed look at view matrix.

  Note:
    In the following, A1 to An are optional batch dimensions.

  Args:
    camera_position: A tensor of shape `[A1, ..., An, 3]`, where the last
      dimension represents the 3D position of the camera.
    look_at: A tensor of shape `[A1, ..., An, 3]`, with the last dimension
      storing the position where the camera is looking at.
    up_vector: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      defines the up vector of the camera.
    name: A name for this op. Defaults to 'right_handed'.

  Raises:
    ValueError: if the all the inputs are not of the same shape, or if any input
    of of an unsupported shape.

  Returns:
    A tensor of shape `[A1, ..., An, 4, 4]`, containing right handed look at
    matrices.
  """
  with tf.name_scope(name):
    camera_position = tf.convert_to_tensor(value=camera_position)
    look_at = tf.convert_to_tensor(value=look_at)
    up_vector = tf.convert_to_tensor(value=up_vector)

    shape.check_static(
        tensor=camera_position,
        tensor_name="camera_position",
        has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=look_at, tensor_name="look_at", has_dim_equals=(-1, 3))
    shape.check_static(
        tensor=up_vector, tensor_name="up_vector", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(camera_position, look_at, up_vector),
        last_axes=-2,
        tensor_names=("camera_position", "look_at", "up_vector"),
        broadcast_compatible=False)

    z_axis = tf.linalg.l2_normalize(look_at - camera_position, axis=-1)
    horizontal_axis = tf.linalg.l2_normalize(
        vector.cross(z_axis, up_vector), axis=-1)
    vertical_axis = vector.cross(horizontal_axis, z_axis)

    batch_shape = tf.shape(input=horizontal_axis)[:-1]
    zeros = tf.zeros(
        shape=tf.concat((batch_shape, (3,)), axis=-1),
        dtype=horizontal_axis.dtype)
    one = tf.ones(
        shape=tf.concat((batch_shape, (1,)), axis=-1),
        dtype=horizontal_axis.dtype)

    matrix = tf.concat(
        (horizontal_axis, -vector.dot(horizontal_axis, camera_position),
         vertical_axis, -vector.dot(vertical_axis, camera_position), -z_axis,
         vector.dot(z_axis, camera_position), zeros, one),
        axis=-1)
    matrix_shape = tf.shape(input=matrix)
    output_shape = tf.concat((matrix_shape[:-1], (4, 4)), axis=-1)
    return tf.reshape(matrix, shape=output_shape)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
