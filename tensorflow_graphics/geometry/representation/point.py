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
"""Tensorflow point utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def distance_to_ray(point, origin, direction, keepdims=True, name=None):
  """Computes the distance from a M-d point to a M-d ray.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    point: A tensor of shape `[A1, ..., An, M]`.
    origin: A tensor of shape `[A1, ..., An, M]`.
    direction: A tensor of shape `[A1, ..., An, M]`. The last dimension must be
      normalized.
    keepdims: A `bool`, whether to keep the last dimension with length 1 or to
      remove it.
    name: A name for this op. Defaults to "point_distance_to_ray".

  Returns:
    A tensor of shape `[A1, ..., An, 1]` containing the distance from each point
    to the corresponding ray.

  Raises:
    ValueError: If the shape of `point`, `origin`, or 'direction' is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "point_distance_to_ray",
                               [point, origin, direction]):
    point = tf.convert_to_tensor(value=point)
    origin = tf.convert_to_tensor(value=origin)
    direction = tf.convert_to_tensor(value=direction)

    shape.compare_dimensions((point, origin, direction), -1,
                             ("point", "origin", "direction"))
    shape.compare_batch_dimensions(
        tensors=(point, origin, direction),
        last_axes=-2,
        broadcast_compatible=True)
    direction = asserts.assert_normalized(direction)

    vec = point - origin
    dot = vector.dot(vec, direction)
    vec -= dot * direction
    return tf.norm(tensor=vec, axis=-1, keepdims=keepdims)


def project_to_ray(point, origin, direction, name=None):
  """Computes the projection of a M-d point on a M-d ray.

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    point: A tensor of shape `[A1, ..., An, M]`.
    origin: A tensor of shape `[A1, ..., An, M]`.
    direction: A tensor of shape `[A1, ..., An, M]`. The last dimension must be
      normalized.
    name: A name for this op. Defaults to "point_project_to_ray".

  Returns:
    A tensor of shape `[A1, ..., An, M]` containing the projected point.

  Raises:
    ValueError: If the shape of `point`, `origin`, or 'direction' is not
    supported.
  """
  with tf.compat.v1.name_scope(name, "point_project_to_ray",
                               [point, origin, direction]):
    point = tf.convert_to_tensor(value=point)
    origin = tf.convert_to_tensor(value=origin)
    direction = tf.convert_to_tensor(value=direction)

    shape.compare_dimensions((point, origin, direction), -1,
                             ("point", "origin", "direction"))
    shape.compare_batch_dimensions(
        tensors=(point, origin, direction),
        last_axes=-2,
        broadcast_compatible=True)
    direction = asserts.assert_normalized(direction)

    vec = point - origin
    dot = vector.dot(vec, direction)
    return origin + dot * direction


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
