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
"""Tensorflow triangle utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.math import vector
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def normal(v0: type_alias.TensorLike,
           v1: type_alias.TensorLike,
           v2: type_alias.TensorLike,
           clockwise: bool = False,
           normalize: bool = True,
           name: str = "triangle_normal") -> tf.Tensor:
  """Computes face normals (triangles).

  Note:
    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    v0: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vertex of a triangle.
    v1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vertex of a triangle.
    v2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the third vertex of a triangle.
    clockwise: Winding order to determine front-facing triangles.
    normalize: A `bool` indicating whether output normals should be normalized
      by the function.
    name: A name for this op. Defaults to "triangle_normal".

  Returns:
    A tensor of shape `[A1, ..., An, 3]`, where the last dimension represents
      a normalized vector.

  Raises:
    ValueError: If the shape of `v0`, `v1`, or `v2` is not supported.
  """
  with tf.name_scope(name):
    v0 = tf.convert_to_tensor(value=v0)
    v1 = tf.convert_to_tensor(value=v1)
    v2 = tf.convert_to_tensor(value=v2)

    shape.check_static(tensor=v0, tensor_name="v0", has_dim_equals=(-1, 3))
    shape.check_static(tensor=v1, tensor_name="v1", has_dim_equals=(-1, 3))
    shape.check_static(tensor=v2, tensor_name="v2", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(v0, v1, v2), last_axes=-2, broadcast_compatible=True)

    normal_vector = vector.cross(v1 - v0, v2 - v0, axis=-1)
    normal_vector = asserts.assert_nonzero_norm(normal_vector)
    if not clockwise:
      normal_vector *= -1.0
    if normalize:
      return tf.nn.l2_normalize(normal_vector, axis=-1)
    return normal_vector


def area(v0: type_alias.TensorLike,
         v1: type_alias.TensorLike,
         v2: type_alias.TensorLike,
         name: str = "triangle_area") -> tf.Tensor:
  """Computes triangle areas.

    Note: Computed triangle area = 0.5 * | e1 x e2 | where e1 and e2 are edges
      of triangle. A degenerate triangle will return 0 area, whereas the normal
      for a degenerate triangle is not defined.


    In the following, A1 to An are optional batch dimensions, which must be
    broadcast compatible.

  Args:
    v0: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the first vertex of a triangle.
    v1: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the second vertex of a triangle.
    v2: A tensor of shape `[A1, ..., An, 3]`, where the last dimension
      represents the third vertex of a triangle.
    name: A name for this op. Defaults to "triangle_area".

  Returns:
    A tensor of shape `[A1, ..., An, 1]`, where the last dimension represents
      a normalized vector.
  """
  with tf.name_scope(name):
    v0 = tf.convert_to_tensor(value=v0)
    v1 = tf.convert_to_tensor(value=v1)
    v2 = tf.convert_to_tensor(value=v2)

    shape.check_static(tensor=v0, tensor_name="v0", has_dim_equals=(-1, 3))
    shape.check_static(tensor=v1, tensor_name="v1", has_dim_equals=(-1, 3))
    shape.check_static(tensor=v2, tensor_name="v2", has_dim_equals=(-1, 3))
    shape.compare_batch_dimensions(
        tensors=(v0, v1, v2), last_axes=-2, broadcast_compatible=True)

    normals = vector.cross(v1 - v0, v2 - v0, axis=-1)
    return 0.5 * tf.linalg.norm(tensor=normals, axis=-1, keepdims=True)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
