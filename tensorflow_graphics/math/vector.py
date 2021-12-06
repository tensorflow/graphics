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
"""Tensorflow vector utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util.type_alias import TensorLike


def cross(vector1: TensorLike,
          vector2: TensorLike,
          axis: int = -1,
          name: str = "vector_cross") -> TensorLike:
  """Computes the cross product between two tensors along an axis.

  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.

  Args:
    vector1: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    vector2: A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension
      i = axis represents a 3d vector.
    axis: The dimension along which to compute the cross product.
    name: A name for this op which defaults to "vector_cross".

  Returns:
    A tensor of shape `[A1, ..., Ai = 3, ..., An]`, where the dimension i = axis
    represents the result of the cross product.
  """
  with tf.name_scope(name):
    vector1 = tf.convert_to_tensor(value=vector1)
    vector2 = tf.convert_to_tensor(value=vector2)

    shape.check_static(
        tensor=vector1, tensor_name="vector1", has_dim_equals=(axis, 3))
    shape.check_static(
        tensor=vector2, tensor_name="vector2", has_dim_equals=(axis, 3))
    shape.compare_batch_dimensions(
        tensors=(vector1, vector2), last_axes=-1, broadcast_compatible=True)

    vector1_x, vector1_y, vector1_z = tf.unstack(vector1, axis=axis)
    vector2_x, vector2_y, vector2_z = tf.unstack(vector2, axis=axis)
    n_x = vector1_y * vector2_z - vector1_z * vector2_y
    n_y = vector1_z * vector2_x - vector1_x * vector2_z
    n_z = vector1_x * vector2_y - vector1_y * vector2_x
    return tf.stack((n_x, n_y, n_z), axis=axis)


def dot(vector1: TensorLike,
        vector2: TensorLike,
        axis: int = -1,
        keepdims: bool = True,
        name: str = "vector_dot") -> TensorLike:
  """Computes the dot product between two tensors along an axis.

  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.

  Args:
    vector1: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    vector2: Tensor of rank R and shape `[A1, ..., Ai, ..., An]`, where the
      dimension i = axis represents a vector.
    axis: The dimension along which to compute the dot product.
    keepdims: If True, retains reduced dimensions with length 1.
    name: A name for this op which defaults to "vector_dot".

  Returns:
    A tensor of shape `[A1, ..., Ai = 1, ..., An]`, where the dimension i = axis
    represents the result of the dot product.
  """
  with tf.name_scope(name):
    vector1 = tf.convert_to_tensor(value=vector1)
    vector2 = tf.convert_to_tensor(value=vector2)

    shape.compare_batch_dimensions(
        tensors=(vector1, vector2), last_axes=-1, broadcast_compatible=True)
    shape.compare_dimensions(
        tensors=(vector1, vector2),
        axes=axis,
        tensor_names=("vector1", "vector2"))

    return tf.reduce_sum(
        input_tensor=vector1 * vector2, axis=axis, keepdims=keepdims)


def reflect(vector: TensorLike,
            normal: TensorLike,
            axis: int = -1,
            name: str = "vector_reflect") -> TensorLike:
  r"""Computes the reflection direction for an incident vector.

  For an incident vector \\(\mathbf{v}\\) and normal $$\mathbf{n}$$ this
  function computes the reflected vector as
  \\(\mathbf{r} = \mathbf{v} - 2(\mathbf{n}^T\mathbf{v})\mathbf{n}\\).

  Note:
    In the following, A1 to An are optional batch dimensions, which should be
    broadcast compatible.

  Args:
    vector: A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i =
      axis represents a vector.
    normal: A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i =
      axis represents a normal around which the vector needs to be reflected.
      The normal vector needs to be normalized.
    axis: The dimension along which to compute the reflection.
    name: A name for this op which defaults to "vector_reflect".

  Returns:
    A tensor of shape `[A1, ..., Ai, ..., An]`, where the dimension i = axis
    represents a reflected vector.
  """
  with tf.name_scope(name):
    vector = tf.convert_to_tensor(value=vector)
    normal = tf.convert_to_tensor(value=normal)

    shape.compare_dimensions(
        tensors=(vector, normal), axes=axis, tensor_names=("vector", "normal"))
    shape.compare_batch_dimensions(
        tensors=(vector, normal), last_axes=-1, broadcast_compatible=True)
    normal = asserts.assert_normalized(normal, axis=axis)

    dot_product = dot(vector, normal, axis=axis)
    return vector - 2.0 * dot_product * normal


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
