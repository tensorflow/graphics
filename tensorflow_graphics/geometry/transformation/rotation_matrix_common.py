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
"""This module contains routines shared for rotation matrices."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def is_valid(matrix, atol=1e-3, name=None):
  r"""Determines if a matrix in K-dimensions is a valid rotation matrix.

  Determines if a matrix $$\mathbf{R}$$ is a valid rotation matrix by checking
  that $$\mathbf{R}^T\mathbf{R} = \mathbf{I}$$ and $$\det(\mathbf{R}) = 1$$.

  Note: In the following, A1 to An are optional batch dimensions.

  Args:
    matrix: A tensor of shape `[A1, ..., An, K, K]`, where the last two
      dimensions represent a rotation matrix in K-dimensions.
    atol: The absolute tolerance parameter.
    name: A name for this op that defaults to "rotation_matrix_common_is_valid".

  Returns:
    A tensor of type `bool` and shape `[A1, ..., An, 1]` where False indicates
    that the input is not a valid rotation matrix.
  """
  with tf.compat.v1.name_scope(name, "rotation_matrix_common_is_valid",
                               [matrix]):
    matrix = tf.convert_to_tensor(value=matrix)

    shape.check_static(
        tensor=matrix, tensor_name="matrix", has_rank_greater_than=1)
    shape.compare_dimensions(
        tensors=(matrix, matrix),
        tensor_names=("matrix", "matrix"),
        axes=(-1, -2))

    distance_to_unit_determinant = tf.abs(tf.linalg.det(matrix) - 1.)
    # Computes how far the product of the transposed rotation matrix with itself
    # is from the identity matrix.
    ndims = matrix.shape.ndims
    permutation = list(range(ndims - 2)) + [ndims - 1, ndims - 2]
    identity = tf.eye(
        tf.compat.v1.dimension_value(matrix.shape[-1]), dtype=matrix.dtype)
    difference_to_identity = tf.matmul(
        tf.transpose(a=matrix, perm=permutation), matrix) - identity
    norm_diff = tf.norm(tensor=difference_to_identity, axis=(-2, -1))
    # Computes the mask of entries that satisfies all conditions.
    mask = tf.logical_and(distance_to_unit_determinant < atol, norm_diff < atol)
    output = tf.compat.v1.where(
        mask, tf.ones_like(distance_to_unit_determinant, dtype=bool),
        tf.zeros_like(distance_to_unit_determinant, dtype=bool))
    return tf.expand_dims(output, axis=-1)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
