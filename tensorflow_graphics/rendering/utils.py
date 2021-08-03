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
"""Various util functions common for all rasterizers."""

from typing import List, Optional
import tensorflow as tf

from tensorflow_graphics.util import type_alias


def dim_value(dim: Optional[int] = None) -> int:
  return 1 if dim is None else tf.compat.dimension_value(dim)


def merge_batch_dims(tensor: type_alias.TensorLike,
                     last_axis: int) -> type_alias.TensorLike:
  """Merges all dimensions into one starting from 0 till `last_axis` exluding."""
  return tf.reshape(tensor, [-1] + tensor.shape.as_list()[last_axis:])


def restore_batch_dims(tensor: type_alias.TensorLike,
                       batch_shape: List[int]) -> type_alias.TensorLike:
  """Unpack first dimension into batch_shape, preserving the rest of the dimensions."""
  return tf.reshape(tensor, batch_shape + tensor.shape.as_list()[1:])


def transform_homogeneous(matrices: type_alias.TensorLike,
                          vertices: type_alias.TensorLike) -> tf.Tensor:
  """Applies 4x4 homogenous matrix transformations to xyz vertices.

  The vertices are input and output as as row-major, but are interpreted as
  column vectors multiplied on the right-hand side of the matrices. More
  explicitly, this function computes (MV^T)^T where M represents transformation
  matrices and V stands for vertices.
  Since input vertices are xyz they are extended to xyzw with w=1.

  Args:
    matrices: A tensor of shape `[batch, 4, 4]` containing batches of view
      projection matrices.
    vertices: A tensor of shape `[batch, num_vertices, 3]` containing batches of
      vertices, each defined by a 3D point.

  Returns:
    A [batch, N, 4] Tensor of xyzw vertices.
  """
  homogeneous_coord = tf.ones_like(vertices[..., 0:1])
  vertices = tf.concat([vertices, homogeneous_coord], -1)

  return tf.matmul(vertices, matrices, transpose_b=True)
