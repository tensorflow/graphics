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
"""Utility function to pad and pack lists of tensors to a common shape."""

import tensorflow as tf

from tensorflow_graphics.util import shape


def pad_list(tensors, mode='CONSTANT', constant_values=0):
  """
  Pads stacks and computes sizes of unpadded tensors in a list.

  Note:
    In the following, A1 to An are optional batch dimensions.

    If tensors are already of same shape, sizes are assumed to be the along the
    first dimension of each tensor. E.g., if there are two tensors of shape
    [4,3] to be stacked and padded, the resulting tensor will be of shape
    [2, 4, 3] and sizes will be [4, 4].

    If only one tensor is provided, it will also contain an additional batch
    dimension of size 1.

    Example:
      ```python
          t = [tf.constant([]),
              tf.constant([1.]),
              tf.constant([2., 2.]),
              tf.constant([3., 3., 3.])]
          # 'constant_values' is 0.
          # rank of 't' is 2.
            pad_list(t)   # Result (added padding in asterisk)(
                          # tf.Tensor([[*0.*, *0.*, *0.*],
                          #            [1.,   *0.*, *0.*],
                          #            [2.,    2.,  *0.*],
                          #            [3.,    3.,   3.]]),
                          #  tf.Tensor([0, 1, 2, 3])
                          # )
      ```

  Args:
    tensors: list of float32 tensors of shape `[A1,...,An, D]`, where one of the
      batch dimensions differs for each tensor.
    mode: One of "CONSTANT", "REFLECT", or "SYMMETRIC" (case-insensitive).
    constant_values: In "CONSTANT" mode, the scalar pad value to use.
      Must be same type as tensor.

  Returns:
    float32 tensor of shape `[A1,...,An, D]` with the padded and stacked values.
    and an int32 tensor denoting the sizes of the unpadded arrays.

  Raises:
    ValueError: If tensors are not of same rank or have more than 1 unequal
      dimensions.
  """
  if not len(tensors) > 1:  # Todo: Fix for 1D tensors
    return tf.expand_dims(tensors[0], 0), tf.constant([len(tensors[0])])

  # check if all tensors have same rank
  tensor_rank = {tf.rank(tensor).numpy() for tensor in tensors}
  if not len(tensor_rank) == 1:
    raise ValueError('All tensors need to have same rank.')

  if tensor_rank.pop() > 1:
    shape.compare_dimensions(tensors, -1)

  shapes = tf.stack([tensor.shape for tensor in tensors], axis=0)
  padding_dim = tf.where(tf.reduce_max(shapes, 0) != tf.reduce_min(shapes, 0))
  if len(padding_dim) > 1:
    raise ValueError('Only one dimension with unequal length supported.')

  # all tensors are already of same shape
  if len(padding_dim) == 0:
    return tf.stack(tensors, 0), tf.repeat(tensors[0].shape[-0], [len(tensors)])

  padding_dim = tf.squeeze(padding_dim)

  sizes = shapes[:, padding_dim]
  max_length = max(sizes.numpy())

  padded = []
  for i, tensor in enumerate(tensors):
    pad_length = [0, max_length - sizes[i]]
    padding = tf.scatter_nd([[padding_dim]], [pad_length], [tf.rank(tensor), 2])
    padded.append(tf.pad(tensor, padding, mode, constant_values))

  return tf.stack(padded, 0), sizes
