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
"""Tensorflow grid utility functions."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _grid(starts, stops, nums):
  """Generates a M-D uniform axis-aligned grid.

  Warning:
    This op is not differentiable. Indeed, the gradient of tf.linspace and
    tf.meshgrid are currently not defined.

  Args:
    starts: A tensor of shape `[M]` representing the start points for each
      dimension.
    stops: A tensor of shape `[M]` representing the end points for each
      dimension.
    nums: A tensor of shape `[M]` representing the number of subdivisions for
      each dimension.

  Returns:
    A tensor of shape `[nums[0], ..., nums[M-1], M]` containing an M-D uniform
      grid.
  """
  params = [tf.unstack(tensor) for tensor in [starts, stops, nums]]
  layout = [tf.linspace(*param) for param in zip(*params)]
  return tf.stack(tf.meshgrid(*layout, indexing="ij"), axis=-1)


def generate(starts, stops, nums, name="grid_generate"):
  r"""Generates a M-D uniform axis-aligned grid.

  Warning:
    This op is not differentiable. Indeed, the gradient of tf.linspace and
    tf.meshgrid are currently not defined.

  Note:
    In the following, `B` is an optional batch dimension.

  Args:
    starts: A tensor of shape `[M]` or `[B, M]`, where the last dimension
      represents a M-D start point.
    stops: A tensor of shape `[M]` or `[B, M]`, where the last dimension
      represents a M-D end point.
    nums: A tensor of shape `[M]` representing the number of subdivisions for
      each dimension.
    name: A name for this op. Defaults to "grid_generate".

  Returns:
    A tensor of shape `[nums[0], ..., nums[M-1], M]` containing an M-D uniform
      grid or a tensor of shape `[B, nums[0], ..., nums[M-1], M]` containing B
      M-D uniform grids. Please refer to the example below for more details.

  Raises:
    ValueError: If the shape of `starts`, `stops`, or `nums` is not supported.

  Examples:
    ```python
    print(generate((-1.0, -2.0), (1.0, 2.0), (3, 5)))
    >>> [[[-1. -2.]
          [-1. -1.]
          [-1.  0.]
          [-1.  1.]
          [-1.  2.]]
         [[ 0. -2.]
          [ 0. -1.]
          [ 0.  0.]
          [ 0.  1.]
          [ 0.  2.]]
         [[ 1. -2.]
          [ 1. -1.]
          [ 1.  0.]
          [ 1.  1.]
          [ 1.  2.]]]
    ```
    Generates a 3x5 2d grid from -1.0 to 1.0 with 3 subdivisions for the x
    axis and from -2.0 to 2.0 with 5 subdivisions for the y axis. This lead to a
    tensor of shape (3, 5, 2).
  """
  with tf.name_scope(name):
    starts = tf.convert_to_tensor(value=starts)
    stops = tf.convert_to_tensor(value=stops)
    nums = tf.convert_to_tensor(value=nums)

    shape.check_static(
        tensor=starts,
        tensor_name="starts",
        has_rank_greater_than=0,
        has_rank_less_than=3)
    shape.check_static(
        tensor=stops,
        tensor_name="stops",
        has_rank_greater_than=0,
        has_rank_less_than=3)
    shape.check_static(tensor=nums, tensor_name="nums", has_rank=1)
    shape.compare_batch_dimensions(
        tensors=(starts, stops), last_axes=(-1, -1), broadcast_compatible=False)
    shape.compare_dimensions((starts, stops, nums), -1,
                             ("starts", "stops", "nums"))

    if starts.shape.ndims == 1:
      return _grid(starts, stops, nums)
    else:
      return tf.stack([
          _grid(starts, stops, nums)
          for starts, stops in zip(tf.unstack(starts), tf.unstack(stops))
      ])


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
