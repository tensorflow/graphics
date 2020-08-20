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
''' helper functions for point clouds '''

import tensorflow as tf
from tensorflow_graphics.geometry.convolution.utils import flatten_batch_to_2d
from pylib.pc import PointCloud


def check_valid_point_cloud_input(points, sizes, batch_ids):
  """Checks that the inputs to the constructor of class 'PointCloud' are valid.

  Args:
    points: A `float` `Tensor` of shape `[N, D]` or `[A1, ..., An, V, D]`.
    sizes:  An `int` `Tensor` of shape `[A1, ..., An]` or `None`.
    batch_ids: An `int` `Tensor` of shape `[N]` or `None`.

  Raises:
    Value Error: If input dimensions are invalid or no valid segmentation
      is given.

  """

  if points.shape.ndims == 2 and sizes is None and batch_ids is None:
    raise ValueError('Missing input! Either sizes or batch_ids must be given.')
  if points.shape.ndims == 1:
    raise ValueError(
        'Invalid input! Point tensor is of dimension 1 \
        but should be at least 2!')
  if points.shape.ndims == 2 and batch_ids is not None:
    if points.shape[0] != batch_ids.shape[0]:
      raise AssertionError('Invalid sizes! Sizes of points and batch_ids are' +
                           ' not equal.')


def check_valid_point_hierarchy_input(point_cloud, cell_sizes, pool_mode):
  """ Checks that inputs to the constructor of class 'PontHierarchy' are valid.

  Args:
    point_cloud: A 'PointCloud' instance.
    cell_sizes: A `list` of `float` `Tensors`.
    pool_mode: An `int`.

  Raises:
    TypeError: if input is of invalid type
    ValueError: if pool_mode is invalid, or cell_sizes dimension are invalid
      or non-positive

  """
  if not isinstance(point_cloud, (PointCloud)):
    raise TypeError('Input must be instance of class PointCloud')
  if pool_mode not in [0, 1]:
    raise ValueError('Unknown pooling mode.')
  for curr_cell_sizes in cell_sizes:
    if any(curr_cell_sizes <= 0):
      raise ValueError('cell size must be positive.')
    if not curr_cell_sizes.shape[0] in [1, point_cloud.dimension_]:
      raise ValueError(
          'Invalid number of cell sizes for point cloud' +\
          f'dimension. Must be 1 or {point_cloud.dimension_} but is' +\
          f'{curr_cell_sizes.shape[0]}.')


def _flatten_features(features, point_cloud: PointCloud):
  """ Converts features of shape `[A1, ..., An, C]` to shape `[N, C]`.

  Args:
    features: A `Tensor`.
    point_cloud: A `PointCloud` instance.

  Returns:
    A `Tensor` of shape `[N, C]`.

  """
  if features.shape.ndims > 2:
    sizes = point_cloud.get_sizes()
    features, _ = flatten_batch_to_2d(features, sizes)
    sorting = tf.math.invert_permutation(point_cloud._sorted_indices_batch)
    features = tf.gather(features, sorting)
  return features


def cast_to_num_dims(values, num_dims, dtype=tf.float32):
  """ Converts an input to the specified `dtype` and repeats it `num_dims`
  times.

  Args:
    values: Must be convertible to a `Tensor` of shape `[], [1]` or
      `[num_dims]`.
    dtype: A `tf.dtype`.

  Returns:
    A `dtype` `Tensor` of shape `[num_dims]`.

  """
  values = tf.cast(tf.convert_to_tensor(value=values),
                   dtype=dtype)
  if values.shape == [] or values.shape[0] == 1:
    values = tf.repeat(values, num_dims)
  return values
