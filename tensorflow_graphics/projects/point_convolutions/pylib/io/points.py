# Copyright 2020 Google LLC
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
# Lint as: python3
"""Small functions to load point clouds"""

import os
import tensorflow as tf
import numpy as np
from tensorflow_graphics.io.triangle_mesh import load as load_mesh


def load_points_from_file_to_tensor(filename,
                                    delimiter=',',
                                    dimension=3,
                                    dtype=tf.float32):
  """ Loads point clouds with features from ASCII files using tf.io.gfile.GFile()

  Args:
    filename: A `string` path to the file.
    delimiter: A `string` delimiter that separates the points in the file.
    dimension: An `int` `D1` , the first `D1` elements in each line are
      treated as point coordinates, the rest as features.
    dtype: A `tf.dtype` of the output tensors

  Returns:
    points: A `Tensor` of shape `[N, D1]` and type dtype
    features: A `Tensor` of shape `[N, D2]` and type dtype

  Raises:
    TypeError: if filename is not of type 'string'
    FileNotFoundError: if filename does not exist

  """

  points = []
  features = []
  if isinstance(filename, str):
    if tf.io.gfile.exists(filename):
      with tf.io.gfile.GFile(filename, 'r') as in_file:
        for line in in_file:
          line_elements = line[:-1].split(delimiter)
          points.append(line_elements[0:dimension])
          if len(line_elements) > 3:
            features.append(line_elements[dimension:])

      points = tf.convert_to_tensor(value=points)
      features = tf.convert_to_tensor(value=features)
      points = tf.strings.to_number(input=points, out_type=dtype)
      features = tf.strings.to_number(input=features, out_type=dtype)
      return points, features
    else:
      raise FileNotFoundError(f"No such file: {filename}")
  else:
    raise TypeError("'filename' must be of type 'string'")


def load_points_from_file_to_numpy(filename,
                                   delimiter=',',
                                   dimension=3,
                                   max_num_points=None,
                                   dtype=np.float32):
  """ Loads point clouds with features from ASCII files using tf.io.gfile.GFile()

  Args:
    filename: `string` path to the file
    delimiter: `string` delimiter that separates the points in the file
    dimension: `int` D1 , the first D1 elements in each line are treated as
      point coordinates, the rest as features
    max_num_points: An `int` the maximum number of lines to read.
    dtype: `np.dtype` of the output array

  Returns:
    points: A numpy array of shape [N,D1] and type dtype
    features: A numpy array of shape [N,D2] and type dtype

  Raises:
    TypeError: if filename is not of type 'string'
    FileNotFoundError: if filename does not exist

  """

  points = []
  features = []
  if isinstance(filename, str):
    if tf.io.gfile.exists(filename):
      with tf.io.gfile.GFile(filename, 'r') as in_file:
        i = 0
        for line in in_file:
          if max_num_points is not None and i < max_num_points:
            line_elements = line[:-1].split(delimiter)
            points.append(line_elements[0:dimension])
            if len(line_elements) > 3:
              features.append(line_elements[dimension:])
          else:
            break
          i += 1

      points = np.array(points, dtype=dtype)
      features = np.array(features, dtype=dtype)
      return points, features
    else:
      raise FileNotFoundError(f"No such file: {filename}")
  else:
    raise TypeError("'filename' must be of type 'string'")


def load_batch_of_points(filenames,
                         batch_shape=[-1],
                         delimiter=',',
                         point_dimension=3,
                         dtype=tf.float32):
  """ Loads a batch of point clouds form the given ASCII files and creates
  zero padded `Tensor` for the point coordinates and the features and a `sizes`
  tensor with the number of points per point cloud.

  Args:
    filenames: `list` of `string`, the paths to the files.
    batch_shape: A 1D `int` `Tensor`, `[A1, ..., An]`.
    delimiter: A `string` delimiter that separates the points in the file.
    point_dimension: An `int` `D1` , the first `D1` elements in each line are
      treated as point coordinates, the rest as features.
    dtype: A `tf.dtype` of the output.

  Returns:
    points: A `Tensor`of shape `[A1, ..., An, V, D1]` and type `dtype`.
    features: A `Tensor`of shape `[A1, ..., An, V, D2]` and type `dtype`.
    sizes: An `int` `Tensor` of shape `[A1, ... , An]`.
  """

  batch_size = len(filenames)
  if batch_shape != [-1] and tf.reduce_prod(batch_shape) != batch_size:
    raise ValueError(
        f'Invalid batch shape {batch_shape} for batch size {batch_size}')
  points = []
  features = []
  max_num_points = 0
  sizes = []
  for filename in filenames:
    curr_points, curr_features = load_points_from_file_to_tensor(
        filename=filename, delimiter=delimiter, dtype=dtype)
    points.append(curr_points)
    features.append(curr_features)
    sizes.append(len(curr_points))

  sizes = tf.convert_to_tensor(value=sizes)
  max_num_points = tf.reduce_max(sizes)

  feature_dimension = features[0].shape[1]
  for i in range(batch_size):
    pad_size = max_num_points - sizes[i]
    points[i] = tf.concat(
        (points[i], tf.zeros(shape=[pad_size, point_dimension], dtype=dtype)),
        axis=0)
    features[i] = tf.concat(
        (features[i], tf.zeros(shape=[pad_size, feature_dimension],
                               dtype=dtype)), axis=0)

  points = tf.stack(values=points, axis=0)
  features = tf.stack(values=features, axis=0)

  points = tf.reshape(
      tensor=points, shape=batch_shape + [max_num_points, point_dimension])
  features = tf.reshape(
      tensor=features, shape=batch_shape + [max_num_points, feature_dimension])
  sizes = tf.reshape(tensor=sizes, shape=batch_shape)

  return points, features, sizes


def load_batch_of_meshes(filenames,
                         batch_shape=[-1],
                         file_type=None,
                         dtype=tf.float32,
                         **kwargs):
  """ Loads a batch of point clouds from the given mesh files and creates a
  zero padded `Tensor` for the point coordinates and a `sizes` tensor with the
  number of points per point cloud.

  Args:
    filenames: A `list` of `strings`, the paths to the files.
    batch_shape: A 1D `int` `Tensor` `[A1, ..., An]`
    filename: A `string`, the path to the file.
    file_type: A `string` specifying the type of the file (e.g. 'obj', 'stl').
      If not specified the file_type will be inferred from the file name.
    **kwargs: Additional arguments that should be passed to trimesh.load().
    dtype: A `tf.dtype` of the output tensors

  Returns:
    points: A `Tensor`of shape `[A1, ..., An, V, 3]` and type `dtype`.
    sizes: A int `Tensor`of shape `[A1, ..., An]`.
  """
  batch_size = len(filenames)
  points = []
  sizes = []
  for filename in filenames:
    mesh = load_mesh(filename, file_type=file_type, **kwargs)
    curr_points = tf.convert_to_tensor(value=mesh.vertices, dtype=dtype)
    points.append(curr_points)
    sizes.append(len(curr_points))
    tf.convert_to_tensor(sizes)
  max_num_points = tf.reduce_max(sizes)

  for i in range(batch_size):
    pad_size = max_num_points - sizes[i]
    points[i] = tf.concat(
        (points[i], tf.zeros(shape=[pad_size, 3], dtype=dtype)), axis=0)
  points = tf.stack(values=points, axis=0)
  points = tf.reshape(tensor=points, shape=batch_shape + [max_num_points, 3])
  sizes = tf.reshape(tensor=sizes, shape=batch_shape)

  return points, sizes
