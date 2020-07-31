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
"""
Implementation of the vert align operation for Mesh R-CNN.

This operation is also called 'perceptual feature pooling' in Wang et al.

Mesh R-CNN uses bilinear interpolation and border padding. Thus, this
implementation does not provide other algorithms.

References:
  * Georgia Gkioxari, Jitendra Malik, & Justin Johnson. (2019). Mesh R-CNN.
  * Nanyang Wang, Yinda Zhang, Zhuwen Li, Yanwei Fu, Wei Liu, & Yu-Gang Jiang.
    (2018). Pixel2Mesh: Generating 3D Mesh Models from Single RGB Images.
"""

import tensorflow as tf


def vert_align(features,
               vertices):
  """
  Sample vertex features from a feature map.

  Args:
    features: float32 tensor of shape `[N, H, W, C]` representing image features
      from which to sample the features. N is a batch dimension.
    vertices: list of N float32 tensors of shape `[V, 3]` containing the vertex
      positions for which to sample the image features.

  Returns:
    float32 tensor of shape `[N, V, C]` containing sampled features per vertex.
  """

  features = tf.convert_to_tensor(features)

  if not all([v.shape.rank == 2 for v in vertices]):
    raise ValueError('vertices should be 2 dimensional.')

  if not features._rank() == 4:
    raise ValueError('features must of shape (N, H, W, C).')
  verts_2d = [tf.cast(v[..., :2], tf.float32) for v in vertices]
  grid = _spatial_normalize_grid_to_feature(verts_2d, features)
  padded_grid, padding_offsets = pad_query_points(grid)
  print(padded_grid.shape)
  sample_grid = tf.reshape(padded_grid, (features.shape[0], 1, -1, 2))
  print(sample_grid.shape)
  sampled_features = grid_sample_2d(features, sample_grid)

  return [s_feat[:padding_offsets[i]] for i, s_feat in
          enumerate(sampled_features)]


def pad_query_points(points):
  """
  Pads and stacks query points into a tensor of shape `[N, P, 3]`

  Args:
    points: list of N float32 tensors of shape `[P, 3]` containing the
      coordinates for which to sample the image features.

  Returns:
    float32 tensor of shape `[N, P, 3]` with the padded and stacked query points
    and a list of ints denoting the lengths of the unpadded arrays.
  """
  max_num_points = len(max(points, key=len))
  padded_vertices = []
  original_lengths = []
  for point in points:
    if point.shape[-1] < 2:
      raise ValueError('points.shape[-1] has to be at least 2.')

    original_lengths.append(point.shape[0])
    pad_length = max_num_points - point.shape[0]
    pad = tf.zeros((pad_length, 2), dtype=tf.float32)
    if pad_length == 0:
      padded_vertices.append(point)
    else:
      padded_vertices.append(tf.concat([point, pad], 0))

  return tf.stack(padded_vertices), original_lengths


def _spatial_normalize_grid_to_feature(grids, feature_map):
  """
  Normalize batch of 2D coordinates (x, y) such that (-1, -1) corresponds to
  top-left and (+1, +1) to bottom-right location in the input feature map.

  Args:
    grids: list of `N1` float32 tensors of shape `[N2,2]` containing the 2D
    coordinates that should be normalized.
    feature_map: float32 tensor of shape `[N1, H, W, C]` that is used to normalize
      spatial dimension leaving the channels C untouched.

  Returns:
    list with `N1` float32 tensors of shape `[N2,2]` with normalized coordinates.
  """
  if not all([g.shape[-1] == 2 for g in grids]):
    raise ValueError('Grid must contain 2D coordinats.')
  if feature_map._rank() != 4:
    raise ValueError('feature_map must be a tensor of rank 4.')
  normalized_grids = []
  for i, grid in enumerate(grids):
    H, W, _ = feature_map[i].shape
    extent = tf.constant([H / 2, W / 2], dtype=tf.float32)
    normalized_grids.append((grid - extent) / extent)
  return normalized_grids


def grid_sample_2d(source, grid):
  """
  Performs 2D grid sampling with bilinear interpolation of the source according
  to normalized coordinates provided by the grid.

  Args:
    source: float32 tensor of shape `[N, H, W, C]` representing the source
      from which to sample.
    grid: float32 tensor of shape `[N, X, Y]` containing the samling locations.

  Returns:
    float32 tensor of shape `[N, X, Y, C]` containing the result of the
      interpolation.
  """

  N, H, W, _ = source.shape

  # Find interpolation neighbours
  y, x = grid[..., 0], grid[..., 1]
  y = tf.cast(H - 1, grid.dtype) * (y + 1) / 2
  x = tf.cast(W - 1, grid.dtype) * (x + 1) / 2
  low = tf.maximum(tf.cast(tf.floor(y), tf.int32), 0)
  high = tf.minimum(low + 1, H - 1)
  right = tf.maximum(tf.cast(tf.floor(x), tf.int32), 0)
  left = tf.minimum(right + 1, W - 1)

  # Gather pixel values
  index = tf.tile(tf.range(N)[:, None, None],
                  tf.concat([[1], tf.shape(y)[1:]], axis=0))
  value_bottom_right = tf.gather_nd(source,
                                    tf.stack([index, low, right], axis=-1))
  value_bottom_left = tf.gather_nd(source,
                                   tf.stack([index, low, left], axis=-1))
  value_top_right = tf.gather_nd(source,
                                 tf.stack([index, high, right], axis=-1))
  value_top_left = tf.gather_nd(source, tf.stack([index, high, left], axis=-1))

  # Interpolation coefficients
  d_y = tf.cast(y, source.dtype) - tf.cast(low, source.dtype)
  d_y = tf.expand_dims(d_y, -1)
  d_x = tf.cast(x, source.dtype) - tf.cast(right, source.dtype)
  d_x = tf.expand_dims(d_x, -1)

  # Compute bilinear interpolation
  inter_y_right = value_bottom_right * (1 - d_y) + value_top_right * d_y
  inter_y_left = value_bottom_left * (1 - d_y) + value_top_left * d_y
  interpolated = inter_y_right * (1 - d_x) + inter_y_left * d_x

  return interpolated
