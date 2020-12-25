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
# Lint as: python3
"""Code for performing regular grid interpolation for TF.

Works for arbitrary grid dimension. Only implemented linear interpolation.
"""
import tensorflow.compat.v1 as tf


def get_interp_coefficients(grid,
                            pts,
                            min_grid_value=(0, 0, 0),
                            max_grid_value=(1, 1, 1)):
  """Regular grid interpolator, returns inpterpolation coefficients.

  Args:
    grid: `[batch_size, *size, features]` tensor, input feature grid.
    pts: `[batch_size, num_points, dim]` tensor, coordinates of points that
    in each dim are within the range (min_grid_value[dim], max_grid_value[dim]).
    min_grid_value: tuple, minimum value in each dimension corresponding to the
      grid.
    max_grid_value: tuple, maximum values in each dimension corresponding to the
      grid.
  Returns:
    lat: `[batch_size, num_points, 2**dim, features]` tensor, neighbor
    latent codes for each input point.
    weights: `[batch_size, num_points, 2**dim]` tensor, bi/tri-linear
    interpolation weights for each neighbor.
    xloc: `[batch_size, num_points, 2**dim, dim]`tensor, relative coordinates.

  """
  # get dimensions
  bs = grid.get_shape().as_list()[0]
  npts = tf.shape(pts)[1]
  size = tf.shape(grid)[1:-1]
  cubesize = 1.0/(tf.cast(size, tf.float32)-1.0)
  dim = len(grid.get_shape().as_list()) - 2

  # normalize coords for interpolation
  if isinstance(min_grid_value, list) or isinstance(min_grid_value, tuple):
    min_grid_value = tf.constant(min_grid_value, dtype=tf.float32)
  if isinstance(max_grid_value, list) or isinstance(min_grid_value, tuple):
    max_grid_value = tf.constant(max_grid_value, dtype=tf.float32)
  bbox = max_grid_value - min_grid_value
  pts = (pts - min_grid_value) / bbox
  pts = tf.clip_by_value(pts, 1e-6, 1-1e-6)  # clip to boundary of the bbox

  # find neighbor indices
  ind0 = tf.floor(pts / cubesize)  # `[batch_size, num_points, dim]`
  ind1 = tf.ceil(pts / cubesize)  # `[batch_size, num_points, dim]`
  ind01 = tf.stack([ind0, ind1], axis=0)  # `[2, batch_size, num_points, dim]`
  ind01 = tf.transpose(ind01, perm=[0, 3, 1, 2])  # `[2, d, b, n]`
  ind01 = tf.cast(ind01, tf.int32)

  # generate combinations for enumerating neighbors
  tmp = tf.constant([0, 1], dtype=tf.int32)
  com_ = tf.stack(tf.meshgrid(*tuple([tmp] * dim), indexing="ij"),
                  axis=-1)
  com_ = tf.reshape(com_, [-1, dim])  # `[2**dim, dim]`
  dim_ = tf.reshape(tf.range(dim), [1, -1])
  dim_ = tf.tile(dim_, [2**dim, 1])  # `[2**dim, dim]`
  gather_ind = tf.stack([com_, dim_], axis=-1)  # `[2**dim, dim, 2]`
  gather_ind_ = tf.stack([1-com_, dim_], axis=-1)  # `[2**dim, dim, 2]`
  ind_ = tf.gather_nd(ind01, gather_ind)  # [2**dim, dim, batch_size, num_pts]
  ind_n = tf.transpose(ind_, perm=[2, 3, 0, 1])  # neighbor indices
  # `[batch_size, num_pts, 2**dim, dim]`
  ind_b = tf.reshape(tf.range(bs), [-1, 1, 1, 1])
  ind_b = tf.broadcast_to(ind_b, [bs, npts, 2**dim, 1])  # dummy batch indices
  # `[batch_size, num_pts, 2**dim, 1]`
  gather_ind2 = tf.concat([ind_b, ind_n], axis=-1)
  lat = tf.gather_nd(grid, gather_ind2)
  # `[batch_size, num_points, 2**dim, in_features]`

  # weights of neighboring nodes
  xyz0 = ind0 * cubesize  # `[batch_size, num_points, dim]`
  xyz1 = (ind0 + 1) * cubesize  # `[batch_size, num_points, dim]`
  xyz01 = tf.stack([xyz0, xyz1], axis=-1)  # [batch_size, num_points, dim, 2]`
  xyz01 = tf.transpose(xyz01, perm=[3, 2, 0, 1])  # [2, d, batch, npts]
  pos = tf.gather_nd(xyz01, gather_ind)  # `[2**dim, dim, batch, num_points]`
  pos = tf.transpose(pos, perm=[2, 3, 0, 1])
  pos_ = tf.gather_nd(xyz01, gather_ind_)  # [2**dim, dim, batch, num_points]`
  pos_ = tf.transpose(pos_, perm=[2, 3, 0, 1])
  # `[batch_size, num_points, 2**dim, dim]`

  dxyz_ = tf.abs(tf.expand_dims(pts, -2) - pos_) / cubesize
  weights = tf.reduce_prod(dxyz_, axis=-1, keepdims=False)
  # `[batch_size, num_points, 2**dim]
  xloc = (tf.expand_dims(pts, -2) - pos) / cubesize
  # `[batch, num_points, 2**dim, dim]`
  return lat, weights, xloc


def regular_grid_interpolation(grid,
                               pts,
                               min_grid_value=(0, 0, 0),
                               max_grid_value=(1, 1, 1)):
  """Regular grid interpolator, returns inpterpolation values.

  Args:
    grid: `[batch_size, *size, features]` tensor, input feature grid.
    pts: `[batch_size, num_points, dim]` tensor, coordinates of points that
    in each dim are within the range (min_grid_value[dim], max_grid_value[dim]).
    min_grid_value: tuple, minimum value in each dimension corresponding to the
      grid.
    max_grid_value: tuple, maximum values in each dimension corresponding to the
      grid.
  Returns:
    vals: `[batch_size, num_points, features]` tensor, values
  """
  lats, weights, _ = get_interp_coefficients(grid, pts, min_grid_value,
                                             max_grid_value)
  vals = tf.reduce_sum(lats * weights[..., tf.newaxis], axis=-2)
  return vals
