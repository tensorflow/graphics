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
"""Methods to sample point clouds. """

import tensorflow as tf

from pylib.pc.custom_ops import sampling

from pylib.pc import PointCloud, Neighborhood, Grid
from pylib.pc.utils import cast_to_num_dims

sample_modes = {'average': 1,
                'cell average': 1,
                'cell_average': 1,
                'poisson': 0,
                'poisson disk': 0,
                'poisson_disk': 0}


def poisson_disk_sampling(point_cloud,
                          radius=None,
                          neighborhood=None,
                          return_ids=False,
                          name=None):
  """ Poisson disk sampling of a point cloud.

  Note: Either `radius` or `neighborhood` must be provided.

  Args:
    point_cloud: A `PointCloud` instance.
    radius: A `float` or a `float` `Tensor` of shape `[D]`, the radius for the
      Poisson disk sampling.
    neighborhood: A `Neighborhood` instance.
    return_ids: A `bool`, if `True` returns the indices of the sampled points.
      (optional)

    Returns:
      A `PointCloud` instance.
      An `int` `Tensor` of shape `[S]`, if `return_ids` is `True`.

    Raises:
      ValueError: If no radius or neighborhood is given.

  """
  if radius is None and neighborhood is None:
    raise ValueError(
        "Missing Argument! Either radius or neighborhood must be given!")
  if neighborhood is None:
    # compute neighborhood
    radii = cast_to_num_dims(radius, point_cloud)
    grid = Grid(point_cloud, radii)
    neighborhood = Neighborhood(grid, radii)

  #Compute the sampling.
  sampled_points, sampled_batch_ids, sampled_indices = \
      sampling(neighborhood, 1)

  sampled_point_cloud = PointCloud(
      points=sampled_points, batch_ids=sampled_batch_ids,
      batch_size=neighborhood._point_cloud_sampled._batch_size)

  if return_ids:
    sampled_indices = tf.gather(neighborhood._grid._sorted_indices,
                                sampled_indices)
    return sampled_point_cloud, sampled_indices
  else:
    return sampled_point_cloud


def cell_average_sampling(point_cloud,
                          cell_sizes=None,
                          grid=None,
                          name=None):
  """ Cell average sampling of a point cloud.

  Note: Either `cell_sizes` or `grid` must be provided.

  Args:
    point_cloud: A `PointCloud` instance.
    cell_sizes: A `float` or a `float` `Tensor` of shape `[D]`, the cell sizes
      for the sampling.
    grid: A `Grid` instance.

    Returns:
      A `PointCloud` instance.

    Raises:
      ValueError: If no radius or grid is given.

  """
  if cell_sizes is None and grid is None:
    raise ValueError(
        "Missing Argument! Either cell_sizes or grid must be given!")
  if grid is None:
    # compute grid
    cell_sizes = cast_to_num_dims(cell_sizes, point_cloud)
    grid = Grid(point_cloud, cell_sizes)

  neighborhood = Neighborhood(grid, cell_sizes)

  #Compute the sampling.
  sampled_points, sampled_batch_ids, sampled_indices = \
      sampling(neighborhood, 0)

  sampled_point_cloud = PointCloud(
      points=sampled_points, batch_ids=sampled_batch_ids,
      batch_size=neighborhood._point_cloud_sampled._batch_size)

  return sampled_point_cloud


def sample(neighborhood, sample_mode='poisson', name=None):
  """ Sampling for a neighborhood.

  Args:
    neighborhood: A `Neighborhood` instance.
    sample_mode: A `string`, either `'poisson'`or `'cell average'`.

  Returns:
    A `PointCloud` instance, the sampled points.
    An `int` `Tensor` of shape `[S]`, the indices of the sampled points,
      `None` for cell average sampling.

  """
  sample_mode_value = sample_modes[sample_mode.lower()]
  #Compute the sampling.
  sampled_points, sampled_batch_ids, sampled_indices = \
      sampling(neighborhood, sample_mode_value)

  #Save the sampled point cloud.
  if sample_mode_value == 0:
    sampled_indices = tf.gather(
        neighborhood._grid._sorted_indices, sampled_indices)
  else:
    sampled_indices = None
  sampled_point_cloud = PointCloud(
      points=sampled_points, batch_ids=sampled_batch_ids,
      batch_size=neighborhood._point_cloud_sampled._batch_size_numpy)
  return sampled_point_cloud, sampled_indices
