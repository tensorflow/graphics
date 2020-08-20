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
"""Class to represent a point cloud hierarchy."""

import numpy as np
import tensorflow as tf

from pylib.pc.utils import check_valid_point_hierarchy_input

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import Neighborhood
from pylib.pc import sample
from pylib.pc.utils import cast_to_num_dims


class PointHierarchy:
  """ A hierarchy of sampled point clouds.

  Args:
    point_cloud: A `PointCloud` instance..
    cell_sizes: A list of  `floats` or `float` `Tensors` of shape `[D]`,
      the cell sizes for the sampling. The length of the list defines
      the number of samplings.
    sample_mode: A `string`, either `'poisson'`or `'cell average'`.

  """

  def __init__(self,
               point_cloud: PointCloud,
               cell_sizes,
               sample_mode='poisson',
               name=None):
    with tf.compat.v1.name_scope(
        name, "hierarchical point cloud constructor",
        [self, point_cloud, cell_sizes, sample_mode]):

      # check_valid_point_hierarchy_input(point_cloud,cell_sizes,sample_mode)

      #Initialize the attributes.
      self._aabb = point_cloud.get_AABB()
      self._point_clouds = [point_cloud]
      self._cell_sizes = []
      self._neighborhoods = []

      self._dimension = point_cloud._dimension
      self._batch_shape = point_cloud._batch_shape

      #Create the different sampling operations.
      cur_point_cloud = point_cloud
      for sample_iter, cur_cell_sizes in enumerate(cell_sizes):
        cur_cell_sizes  = tf.convert_to_tensor(
            value=cur_cell_sizes, dtype=tf.float32)

        # Check if the cell size is defined for all the dimensions.
        # If not, the last cell size value is tiled until all the dimensions
        # have a value.
        cur_num_dims = cur_cell_sizes.shape[0]
        if cur_num_dims < self._dimension:
          cur_cell_sizes = np.concatenate(
              (cur_cell_sizes, np.tile(cur_cell_sizes[-1],
               self._dimension - cur_num_dims)))
        elif cur_num_dims > self._dimension:
          raise ValueError(
              f'Too many dimensions in cell sizes {cur_num_dims} \
                instead of max. {self._dimension}')
        self._cell_sizes.append(cur_cell_sizes)

        #Create the sampling operation.
        cell_sizes_tensor = tf.convert_to_tensor(cur_cell_sizes, np.float32)

        cur_grid = Grid(cur_point_cloud, cell_sizes_tensor, self._aabb)
        cur_neighborhood = Neighborhood(cur_grid, cell_sizes_tensor)
        cur_point_cloud, _ = sample(cur_neighborhood, sample_mode)

        self._neighborhoods.append(cur_neighborhood)
        cur_point_cloud.set_batch_shape(self._batch_shape)
        self._point_clouds.append(cur_point_cloud)

  def get_points(self, batch_id=None, max_num_points=None, name=None):
    """ Returns the points.

    Note:
      In the following, A1 to An are optional batch dimensions.

      If called withoud specifying 'id' returns the points in padded format
      `[A1, ..., An, V, D]`.

    Args:
      batch_id: An `int`, identifier of point cloud in the batch, if `None`
        returns all points.

    Return:
      A list of `float` `Tensors` of shape
          `[N_i, D]`, if 'batch_id' was given
      or
        `[A1, ..., An, V_i, D]`, if no 'batch_id' was given.
    """
    with tf.compat.v1.name_scope(
        name, "get points of specific batch id", [self, batch_id]):
      points = []
      for point_cloud in self._point_clouds:
        points.append(point_cloud.get_points(batch_id))
      return points

  def get_sizes(self, name=None):
    """ Returns the sizes of the point clouds in the point hierarchy.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Returns:
      A `list` of `Tensors` of shape '`[A1, .., An]`'

    """

    with tf.compat.v1.name_scope(name, "get point hierarchy sizes", [self]):
      sizes = []
      for point_cloud in self._point_clouds:
        sizes.append(point_cloud.get_sizes())
      return sizes

  def set_batch_shape(self, batch_shape, name=None):
    """ Function to change the batch shape.

      Use this to set a batch shape instead of using 'self._batch_shape'
      to also change dependent variables.

    Note:
      In the following, A1 to An are optional batch dimensions.

    Args:
      batch_shape: An 1D `int` `Tensor` `[A1, ..., An]`.

    Raises:
      ValueError: if shape does not sum up to batch size.

    """
    with tf.compat.v1.name_scope(
        name, "set batch shape of point hierarchy", [self, batch_shape]):
      for point_cloud in self._point_clouds:
        point_cloud.set_batch_shape(batch_shape)

  def get_neighborhood(self, i=None, transposed=False):
    """ Returns the neighborhood between level `i` and `i+1` of the hierarchy.
    If called without argument returns a list of all neighborhoods.

    Args:
      i: An `int`, can be negative but must be in range
        `[-num_levels, num_levels-1]`.
      transposed: A `bool`, if `True` returns the neighborhood between
        level `i+1` and `i`.

    Returs:
      A `Neighborhood` instance or a `list` of `Neighborhood` instances.

    """
    if i is None:
      if transposed:
        return [nb.transposed() for nb in self._neighborhoods]
      else:
        return self._neighborhoods
    else:
      if transposed:
        return self._neighborhoods[i].transpose()
      else:
        return self._neighborhoods[i]

  def __getitem__(self, index):
    return self._point_clouds[index]

  def __len__(self):
    return len(self._point_clouds)
