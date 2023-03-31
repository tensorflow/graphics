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
import tensorflow as tf

from pylib.pc.custom_ops import compute_keys, build_grid_ds
from pylib.pc import PointCloud, AABB


class Grid:
  """ 2D regular grid of a point cloud.

  Args:
    point_cloud : A `PointCloud` instance to distribute in the grid.
    cell_sizes A `float` `Tensor` of shape `[D]`, the sizes of the grid
      cells in each dimension.
    aabb: An `AABB` instance, the bounding box of the grid, if `None`
      the bounding box of `point_cloud` is used. (optional)

  """

  def __init__(self, point_cloud: PointCloud, cell_sizes, aabb=None,
               name=None):
    cell_sizes = tf.cast(tf.convert_to_tensor(value=cell_sizes),
                         tf.float32)
    if cell_sizes.shape == [] or cell_sizes.shape[0] == 1:
      cell_sizes = tf.repeat(cell_sizes, point_cloud._dimension)
    #Save the attributes.
    self._batch_size = point_cloud._batch_size_numpy
    self._cell_sizes = cell_sizes
    self._point_cloud = point_cloud
    self._aabb = point_cloud.get_AABB()
    #Compute the number of cells in the grid.
    aabb_sizes = self._aabb._aabb_max - self._aabb._aabb_min
    batch_num_cells = tf.cast(
        tf.math.ceil(aabb_sizes / self._cell_sizes), tf.int32)
    self._num_cells = tf.maximum(
        tf.reduce_max(batch_num_cells, axis=0), 1)

    #Compute the key for each point.
    self._cur_keys = compute_keys(
        self._point_cloud, self._num_cells,
        self._cell_sizes)

    #Sort the keys.
    self._sorted_indices = tf.argsort(
        self._cur_keys, direction='DESCENDING')
    self._sorted_keys = tf.gather(self._cur_keys, self._sorted_indices)

    #Get the sorted points and batch ids.
    self._sorted_points = tf.gather(
        self._point_cloud._points, self._sorted_indices)
    self._sorted_batch_ids = tf.gather(
        self._point_cloud._batch_ids, self._sorted_indices)

    self._fast_DS = None

  def get_DS(self):
    """ Method to get the 2D-Grid datastructure.

    Note: By default the data structure is not build on initialization,
    but with this method

    Returns:
      A `int` `Tensor` of shape `[num_cells[0], num_cells[1], 2]`, where
      `[i,j,0]:[i,j,1]` is the range of points in cell `i,j`.
      The indices are with respect to the sorted points of the grid.

    """
    if self._fast_DS is None:
      #Build the fast access data structure.
      self._fast_DS = build_grid_ds(
          self._sorted_keys, self._num_cells, self._batch_size)
    return self._fast_DS
