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
# See the License for the specific
"""Class to test build_grid_ds tensorflow implementation"""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud
from pylib.pc.tests import utils
from pylib.pc.custom_ops.custom_ops_tf import build_grid_ds_tf
from pylib.pc.custom_ops import compute_keys


class BuildGridDSTF(test_case.TestCase):

  @parameterized.parameters(
    (100, 32, 30, 0.1, 2),
    (200, 16, 1, 0.2, 2),
    (200, 8, 1, np.sqrt(2), 2),
    (100, 32, 30, 0.1, 3),
    (200, 16, 1, 0.2, 3),
    (200, 8, 1, np.sqrt(3), 3),
    (100, 32, 30, 0.1, 4),
    (200, 16, 1, 0.2, 4),
    (200, 8, 1, np.sqrt(4), 4)
  )
  def test_grid_datastructure(self,
                              num_points,
                              batch_size,
                              scale,
                              radius,
                              dimension):
    radius = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=True)
    point_cloud = PointCloud(points, batch_ids)
    #Compute the number of cells in the grid.
    aabb = point_cloud.get_AABB()
    aabb_sizes = aabb._aabb_max - aabb._aabb_min
    batch_num_cells = tf.cast(
        tf.math.ceil(aabb_sizes / radius), tf.int32)
    total_num_cells = tf.maximum(
        tf.reduce_max(batch_num_cells, axis=0), 1)
    keys = compute_keys(point_cloud, total_num_cells, radius)
    keys = tf.sort(keys, direction='DESCENDING')
    ds_tf = build_grid_ds_tf(keys, total_num_cells, batch_size)

    keys = keys.numpy()
    ds_numpy = np.full((batch_size, total_num_cells[0],
                        total_num_cells[1], 2), 0)
    if dimension == 2:
      cells_per_2D_cell = 1
    elif dimension > 2:
      cells_per_2D_cell = np.prod(total_num_cells[2:])
    for key_iter, key in enumerate(keys):
      curDSIndex = key // cells_per_2D_cell
      yIndex = curDSIndex % total_num_cells[1]
      auxInt = curDSIndex // total_num_cells[1]
      xIndex = auxInt % total_num_cells[0]
      curbatch_ids = auxInt // total_num_cells[0]

      if key_iter == 0:
        ds_numpy[curbatch_ids, xIndex, yIndex, 0] = key_iter
      else:
        prevKey = keys[key_iter - 1]
        prevDSIndex = prevKey // cells_per_2D_cell
        if prevDSIndex != curDSIndex:
            ds_numpy[curbatch_ids, xIndex, yIndex, 0] = key_iter

      nextIter = key_iter + 1
      if nextIter >= len(keys):
        ds_numpy[curbatch_ids, xIndex, yIndex, 1] = len(keys)
      else:
        nextKey = keys[key_iter + 1]
        nextDSIndex = nextKey // cells_per_2D_cell
        if nextDSIndex != curDSIndex:
          ds_numpy[curbatch_ids, xIndex, yIndex, 1] = key_iter + 1

    # check if the data structure is equal
    self.assertAllEqual(ds_tf, ds_numpy)

if __name__ == '__main__':
  test_case.main()
