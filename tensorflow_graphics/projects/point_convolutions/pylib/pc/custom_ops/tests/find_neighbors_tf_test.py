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
"""Class to test find neighbors tensorflow implementation"""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, Grid
from pylib.pc.tests import utils
from pylib.pc.custom_ops.custom_ops_tf import find_neighbors_tf
from pylib.pc.custom_ops.custom_ops_tf import find_neighbors_no_grid


class FindNeighborsTF(test_case.TestCase):

  @parameterized.parameters(
    (10, 4, 0.05, 1),
    (10, 4, 0.11, 7),
    (10, 4, 0.142, 19),
    (10, 4, 0.174, 27),
  )
  def test_neighbors_on_3D_meshgrid(self,
                                    num_points_cbrt,
                                    num_points_samples_cbrt,
                                    radius,
                                    expected_num_neighbors):
    num_points = num_points_cbrt**3
    num_samples = num_points_samples_cbrt**3

    points = utils._create_uniform_distributed_point_cloud_3D(
        num_points_cbrt, flat=True)
    batch_ids = np.zeros(num_points)
    points_samples = utils._create_uniform_distributed_point_cloud_3D(
        num_points_samples_cbrt, bb_min=1 / (num_points_samples_cbrt + 1),
        flat=True)
    batch_ids_samples = np.zeros(num_samples)
    point_cloud = PointCloud(points, batch_ids)
    point_cloud_samples = PointCloud(points_samples, batch_ids_samples)
    cell_sizes = np.float32(np.repeat([radius], 3))
    grid = Grid(point_cloud, cell_sizes)

    # with grid
    neigh_ranges, _ = find_neighbors_tf(grid, point_cloud_samples, cell_sizes)
    num_neighbors = np.zeros(num_samples)
    num_neighbors[0] = neigh_ranges[0]
    num_neighbors[1:] = neigh_ranges[1:] - neigh_ranges[:-1]
    expected_num_neighbors = \
        np.ones_like(num_neighbors) * expected_num_neighbors
    self.assertAllEqual(num_neighbors, expected_num_neighbors)

  @parameterized.parameters(
    (10, 4, 0.05, 1),
    (10, 4, 0.11, 7),
    (10, 4, 0.142, 19),
    (10, 4, 0.174, 27),
  )
  def test_neighbors_on_3D_meshgrid_without_gridDS(self,
                                                   num_points_cbrt,
                                                   num_points_samples_cbrt,
                                                   radius,
                                                   expected_num_neighbors):
    num_points = num_points_cbrt**3
    num_samples = num_points_samples_cbrt**3

    points = utils._create_uniform_distributed_point_cloud_3D(
        num_points_cbrt, flat=True)
    batch_ids = np.zeros(num_points)
    points_samples = utils._create_uniform_distributed_point_cloud_3D(
        num_points_samples_cbrt, bb_min=1 / (num_points_samples_cbrt + 1),
        flat=True)
    batch_ids_samples = np.zeros(num_samples)
    point_cloud = PointCloud(points, batch_ids)
    point_cloud_samples = PointCloud(points_samples, batch_ids_samples)

    # without grid
    neigh_ranges, _ = find_neighbors_no_grid(
        point_cloud, point_cloud_samples, radius)
    num_neighbors = np.zeros(num_samples)
    num_neighbors[0] = neigh_ranges[0]
    num_neighbors[1:] = neigh_ranges[1:] - neigh_ranges[:-1]
    expected_num_neighbors = \
        np.ones_like(num_neighbors) * expected_num_neighbors
    self.assertAllEqual(num_neighbors, expected_num_neighbors)

if __name__ == '__main__':
  test_case.main()
