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
"""Class to test point sampling operations"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import sample
from pylib.pc import Neighborhood
from pylib.pc.tests import utils


class SamplingTest(test_case.TestCase):

  @parameterized.parameters(
    (100, 8, 0.1, 3),
    (100, 8, 0.1, 3),
    (100, 16, 0.1, 4)
  )
  def test_sampling_poisson_disk_on_random(
        self, num_points, batch_size, cell_size, dimension):
    cell_sizes = np.float32(np.repeat(cell_size, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points)
    point_cloud = PointCloud(points, batch_ids)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes)
    sampled_point_cloud, _ = sample(neighborhood, 'poisson')

    sampled_points = sampled_point_cloud._points.numpy()
    sampled_batch_ids = sampled_point_cloud._batch_ids.numpy()

    min_dist = 1.0
    for i in range(batch_size):
      indices = np.where(sampled_batch_ids == i)
      diff = np.expand_dims(sampled_points[indices], 1) - \
          np.expand_dims(sampled_points[indices], 0)
      dists = np.linalg.norm(diff, axis=2)
      dists = np.sort(dists, axis=1)
      min_dist = min(min_dist, np.amin(dists[:, 1]))

    self.assertLess(min_dist, cell_size + 1e-3)

  @parameterized.parameters(
    (6, 1),
    (100, 5)
  )
  def test_sampling_poisson_disk_on_uniform(self, num_points_sqrt, scale):
    points = utils._create_uniform_distributed_point_cloud_2D(
        num_points_sqrt, scale=scale)
    cell_sizes = scale * np.array([2, 2], dtype=np.float32) \
        / num_points_sqrt
    batch_ids = np.zeros([len(points)])
    point_cloud = PointCloud(points, batch_ids)
    grid  = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes)
    sample_point_cloud, _ = sample(neighborhood, 'poisson')

    sampled_points = sample_point_cloud._points.numpy()
    expected_num_pts = num_points_sqrt ** 2 // 2
    self.assertTrue(len(sampled_points) == expected_num_pts)

  @parameterized.parameters(
    # currently only 3D supported
    # (100, 2, 0.1, 2),
    # (100, 8, 0.7, 2),
    # (100, 32, np.sqrt(2), 2),
    (100, 2, 0.1, 3),
    (100, 8, 0.7, 3),
    (50, 2, np.sqrt(3), 3),
    # (40, 2, 0.1, 4)
  )
  def test_sampling_average_on_random(
        self, num_points, batch_size, cell_size, dimension):
    cell_sizes = np.repeat(cell_size, dimension)
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points)
    #print(points.shape, batch_ids.shape)
    point_cloud = PointCloud(points=points, batch_ids=batch_ids)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes)
    sample_point_cloud, _ = sample(neighborhood, 'average')

    sampled_points_tf = sample_point_cloud._points.numpy()
    sorted_keys = neighborhood._grid._sorted_keys.numpy()
    sorted_points = neighborhood._grid._sorted_points.numpy()

    sampled_points_numpy = []
    cur_point = np.repeat(0.0, dimension)
    cur_key = -1
    cur_num_points = 0.0
    for pt_id, cur_key_point in enumerate(sorted_keys):
      if cur_key_point != cur_key:
        if cur_key != -1:
          cur_point /= cur_num_points
          sampled_points_numpy.append(cur_point)
        cur_key = cur_key_point
        cur_point = [0.0, 0.0, 0.0]
        cur_num_points = 0.0
      cur_point += sorted_points[pt_id]
      cur_num_points += 1.0
    cur_point /= cur_num_points
    sampled_points_numpy.append(cur_point)

    equal = True
    for point_numpy in sampled_points_numpy:
      found = False
      for point_tf in sampled_points_tf:
        if np.all(np.abs(point_numpy - point_tf) < 0.0001):
          found = True
      equal = equal and found
    self.assertTrue(equal)


if __name__ == '__main__':
  test_case.main()
