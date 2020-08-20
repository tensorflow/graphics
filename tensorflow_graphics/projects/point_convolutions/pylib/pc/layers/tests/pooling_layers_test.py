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
"""Class to test pooling layers"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, Grid, Neighborhood, AABB
from pylib.pc.tests import utils
from pylib.pc.layers import GlobalMaxPooling, GlobalAveragePooling
from pylib.pc.layers import MaxPooling, AveragePooling


class PoolingTest(test_case.TestCase):

  @parameterized.parameters(
    (10000, 32, 2),
    (20000, 16, 2),
    (40000, 8, 2),
    (10000, 32, 3),
    (20000, 16, 3),
    (40000, 8, 3),
    (40000, 1, 3),
    (10000, 32, 4),
    (20000, 16, 4),
    (40000, 8, 4)
  )
  def test_global_pooling(self, num_points, batch_size, dimension):
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, dimension)
    point_cloud = PointCloud(points, batch_ids)

    # max pooling
    with self.subTest(name='max_pooling'):
      PoolLayer = GlobalMaxPooling()
      pool_tf = PoolLayer(features, point_cloud)
      pool_numpy = np.empty([batch_size, dimension])
      for i in range(batch_size):
        pool_numpy[i] = np.max(features[batch_ids == i], axis=0)
      self.assertAllClose(pool_numpy, pool_tf)

    # average pooling
    with self.subTest(name='average_pooling'):
      PoolLayer = GlobalAveragePooling()
      pool_tf = PoolLayer(features, point_cloud)
      pool_numpy = np.empty([batch_size, dimension])
      for i in range(batch_size):
        pool_numpy[i] = np.mean(features[batch_ids == i], axis=0)
      self.assertAllClose(pool_numpy, pool_tf)

  @parameterized.parameters(
    # neighbor ids are currently corrupted on dimension 2: todo fix
    # (2000, 200, 16, 0.7, 2),
    # (4000, 400, 8, np.sqrt(2), 2),
    (2000, 200, 16, 0.7, 3),
    (4000, 400, 8, np.sqrt(3), 3),
    (4000, 100, 1, np.sqrt(3), 3),
    (2000, 200, 16, 0.7, 4),
    (4000, 400, 8, np.sqrt(4), 4)
  )
  def test_local_pooling(self,
                         num_points,
                         num_samples,
                         batch_size,
                         radius,
                         dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, dimension)
    point_cloud = PointCloud(points, batch_ids)

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)

    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    neighbor_ids = neighborhood._original_neigh_ids.numpy()
    features_on_neighbors = features[neighbor_ids[:, 0]]

    #max pooling
    with self.subTest(name='max_pooling_to_sampled'):
      PoolLayer = MaxPooling()
      pool_tf = PoolLayer(
          features, point_cloud, point_cloud_samples, cell_sizes)

      pool_numpy = np.empty([num_samples, dimension])
      for i in range(num_samples):
        pool_numpy[i] = np.max(
            features_on_neighbors[neighbor_ids[:, 1] == i], axis=0)

      self.assertAllClose(pool_tf, pool_numpy)

    #max pooling
    with self.subTest(name='average_pooling_to_sampled'):
      PoolLayer = AveragePooling()
      pool_tf = PoolLayer(
          features, point_cloud, point_cloud_samples, cell_sizes)

      pool_numpy = np.empty([num_samples, dimension])
      for i in range(num_samples):
        pool_numpy[i] = np.mean(
            features_on_neighbors[neighbor_ids[:, 1] == i], axis=0)

      self.assertAllClose(pool_tf, pool_numpy)


if __name__ == '__main___':
  test_case.main()
