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
"""Class to test point clouds"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud
from pylib.pc.tests import utils


class PointCloudTest(test_case.TestCase):

  @parameterized.parameters(
    ([32], 100, 3),
    ([5, 2], 100, 2),
    ([2, 3, 4], 100, 4)
  )
  def test_flatten_unflatten_padded(self, batch_shape, num_points, dimension):
    batch_size = np.prod(batch_shape)
    points, sizes = utils._create_random_point_cloud_padded(
        num_points, batch_shape, dimension=dimension)
    point_cloud = PointCloud(points, sizes=sizes)
    retrieved_points = point_cloud.get_points().numpy()
    self.assertAllEqual(points.shape, retrieved_points.shape)
    points = points.reshape([batch_size, num_points, dimension])
    retrieved_points = retrieved_points.reshape(
        [batch_size, num_points, dimension])
    sizes = sizes.reshape([batch_size])
    for i in range(batch_size):
      self.assertAllClose(points[i, :sizes[i]], retrieved_points[i, :sizes[i]])
      self.assertTrue(np.all(retrieved_points[i, sizes[i]:] == 0))

  @parameterized.parameters(
    (100, 32, [8, 4]),
    (100, 16, [2, 2, 2, 2])
  )
  def test_construction_methods(self, max_num_points, batch_size, batch_shape):
    points, sizes = utils._create_random_point_cloud_padded(
        max_num_points, batch_shape)
    num_points = np.sum(sizes)

    sizes_flat = sizes.reshape([batch_size])
    points_flat = points.reshape([batch_size, max_num_points, 3])
    batch_ids = np.repeat(np.arange(0, batch_size), sizes_flat)

    points_seg = np.empty([num_points, 3])
    cur_id = 0
    for pts, size in zip(points_flat, sizes_flat):
      points_seg[cur_id:cur_id + size] = pts[:size]
      cur_id += size

    pc_from_padded = PointCloud(points, sizes=sizes)
    self.assertAllEqual(batch_ids, pc_from_padded._batch_ids)
    self.assertAllClose(points_seg, pc_from_padded._points)

    pc_from_ids = PointCloud(points_seg, batch_ids)
    pc_from_ids.set_batch_shape(batch_shape)

    pc_from_sizes = PointCloud(points_seg, sizes=sizes_flat)
    pc_from_sizes.set_batch_shape(batch_shape)
    self.assertAllEqual(batch_ids, pc_from_sizes._batch_ids)

    points_from_padded = pc_from_padded.get_points(
        max_num_points=max_num_points)
    points_from_ids = pc_from_ids.get_points(
        max_num_points=max_num_points)
    points_from_sizes = pc_from_sizes.get_points(
        max_num_points=max_num_points)

    self.assertAllEqual(points_from_padded, points_from_ids)
    self.assertAllEqual(points_from_ids, points_from_sizes)
    self.assertAllEqual(points_from_sizes, points_from_padded)

  @parameterized.parameters(
    (1000,
     ['Invalid input! Point tensor is of dimension 1 \
        but should be at least 2!',
      'Missing input! Either sizes or batch_ids must be given.',
      'Invalid sizes! Sizes of points and batch_ids are not equal.'])
  )
  def test_exceptions_raised_at_construction(self, num_points, msgs):
    points = np.random.rand(num_points)
    batch_ids = np.zeros(num_points)
    with self.assertRaisesRegex(ValueError, msgs[0]):
      _ = PointCloud(points, batch_ids)
    points = np.random.rand(num_points, 3)
    with self.assertRaisesRegexp(ValueError, msgs[1]):
      _ = PointCloud(points)
    with self.assertRaisesRegexp(AssertionError, msgs[2]):
      _ = PointCloud(points, batch_ids[1:])


if __name__ == '__main__':
  test_case.main()
