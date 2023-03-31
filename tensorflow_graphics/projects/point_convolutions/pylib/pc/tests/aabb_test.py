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
"""Class to test bounding box"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, AABB
from pylib.pc.tests import utils


class AABB_test(test_case.TestCase):

  @parameterized.parameters(
      (1, 1000, 3),
      (8, 1000, 2),
      (32, 1000, 4)
  )
  def test_aabb_min_max(self, batch_size, num_points, dimension):
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension)
    aabb_max_numpy = np.empty([batch_size, dimension])
    aabb_min_numpy = np.empty([batch_size, dimension])
    for i in range(batch_size):
      aabb_max_numpy[i] = np.amax(points[batch_ids == i], axis=0)
      aabb_min_numpy[i] = np.amin(points[batch_ids == i], axis=0)

    aabb_tf = PointCloud(points, batch_ids=batch_ids,
                         batch_size=batch_size).get_AABB()

    self.assertAllClose(aabb_max_numpy, aabb_tf._aabb_max)
    self.assertAllClose(aabb_min_numpy, aabb_tf._aabb_min)

  @parameterized.parameters(
      ([1], 1000, 3),
      ([4, 4], 1000, 2),
      ([1, 2, 3], 100, 4)
  )
  def test_aabb_diameter(self, batch_shape, max_num_points, dimension):
    points, sizes = utils._create_random_point_cloud_padded(
        max_num_points, batch_shape, dimension)
    batch_size = np.prod(batch_shape)
    diameter_numpy = np.empty(batch_size)
    points_flat = np.reshape(points, [batch_size, max_num_points, dimension])
    sizes_flat = np.reshape(sizes, [batch_size])
    for i in range(batch_size):
      curr_pts = points_flat[i][:sizes_flat[i]]
      diag = np.amax(curr_pts, axis=0) - np.amin(curr_pts, axis=0)
      diameter_numpy[i] = np.linalg.norm(diag)
    diameter_numpy = np.reshape(diameter_numpy, batch_shape)

    aabb_tf = PointCloud(points, sizes=sizes).get_AABB()
    diameter_tf = aabb_tf.get_diameter()
    self.assertAllClose(diameter_numpy, diameter_tf)

if __name__ == '__main__':
  test_case.main()
