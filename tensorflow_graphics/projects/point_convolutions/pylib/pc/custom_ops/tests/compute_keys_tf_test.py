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
"""Class to test compute_keys tensorflow implementation"""

import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud
from pylib.pc.tests import utils
from pylib.pc.custom_ops.custom_ops_tf import compute_keys_tf


class ComputeKeysTF(test_case.TestCase):

  @parameterized.parameters(
    # (10000, 32, 30, 0.1, 2), # currently corrupted in 2D
    # (20000, 16, 1, 0.2, 2),
    (200, 8, 1, np.sqrt(2), 2),
    (100, 32, 30, 0.1, 3),
    (200, 16, 1, 0.2, 3),
    (200, 8, 1, np.sqrt(3), 3),
    (100, 32, 30, 0.1, 4),
    (200, 16, 1, 0.2, 4),
    (200, 8, 1, np.sqrt(4), 4)
  )
  def test_compute_keys_tf(self,
                           num_points,
                           batch_size,
                           scale,
                           radius,
                           dimension):
    radius = np.repeat(radius, dimension)
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=False)
    point_cloud = PointCloud(points, batch_ids)

    #Compute the number of cells in the grid.
    aabb = point_cloud.get_AABB()
    aabb_sizes = aabb._aabb_max - aabb._aabb_min
    batch_num_cells = tf.cast(
        tf.math.ceil(aabb_sizes / radius), tf.int32)
    total_num_cells = tf.maximum(
        tf.reduce_max(batch_num_cells, axis=0), 1)

    keys_tf = compute_keys_tf(point_cloud, total_num_cells, radius)
    aabb_min = aabb._aabb_min.numpy()

    aabb_min_per_point = aabb_min[batch_ids, :]
    cell_ind = np.floor((points - aabb_min_per_point) / radius).astype(int)
    cell_ind = np.minimum(np.maximum(cell_ind, [0] * dimension),
                          total_num_cells)
    cell_multiplier = np.flip(np.cumprod(np.flip(total_num_cells)))
    cell_multiplier = np.concatenate((cell_multiplier, [1]), axis=0)
    keys = batch_ids * cell_multiplier[0] + \
        np.sum(cell_ind * cell_multiplier[1:].reshape([1, -1]), axis=1)
    # check unsorted keys
    self.assertAllEqual(keys_tf, keys)

if __name__ == '__main__':
  test_case.main()
