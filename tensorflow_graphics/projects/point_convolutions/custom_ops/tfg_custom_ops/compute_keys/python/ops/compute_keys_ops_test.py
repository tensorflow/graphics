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
"""Class to test key computation of regular grid"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow.python.framework import ops
from tensorflow.python.platform import test
from tensorflow.python.framework import test_util
try:
  from tfg_custom_ops.compute_keys.python.ops.compute_keys_ops import\
       compute_keys
except ImportError:
  import compute_keys


def _create_random_point_cloud_segmented(batch_size,
                                         num_points,
                                         dimension=3,
                                         sizes=None,
                                         scale=1,
                                         clean_aabb=False,
                                         equal_sized_batches=False):
  points = np.random.uniform(0, scale, [num_points, dimension])
  if sizes is None:
    if not equal_sized_batches:
      batch_ids = np.random.randint(0, batch_size, num_points)
      batch_ids[:batch_size] = np.arange(0, batch_size)
    else:
      batch_ids = np.repeat(np.arange(0, batch_size), num_points // batch_size)
  else:
    sizes = np.array(sizes, dtype=int)
    batch_ids = np.repeat(np.arange(0, batch_size), sizes)
  if clean_aabb:
    # adds points such that the aabb is [0,0,0] [1,1,1]*scale
    # to prevent rounding errors
    points = np.concatenate(
        (points, scale * np.ones([batch_size, dimension]),
         np.zeros([batch_size, dimension])))
    batch_ids = np.concatenate(
        (batch_ids, np.arange(0, batch_size), np.arange(0, batch_size)))
  return points, batch_ids


class GridTest(test.TestCase):

  @test_util.run_gpu_only
  def test_compute_keys_with_sort(self):
    num_points = 10000
    batch_size = 32
    radius = 0.1
    dimension = 3
    radius_array = np.repeat(radius, dimension)
    points, batch_ids = _create_random_point_cloud_segmented(
        batch_size, num_points * batch_size, dimension=dimension,
        sizes=np.ones(batch_size, dtype=int) * num_points, clean_aabb=False)
    aabb_min_per_batch = np.empty([batch_size, dimension])
    aabb_max_per_batch = np.empty([batch_size, dimension])
    for i in range(batch_size):
      aabb_min_per_batch[i] = np.amin(points[batch_ids == i], axis=0)
      aabb_max_per_batch[i] = np.amax(points[batch_ids == i], axis=0)
    aabb_sizes = aabb_max_per_batch - aabb_min_per_batch
    total_num_cells = np.max(np.ceil(aabb_sizes / radius), axis=0)
    custom_keys = compute_keys(
        points, batch_ids, aabb_min_per_batch / radius,
        total_num_cells, 1 / radius_array)

    aabb_min_per_point = aabb_min_per_batch[batch_ids, :]
    cell_ind = np.floor((points - aabb_min_per_point) / radius).astype(int)
    cell_ind = np.minimum(np.maximum(cell_ind, [0] * dimension),
                          total_num_cells)
    ref_keys = batch_ids * total_num_cells[0] * \
        total_num_cells[1] * total_num_cells[2] + \
        cell_ind[:, 0] * total_num_cells[1] * total_num_cells[2] + \
        cell_ind[:, 1] * total_num_cells[2] + cell_ind[:, 2]

    # check unsorted keys
    self.assertAllEqual(custom_keys, ref_keys)


if __name__ == '__main__':
  test.main()
