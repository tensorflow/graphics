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
"""Class to test kernel density estimation tensorflow implementation"""

import os
import sys
import numpy as np
from sklearn.neighbors import KernelDensity
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud
from pylib.pc import Grid
from pylib.pc import KDEMode
from pylib.pc import Neighborhood
from pylib.pc.custom_ops.custom_ops_tf import compute_pdf_tf
from pylib.pc.tests import utils


class ComputePDFTFTest(test_case.TestCase):

  @parameterized.parameters(
    (2, 100, 10, 0.2, 0.1, 2),
    (2, 100, 10, 0.7, 0.1, 2),
    (2, 100, 10, np.sqrt(2), 0.1, 2),
    (2, 100, 10, 0.2, 0.2, 3),
    (2, 100, 10, 0.7, 0.1, 3),
    (2, 100, 10, np.sqrt(3), 0.2, 3),
    (2, 100, 10, 0.2, 0.2, 4),
    (2, 100, 10, np.sqrt(4), 0.2, 4)
  )
  def test_compute_pdf_tf(self,
                          batch_size,
                          num_points,
                          num_samples_per_batch,
                          cell_size,
                          bandwidth,
                          dimension):
    cell_sizes = np.float32(np.repeat(cell_size, dimension))
    bandwidths = np.float32(np.repeat(bandwidth, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, batch_size * num_points, dimension,
        equal_sized_batches=True)
    samples = np.full((batch_size * num_samples_per_batch, dimension),
                      0.0, dtype=float)
    for i in range(batch_size):
      cur_choice = np.random.choice(num_points, num_samples_per_batch,
                                    replace=True)
      samples[num_samples_per_batch * i:num_samples_per_batch * (i + 1), :] = \
          points[cur_choice + i * num_points]
    samples_batch_ids = np.repeat(np.arange(0, batch_size),
                                  num_samples_per_batch)

    point_cloud = PointCloud(points, batch_ids, batch_size)
    grid = Grid(point_cloud, cell_sizes)

    point_cloud_samples = PointCloud(samples, samples_batch_ids, batch_size)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    neighbor_ids = neighborhood._neighbors
    pdf_neighbors = Neighborhood(grid, cell_sizes)
    pdf_tf = compute_pdf_tf(pdf_neighbors, bandwidths, KDEMode.constant)
    pdf_tf = tf.gather(pdf_tf, neighbor_ids[:, 0])

    sorted_points = grid._sorted_points.numpy()
    sorted_batch_ids = grid._sorted_batch_ids.numpy()
    neighbor_ids = neighborhood._neighbors

    pdf_real = []
    accum_points = []
    prev_batch_i = -1
    for pt_i, batch_i in enumerate(sorted_batch_ids):
      if batch_i != prev_batch_i:
        if len(accum_points) > 0:
          test_points = np.array(accum_points)
          kde_skl = KernelDensity(bandwidth=bandwidth)
          kde_skl.fit(test_points)
          log_pdf = kde_skl.score_samples(test_points)
          pdf = np.exp(log_pdf)
          if len(pdf_real) > 0:
            pdf_real = np.concatenate((pdf_real, pdf), axis=0)
          else:
            pdf_real = pdf
        accum_points = [sorted_points[pt_i] / cell_size]
        prev_batch_i = batch_i
      else:
        accum_points.append(sorted_points[pt_i] / cell_size)

    test_points = np.array(accum_points)
    kde_skl = KernelDensity(bandwidth=bandwidth)
    kde_skl.fit(test_points)
    log_pdf = kde_skl.score_samples(test_points)
    pdf = np.exp(log_pdf)
    if len(pdf_real) > 0:
      pdf_real = np.concatenate((pdf_real, pdf), axis=0)
    else:
      pdf_real = pdf

    pdf_tf = np.asarray(pdf_tf / float(len(accum_points)))
    pdf_skl = np.asarray(pdf_real)[neighbor_ids[:, 0]]
    self.assertAllClose(pdf_tf, pdf_skl)

  @parameterized.parameters(
    (1, 20, 1, np.sqrt(2), 2),
    (1, 20, 1, np.sqrt(3), 3),
    (1, 20, 1, np.sqrt(4), 4)
  )
  def test_compute_pdf_jacobian(self,
                                batch_size,
                                num_points,
                                num_samples,
                                radius,
                                dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    bandwidths = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, batch_size * num_points, dimension,
        equal_sized_batches=True)
    samples = np.full((batch_size * num_samples, dimension), 0.0, dtype=float)
    for i in range(batch_size):
      cur_choice = np.random.choice(num_points, num_samples, replace=True)
      samples[num_samples * i:num_samples * (i + 1), :] = \
          points[cur_choice + i * num_points]
    samples_batch_ids = np.repeat(np.arange(0, batch_size), num_samples)
    def compute_pdf(points_in):
      point_cloud = PointCloud(points_in, batch_ids, batch_size)
      grid = Grid(point_cloud, cell_sizes)

      point_cloud_samples = PointCloud(samples, samples_batch_ids, batch_size)
      neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
      neighborhood.compute_pdf(bandwidths, KDEMode.constant, normalize=True)
      # account for influence of neighborhood size
      _, _, counts = tf.unique_with_counts(neighborhood._neighbors[:, 1])
      max_num_nb = tf.cast(tf.reduce_max(counts), tf.float32)
      return neighborhood._pdf / max_num_nb

    self.assert_jacobian_is_correct_fn(
        compute_pdf, [np.float32(points)], atol=1e-4)


if __name__ == '__main__':
  test_case.main()
