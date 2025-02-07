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
"""Class to test kernel point convolutions"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, Grid, Neighborhood, KDEMode, AABB
from pylib.pc.tests import utils
from pylib.pc.layers import KPConv


class KPConvTest(test_case.TestCase):

  @parameterized.parameters(
    (200, 20, [1, 3], 16, 0.7, 5, 3),
    (400, 40, [3, 3], 8, 0.7, 5, 3),
    (400, 10, [3, 1], 1, np.sqrt(3), 5, 3),
  )
  def test_convolution(self,
                       num_points,
                       num_samples,
                       num_features,
                       batch_size,
                       radius,
                       num_kernel_points,
                       dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, num_features[0])
    point_cloud = PointCloud(points, batch_ids)

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    # tf
    conv_layer = KPConv(
        num_features[0], num_features[1], num_kernel_points)
    conv_result_tf = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)

    # numpy
    neighbor_ids = neighborhood._original_neigh_ids.numpy()
    nb_ranges = neighborhood._samples_neigh_ranges.numpy()
    nb_ranges = np.concatenate(([0], nb_ranges), axis=0)
    kernel_points = conv_layer._kernel_points.numpy()
    sigma = conv_layer._sigma.numpy()

    # extract variables
    weights = conv_layer._weights.numpy()

    features_on_neighbors = features[neighbor_ids[:, 0]]
    # compute distances to kernel points
    point_diff = (points[neighbor_ids[:, 0]] -\
                  point_samples[neighbor_ids[:, 1]])\
        / np.expand_dims(cell_sizes, 0)
    kernel_point_diff = np.expand_dims(point_diff, axis=1) -\
        np.expand_dims(kernel_points, axis=0)
    distances = np.linalg.norm(kernel_point_diff, axis=2)
    # compute linear interpolation weights for features based on distances
    kernel_weights = np.maximum(1 - (distances / sigma), 0)
    weighted_features = np.expand_dims(features_on_neighbors, axis=2) *\
        np.expand_dims(kernel_weights, axis=1)
    # sum over neighbors (integration)
    weighted_features_per_sample = \
        np.zeros([num_samples, num_features[0], num_kernel_points])
    for i in range(num_samples):
      weighted_features_per_sample[i] = \
          np.sum(weighted_features[nb_ranges[i]:nb_ranges[i + 1]],
                 axis=0)
    # convolution with summation over kernel dimension
    conv_result_np = \
        np.matmul(
            weighted_features_per_sample.reshape(
                -1,
                num_features[0] * num_kernel_points),
            weights)

    self.assertAllClose(conv_result_tf, conv_result_np, atol=1e-5)

  @parameterized.parameters(
    (8, 4, [8, 8], 2, np.sqrt(3) * 1.25, 15, 3)
  )
  def test_conv_rigid_jacobian_params(self,
                                      num_points,
                                      num_samples,
                                      num_features,
                                      batch_size,
                                      radius,
                                      num_kernel_points,
                                      dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    point_cloud = PointCloud(points, batch_ids)
    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    conv_layer = KPConv(
        num_features[0], num_features[1], num_kernel_points)

    features = np.random.rand(num_points, num_features[0])

    with self.subTest(name='features'):
      def conv_features(features_in):
        conv_result = conv_layer(
          features_in, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      self.assert_jacobian_is_correct_fn(
          conv_features, [features], atol=1e-3, delta=1e-3)

    with self.subTest(name='weights'):
      def conv_weights(weigths_in):
        conv_layer._weights = weigths_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights = conv_layer._weights
      self.assert_jacobian_is_correct_fn(
          conv_weights, [weights], atol=1e-3, delta=1e-3)

  @parameterized.parameters(
    (8, 4, [2, 2], 2, np.sqrt(3) * 1.25, 5, 3)
  )
  def test_conv_deformable_jacobian_params(self,
                                           num_points,
                                           num_samples,
                                           num_features,
                                           batch_size,
                                           radius,
                                           num_kernel_points,
                                           dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    point_cloud = PointCloud(points, batch_ids)
    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    conv_layer = KPConv(
        num_features[0], num_features[1], num_kernel_points, deformable=True)

    features = np.random.rand(num_points, num_features[0])

    with self.subTest(name='features'):
      def conv_features(features_in):
        conv_result = conv_layer(
          features_in, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      self.assert_jacobian_is_correct_fn(
          conv_features, [features], atol=1e-3, delta=1e-3)

    with self.subTest(name='weights'):
      def conv_weights(weigths_in):
        conv_layer._weights = weigths_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights = conv_layer._weights
      self.assert_jacobian_is_correct_fn(
          conv_weights, [weights], atol=1e-3, delta=1e-3)

    with self.subTest(name='offsets'):
      def conv_offset_weights(weigths_in):
        conv_layer._kernel_offsets_weights = weigths_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights = conv_layer._kernel_offsets_weights
      self.assert_jacobian_is_correct_fn(
          conv_offset_weights, [weights], atol=1e-3, delta=1e-3)

    with self.subTest(name='loss'):
      def conv_loss(features):
        _ = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_layer.regularization_loss()

      weights = conv_layer._kernel_offsets_weights
      self.assert_jacobian_is_correct_fn(
          conv_loss, [features], atol=1e-3, delta=1e-2)

  @parameterized.parameters(
    (8, 4, [8, 8], 2, np.sqrt(3) * 1.25, 15, 3),
  )
  def test_conv_jacobian_points(self,
                                num_points,
                                num_samples,
                                num_features,
                                batch_size,
                                radius,
                                num_kernel_points,
                                dimension):
    cell_sizes = np.float32(np.repeat(radius, dimension))
    points, batch_ids = utils._create_random_point_cloud_segmented(
        batch_size, num_points, dimension=dimension)
    features = np.random.rand(num_points, num_features[0])

    point_samples, batch_ids_samples = \
        utils._create_random_point_cloud_segmented(
            batch_size, num_samples, dimension=dimension)

    point_cloud_samples = PointCloud(point_samples, batch_ids_samples)
    point_cloud = PointCloud(points, batch_ids)
    grid = Grid(point_cloud, cell_sizes)
    neighborhood = Neighborhood(grid, cell_sizes, point_cloud_samples)
    neighborhood.compute_pdf()

    conv_layer = KPConv(
          num_features[0], num_features[1], num_kernel_points)

    def conv_points(points_in):
      point_cloud._points = points_in
      neighborhood._grid._sorted_points = \
          tf.gather(points_in, grid._sorted_indices)

      conv_result = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)

      return conv_result

    self.assert_jacobian_is_correct_fn(
        conv_points, [np.float32(points)], atol=1e-3, delta=1e-3)


if __name__ == '__main___':
  test_case.main()
