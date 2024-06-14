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
"""Class to test Monte Carlo convolutions"""

import os
import sys
import numpy as np
import tensorflow as tf
from absl.testing import parameterized
from tensorflow_graphics.util import test_case

from pylib.pc import PointCloud, Grid, Neighborhood, KDEMode, AABB
from pylib.pc.tests import utils
from pylib.pc.layers import MCConv


class MCConvTest(test_case.TestCase):

  @parameterized.parameters(
    (2000, 200, [3, 3], 16, 0.7, 8, 2),
    (4000, 400, [3, 3], 8, np.sqrt(2), 8, 2),
    (2000, 200, [1, 3], 16, 0.7, 8, 3),
    (4000, 400, [3, 3], 8, 0.7, 8, 3),
    (4000, 100, [3, 1], 1, np.sqrt(3), 16, 3),
    (2000, 200, [3, 3], 16, 0.7, 8, 4),
    (4000, 400, [1, 3], 8, np.sqrt(4), 32, 4)
  )
  def test_convolution(self,
                       num_points,
                       num_samples,
                       num_features,
                       batch_size,
                       radius,
                       hidden_size,
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
    conv_layer = MCConv(
        num_features[0], num_features[1], dimension, 1, [hidden_size],
        non_linearity_type='relu')
    conv_result_tf = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)

    # numpy
    pdf = neighborhood.get_pdf().numpy()
    neighbor_ids = neighborhood._original_neigh_ids.numpy()
    nb_ranges = neighborhood._samples_neigh_ranges.numpy()

    # extract variables
    hidden_weights = \
        tf.reshape(conv_layer._weights_tf[0], [dimension, hidden_size]).numpy()
    hidden_biases = \
        tf.reshape(conv_layer._bias_tf[0], [1, hidden_size]).numpy()
    weights = \
        tf.reshape(conv_layer._final_weights_tf,
                   [num_features[0] * hidden_size, num_features[1]]).numpy()

    features_on_neighbors = features[neighbor_ids[:, 0]]
    # compute first layer of kernel MLP
    point_diff = (points[neighbor_ids[:, 0]] -\
                  point_samples[neighbor_ids[:, 1]])\
        / np.expand_dims(cell_sizes, 0)

    latent_per_nb = np.dot(point_diff, hidden_weights) + hidden_biases

    latent_relu_per_nb = np.maximum(latent_per_nb, 0)

    # Monte-Carlo integration after first layer
    # weighting with pdf
    weighted_features_per_nb = np.expand_dims(features_on_neighbors, 2) * \
        np.expand_dims(latent_relu_per_nb, 1) / \
        np.expand_dims(pdf, [1, 2])
    nb_ranges = np.concatenate(([0], nb_ranges), axis=0)
    # sum (integration)
    weighted_latent_per_sample = \
        np.zeros([num_samples, num_features[0], hidden_size])
    for i in range(num_samples):
      weighted_latent_per_sample[i] = \
          np.sum(weighted_features_per_nb[nb_ranges[i]:nb_ranges[i + 1]],
                 axis=0)
    # second layer of MLP (linear)
    weighted_latent_per_sample = np.reshape(weighted_latent_per_sample,
                                            [num_samples, -1])
    conv_result_np = np.matmul(weighted_latent_per_sample, weights)

    self.assertAllClose(conv_result_tf, conv_result_np)

  @parameterized.parameters(
    (8, 4, [8, 8], 2, np.sqrt(3) * 1.25, 8, 3)
  )
  def test_conv_jacobian_params(self,
                                num_points,
                                num_samples,
                                num_features,
                                batch_size,
                                radius,
                                hidden_size,
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
    conv_layer = MCConv(
        num_features[0], num_features[1], dimension, 1, [hidden_size])

    features = np.random.rand(num_points, num_features[0])

    with self.subTest(name='features'):
      def conv_features(features_in):
        conv_result = conv_layer(
          features_in, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      self.assert_jacobian_is_correct_fn(
          conv_features, [features], atol=1e-4, delta=1e-3)

    with self.subTest(name='params_basis_axis_proj'):
      def conv_basis(weights_tf_in):
        conv_layer._weights_tf[0] = weights_tf_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights_tf = conv_layer._weights_tf[0]
      self.assert_jacobian_is_correct_fn(
          conv_basis, [weights_tf], atol=1e-4, delta=1e-3)

    with self.subTest(name='params_basis_bias_proj'):
      def conv_basis(bias_tf_in):
        conv_layer._bias_tf[0] = bias_tf_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      bias_tf = conv_layer._bias_tf[0]
      self.assert_jacobian_is_correct_fn(
          conv_basis, [bias_tf], atol=1e-4, delta=1e-4)

    with self.subTest(name='params_second_layer'):
      def conv_weights(weigths_in):
        conv_layer._final_weights_tf = weigths_in
        conv_result = conv_layer(
          features, point_cloud, point_cloud_samples, radius, neighborhood)
        return conv_result

      weights = conv_layer._final_weights_tf
      self.assert_jacobian_is_correct_fn(
          conv_weights, [weights], atol=1e-4, delta=1e-3)

  @parameterized.parameters(
    (8, 4, [8, 8], 2, np.sqrt(3) * 1.25, 8, 3),
  )
  def test_conv_jacobian_points(self,
                                num_points,
                                num_samples,
                                num_features,
                                batch_size,
                                radius,
                                hidden_size,
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

    conv_layer = MCConv(
          num_features[0], num_features[1], dimension, 1, [hidden_size], 'elu')

    def conv_points(points_in):
      point_cloud._points = points_in
      neighborhood._grid._sorted_points = \
          tf.gather(points_in, grid._sorted_indices)

      conv_result = conv_layer(
        features, point_cloud, point_cloud_samples, radius, neighborhood)

      return conv_result

    self.assert_jacobian_is_correct_fn(
        conv_points, [np.float32(points)], atol=1e-4, delta=1e-3)


if __name__ == '__main___':
  test_case.main()
