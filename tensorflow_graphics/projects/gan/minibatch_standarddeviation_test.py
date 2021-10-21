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
"""Tests for minibatch_standarddeviation."""

import tensorflow as tf
from tensorflow_graphics.projects.gan import minibatch_standarddeviation
from tensorflow_graphics.util import test_case


def create_random_testcase(shape):
  batch = tf.random.uniform(shape)
  mean_accross_batches_squared = tf.reduce_mean(batch, 0)**2
  mean_accross_squared_batches = tf.reduce_mean(batch**2, 0)

  expected_stddeviation = tf.reduce_mean(
      tf.sqrt(mean_accross_squared_batches - mean_accross_batches_squared))
  expected = tf.concat(
      [batch,
       tf.broadcast_to(expected_stddeviation, batch.shape[:-1] + [1])], 3)
  return batch, expected


class MinibatchStandarddeviationTest(test_case.TestCase):

  def test_zero_deviation(self):
    batch_size = 8
    resolution = 64
    n_channels = 3

    batch = tf.ones([batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation()
    layer.build(batch.shape)

    is_equal = tf.equal(
        layer.call(batch),
        tf.concat([batch, tf.broadcast_to(0.0, batch.shape[:-1] + [1])], 3))

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_constant_samples(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch = tf.concat([
        tf.ones([1, resolution, resolution, n_channels]) * (batch + 1)
        for batch in range(batch_size)
    ], 0)

    mean = 2.0
    expected_stddeviation = tf.sqrt(
        ((3.0 - mean)**2 + (2.0 - mean)**2 + (1.0 - mean)**2) / 3)

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation()
    layer.build(batch.shape)

    expected = tf.concat(
        [batch,
         tf.broadcast_to(expected_stddeviation, batch.shape[:-1] + [1])], 3)

    difference = tf.abs(layer.call(batch) - expected)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_random(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch, expected = create_random_testcase(
        [batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation()
    layer.build(batch.shape)

    difference = tf.abs(layer.call(batch) - expected)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_distributed_with_replica_ctx(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch = tf.random.uniform([batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation()
    layer.build(batch.shape)
    non_distributed = layer.call(batch)

    strategy = tf.distribute.MirroredStrategy()
    layer_dist = minibatch_standarddeviation.SyncMiniBatchStandardDeviation()
    layer_dist.build(batch.shape)
    distributed = strategy.run(layer_dist.call, [
        strategy.experimental_distribute_values_from_function(lambda _: batch)
    ])

    difference = tf.abs(non_distributed - distributed)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_distributed_without_replica_ctx(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch = tf.random.uniform([batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation()
    layer.build(batch.shape)
    non_distributed = layer.call(batch)

    strategy = tf.distribute.MirroredStrategy()
    layer_dist = minibatch_standarddeviation.SyncMiniBatchStandardDeviation()
    layer_dist.build(batch.shape)

    with strategy.scope():
      distributed = layer_dist.call(batch)

    difference = tf.abs(non_distributed - distributed)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_config(self):
    layer = minibatch_standarddeviation.MiniBatchStandardDeviation(
        channel_axis=3, batch_axis=2)

    config = layer.get_config()

    expected_child_config = {
        "batch_axis": 2,
        "channel_axis": 3,
    }

    expected_super_config = super(
        minibatch_standarddeviation.MiniBatchStandardDeviation,
        layer).get_config()
    expected_config = dict(
        list(expected_child_config.items()) +
        list(expected_super_config.items()))

    self.assertEqual(config, expected_config)

  def test_batch_axis_too_large(self):
    batch_size = 8
    resolution = 64
    n_channels = 3

    batch = tf.ones([batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation(
        batch_axis=10)
    layer.build(batch.shape)

    try:
      layer.call(batch)
      self.fail("Too large a batch_axis argument should cause a ValueError.")
    except ValueError:
      pass

  def test_channel_axis_too_large(self):
    batch_size = 8
    resolution = 64
    n_channels = 3

    batch = tf.ones([batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation(
        channel_axis=10)
    layer.build(batch.shape)

    try:
      layer.call(batch)
      self.fail("Too large a channel_axis argument should cause a ValueError.")
    except ValueError:
      pass

  def test_invalid_channel_axis_argument(self):
    try:
      minibatch_standarddeviation.MiniBatchStandardDeviation(
          channel_axis="expects an integer")
      self.fail("An invalid channel_axis argument should cause a TypeError.")
    except TypeError:
      pass

  def test_invalid_batch_axis_argument(self):
    try:
      minibatch_standarddeviation.MiniBatchStandardDeviation(
          batch_axis="expects an integer")
      self.fail("An invalid channel_axis argument should cause a TypeError.")
    except TypeError:
      pass

  def test_axis_indices(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch, expected = create_random_testcase(
        [batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation(
        batch_axis=-4, channel_axis=3)
    layer.build(batch.shape)

    difference = tf.abs(layer.call(batch) - expected)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))

  def test_axis_specified_as_tuple(self):
    batch_size = 3
    resolution = 64
    n_channels = 3

    batch, expected = create_random_testcase(
        [batch_size, resolution, resolution, n_channels])

    layer = minibatch_standarddeviation.MiniBatchStandardDeviation(
        batch_axis=(0,), channel_axis=(-1,))
    layer.build(batch.shape)

    difference = tf.abs(layer.call(batch) - expected)
    epsilon = 1e-5
    is_equal = difference < epsilon

    self.assertTrue(tf.math.reduce_all(is_equal))


if __name__ == "__main__":
  test_case.main()
