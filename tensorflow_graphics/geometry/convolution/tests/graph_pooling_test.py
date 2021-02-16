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
"""Tests for tensorflow_graphics.geometry.convolution.tests.graph_pooling."""

# pylint: disable=protected-access

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools
from absl.testing import parameterized
import numpy as np
from six.moves import range
import tensorflow as tf

import tensorflow_graphics.geometry.convolution.graph_pooling as gp
from tensorflow_graphics.geometry.convolution.tests import utils_test
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Convert a numpy array to a tf.SparseTensor."""
  return utils_test._dense_to_sparse(data)


def _batch_sparse_eye(batch_shape, num_vertices, dtype):
  """Generate a batch of identity matrices."""
  eye = np.eye(num_vertices, dtype=dtype)
  num_batch_dims = len(batch_shape)
  expand_shape = np.concatenate(
      (np.ones(num_batch_dims, dtype=np.int32), (num_vertices, num_vertices)),
      axis=0)
  eye = np.reshape(eye, expand_shape)
  tile_shape = np.concatenate((batch_shape, (1, 1)), axis=0)
  return _dense_to_sparse(np.tile(eye, tile_shape))


class GraphPoolingTestPoolTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'pool_map' and 'data' must have the same type.", np.float32, np.float64,
       np.int32))
  def test_pool_exception_raised_types(self, err_msg, data_type, pool_map_type,
                                       sizes_type):
    """Tests the correct exceptions are raised for invalid types."""
    data = np.ones((2, 3, 3), dtype=data_type)
    pool_map = _dense_to_sparse(np.ones((2, 3, 3), dtype=pool_map_type))
    sizes = np.array(((1, 2), (2, 3)), dtype=sizes_type)

    with self.assertRaisesRegexp(TypeError, err_msg):
      gp.pool(data, pool_map, sizes)

  @parameterized.parameters(
      ('data must have a rank greater than 1', (3,), (3,), None),
      ('pool_map must have a rank of 2', (3, 3), (3,), None),
      ('sizes must have a rank of 3', (4, 5, 3, 2), (4, 5, 3, 3), (3, 2)),
  )
  def test_pool_exception_raised_shapes(self, err_msg, data_shape,
                                        pool_map_shape, sizes_shape):
    """Tests the correct exceptions are raised for invalid shapes."""
    data = np.ones(data_shape, dtype=np.float32)
    pool_map = _dense_to_sparse(np.ones(pool_map_shape, dtype=np.float32))
    if sizes_shape is not None:
      sizes = np.ones(sizes_shape, dtype=np.int32)
    else:
      sizes = None

    with self.assertRaisesRegexp(ValueError, err_msg):
      gp.pool(data, pool_map, sizes)

  def test_pool_exception_raised_algorithm(self):
    """Tests the correct exception is raised for an invalid algorithm."""
    data = np.ones(shape=(2, 2))
    pool_map = _dense_to_sparse(np.ones(shape=(2, 2)))

    with self.assertRaisesRegexp(
        ValueError, 'The pooling method must be "weighted" or "max"'):
      gp.pool(data, pool_map, sizes=None, algorithm='mean')

  @parameterized.parameters(
      ((2, 3), 4, 3, np.float32),
      ((1,), 6, 1, np.float32),
      ((4, 1, 3), 9, 7, np.float64),
      ((2, 8, 4, 6), 19, 11, np.float64),
  )
  def test_pool_identity(self, batch_shape, num_vertices, num_features,
                         data_type):
    """Tests graph pooling with identity maps."""
    data_shape = np.concatenate((batch_shape, (num_vertices, num_features)))
    data = np.random.uniform(size=data_shape).astype(data_type)
    pool_map = _batch_sparse_eye(batch_shape, num_vertices, data_type)

    pooled_max = gp.pool(data, pool_map, sizes=None, algorithm='max')
    pooled_weighted = gp.pool(data, pool_map, sizes=None, algorithm='weighted')

    self.assertAllClose(pooled_max, data)
    self.assertAllClose(pooled_weighted, data)

  def test_pool_preset_padded(self):
    """Tests pooling with preset data and padding."""
    data = np.reshape(np.arange(12).astype(np.float32), (2, 3, 2))
    sizes = ((2, 3), (3, 3))
    pool_map = _dense_to_sparse(
        np.array((((0.5, 0.5, 0.), (0., 0., 1.), (0., 0., 0.)),
                  ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                 dtype=np.float32))

    pooled_max = gp.pool(data, pool_map, sizes, algorithm='max')
    pooled_weighted = gp.pool(data, pool_map, sizes, algorithm='weighted')
    true_max = (((2., 3.), (4., 5.), (0., 0.)), ((6., 7.), (8., 9.), (10.,
                                                                      11.)))
    true_weighted = (((1., 2.), (4., 5.), (0., 0.)), ((6., 7.), (8., 9.),
                                                      (10., 11.)))

    self.assertAllClose(pooled_max, true_max)
    self.assertAllClose(pooled_weighted, true_weighted)

  def test_pool_preset(self):
    """Tests pooling with preset data."""
    pool_map = np.array(((0.5, 0.5, 0., 0.), (0., 0., 0.5, 0.5)),
                        dtype=np.float32)
    pool_map = _dense_to_sparse(pool_map)
    data = np.reshape(np.arange(8).astype(np.float32), (4, 2))
    max_true = data[(1, 3), :]
    max_weighted = (data[(0, 2), :] + max_true) * 0.5

    pooled_max = gp.pool(data, pool_map, sizes=None, algorithm='max')
    pooled_weighted = gp.pool(data, pool_map, sizes=None, algorithm='weighted')

    self.assertAllClose(pooled_max, max_true)
    self.assertAllClose(pooled_weighted, max_weighted)

  @parameterized.parameters((20, 10, 3), (2, 1, 1), (2, 5, 4), (2, 1, 3))
  def test_pool_random(self, num_input_vertices, num_output_vertices,
                       num_features):
    """Tests pooling with random inputs."""
    pool_map = 0.001 + np.random.uniform(
        size=(num_output_vertices, num_input_vertices))
    data = np.random.uniform(size=(num_input_vertices, num_features))
    true_weighted = np.matmul(pool_map, data)
    true_max = np.tile(
        np.max(data, axis=0, keepdims=True), (num_output_vertices, 1))
    pool_map = _dense_to_sparse(pool_map)

    with self.subTest(name='max'):
      pooled_max = gp.pool(data, pool_map, None, algorithm='max')
      self.assertAllClose(pooled_max, true_max)

    with self.subTest(name='weighted'):
      pooled_weighted = gp.pool(data, pool_map, None, algorithm='weighted')
      self.assertAllClose(pooled_weighted, true_weighted)

  def test_pool_jacobian(self):
    """Tests the jacobian is correct."""
    sizes = ((2, 4), (3, 5))
    data_init = np.random.uniform(size=(2, 5, 3))
    pool_map = np.random.uniform(size=(2, 3, 5))
    data_init[0, -1, :] = 0.
    pool_map[0, -1, :] = 0.
    pool_map = _dense_to_sparse(pool_map)

    def gp_pool(data, algorithm):
      return gp.pool(data, pool_map, sizes, algorithm=algorithm)

    with self.subTest(name='max'):
      self.assert_jacobian_is_correct_fn(lambda data: gp_pool(data, 'max'),
                                         [data_init])

    with self.subTest(name='weighted'):
      self.assert_jacobian_is_correct_fn(lambda data: gp_pool(data, 'weighted'),
                                         [data_init])


class GraphPoolingTestUnpoolTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'pool_map' and 'data' must have the same type.", np.float32, np.float64,
       np.int32))
  def test_unpool_exception_raised_types(self, err_msg, data_type,
                                         pool_map_type, sizes_type):
    """Tests the correct exceptions are raised for invalid types."""
    data = np.ones((2, 3, 3), dtype=data_type)
    pool_map = _dense_to_sparse(np.ones((2, 3, 3), dtype=pool_map_type))
    sizes = np.array(((1, 2), (2, 3)), dtype=sizes_type)

    with self.assertRaisesRegexp(TypeError, err_msg):
      gp.unpool(data, pool_map, sizes)

  @parameterized.parameters(
      ('data must have a rank greater than 1', (3,), (3,), None),
      ('pool_map must have a rank of 2', (3, 3), (3,), None),
      ('sizes must have a rank of 3', (4, 5, 3, 2), (4, 5, 3, 3), (3, 2)),
      ('data must have a rank less than 6', (2, 3, 4, 5, 3, 2),
       (2, 3, 4, 5, 3, 3), None),
  )
  def test_unpool_exception_raised_shapes(self, err_msg, data_shape,
                                          pool_map_shape, sizes_shape):
    """Tests the correct exceptions are raised for invalid shapes."""
    data = np.ones(data_shape, dtype=np.float32)
    pool_map = _dense_to_sparse(np.ones(pool_map_shape, dtype=np.float32))
    if sizes_shape is not None:
      sizes = np.ones(sizes_shape, dtype=np.int32)
    else:
      sizes = None

    with self.assertRaisesRegexp(ValueError, err_msg):
      gp.unpool(data, pool_map, sizes)

  @parameterized.parameters(
      ((2, 3), 4, 3, np.float32),
      ((1,), 6, 1, np.float32),
      ((4, 1, 3), 9, 7, np.float64),
      ((2, 8, 4), 19, 11, np.float64),
  )
  def test_unpool_identity(self, batch_shape, num_vertices, num_features,
                           data_type):
    """Tests graph unpooling with identity maps."""
    data_shape = np.concatenate((batch_shape, (num_vertices, num_features)))
    data = np.random.uniform(size=data_shape).astype(data_type)
    pool_map = _batch_sparse_eye(batch_shape, num_vertices, data_type)

    unpooled = gp.unpool(data, pool_map, sizes=None)
    self.assertAllClose(unpooled, data)

  def test_unpool_preset_padded(self):
    """Tests pooling with preset data and padding."""
    data = np.reshape(np.arange(12).astype(np.float32), (2, 3, 2))
    data[0, -1, :] = 0.
    sizes = ((2, 3), (3, 3))
    pool_map = _dense_to_sparse(
        np.array((((0.5, 0.5, 0.), (0., 0., 1.), (0., 0., 0.)),
                  ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                 dtype=np.float32))

    unpooled = gp.unpool(data, pool_map, sizes)

    true = (((0., 1.), (0., 1.), (2., 3.)), ((6., 7.), (8., 9.), (10., 11.)))
    self.assertAllClose(unpooled, true)

  @parameterized.parameters((20, 4), (2, 1), (12, 4), (6, 3))
  def test_unpool_random(self, num_vertices, num_features):
    """Tests pooling with random data inputs."""
    output_vertices = num_vertices // 2
    pool_map = np.zeros(shape=(output_vertices, num_vertices), dtype=np.float32)
    for i in range(output_vertices):
      pool_map[i, (i * 2, i * 2 + 1)] = (0.5, 0.5)
    data = np.random.uniform(size=(output_vertices,
                                   num_features)).astype(np.float32)

    unpooled = gp.unpool(data, _dense_to_sparse(pool_map), sizes=None)

    with self.subTest(name='direct_unpool'):
      true = np.zeros(shape=(num_vertices, num_features)).astype(np.float32)
      true[0::2, :] = data
      true[1::2, :] = data
      self.assertAllClose(unpooled, true)

    with self.subTest(name='permute_pool_map'):
      permutation = np.random.permutation(num_vertices)
      pool_map_permute = pool_map[:, permutation]
      unpooled_permute = gp.unpool(data, _dense_to_sparse(pool_map_permute),
                                   None)
      true_permute = true[permutation, :]
      self.assertAllClose(unpooled_permute, true_permute)

  def test_unpool_jacobian_random(self):
    """Tests the jacobian is correct."""
    sizes = ((2, 4), (3, 5))
    data_init = np.random.uniform(size=(2, 3, 6))
    pool_map = np.random.uniform(size=(2, 3, 5))
    data_init[0, -1, :] = 0.
    pool_map[0, -1, :] = 0.
    pool_map = _dense_to_sparse(pool_map)

    def gp_unpool(data):
      return gp.unpool(data, pool_map, sizes)

    self.assert_jacobian_is_correct_fn(gp_unpool, [data_init])


class GraphPoolingUpsampleTransposeConvolutionTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'pool_map' and 'data' must have the same type.", np.float32, np.float64,
       np.int32))
  def test_upsample_transposed_convolution_exception_raised_types(
      self, err_msg, data_type, pool_map_type, sizes_type):
    """Tests the correct exceptions are raised for invalid types."""
    data = np.ones((2, 3, 3), dtype=data_type)
    pool_map = _dense_to_sparse(np.ones((2, 3, 3), dtype=pool_map_type))
    sizes = np.array(((1, 2), (2, 3)), dtype=sizes_type)

    with self.assertRaisesRegexp(TypeError, err_msg):
      gp.upsample_transposed_convolution(
          data, pool_map, sizes, kernel_size=1, transposed_convolution_op=None)

  @parameterized.parameters(
      ('data must have a rank greater than 1', (3,), (3,), None),
      ('pool_map must have a rank of 2', (3, 3), (3,), None),
      ('sizes must have a rank of 3', (4, 5, 3, 2), (4, 5, 3, 3), (3, 2)),
      ('data must have a rank less than 6', (2, 3, 4, 5, 3, 2),
       (2, 3, 4, 5, 3, 3), None),
  )
  def test_upsample_transposed_convolution_exception_raised_shapes(
      self, err_msg, data_shape, pool_map_shape, sizes_shape):
    """Tests the correct exceptions are raised for invalid shapes."""
    data = np.ones(data_shape, dtype=np.float32)
    pool_map = _dense_to_sparse(np.ones(pool_map_shape, dtype=np.float32))
    if sizes_shape is not None:
      sizes = np.ones(sizes_shape, dtype=np.int32)
    else:
      sizes = None

    with self.assertRaisesRegexp(ValueError, err_msg):
      gp.upsample_transposed_convolution(
          data, pool_map, sizes, kernel_size=1, transposed_convolution_op=None)

  def test_upsample_transposed_convolution_exception_raised_callable(self):
    """Tests the correct exception is raised for a invalid convolution op."""
    data = np.ones((5, 3))
    pool_map = _dense_to_sparse(np.eye(5))
    err_msg = "'transposed_convolution_op' must be callable."

    with self.assertRaisesRegexp(TypeError, err_msg):
      gp.upsample_transposed_convolution(
          data,
          pool_map,
          sizes=None,
          kernel_size=1,
          transposed_convolution_op=1)

  @parameterized.parameters((1, 1, 1, np.float32), (5, 3, 1, np.float32),
                            (3, 6, 15, np.float64))
  def test_upsample_transposed_convolution_zero_kernel(self, num_vertices,
                                                       num_features,
                                                       kernel_size, data_type):
    """Tests the upsampling with a zero kernel."""
    data = np.random.uniform(size=(num_vertices,
                                   num_features)).astype(data_type)
    pool_map = np.zeros(
        shape=(num_vertices, num_vertices * kernel_size), dtype=data_type)
    for i in range(num_vertices):
      pool_map[i, np.arange(kernel_size * i, kernel_size *
                            (i + 1))] = (1.0 / kernel_size)
    pool_map = _dense_to_sparse(pool_map)

    # Transposed convolution op with a zero kernel.
    transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=(1, kernel_size),
        strides=(1, kernel_size),
        padding='valid',
        use_bias=False,
        kernel_initializer='zeros')

    upsampled = gp.upsample_transposed_convolution(
        data,
        pool_map,
        sizes=None,
        kernel_size=kernel_size,
        transposed_convolution_op=transposed_convolution_op)

    # Initializes variables of the transpose conv layer.
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllEqual(
        tf.shape(input=upsampled), (num_vertices * kernel_size, num_features))
    self.assertAllEqual(upsampled, tf.zeros_like(upsampled))

  @parameterized.parameters(
      itertools.product((3,), (6,), (3,), list(range(3)), list(range(6)),
                        list(range(6))),)
  def test_upsample_transposed_convolution_selector_kernel_random(
      self, num_vertices, num_features, kernel_size, kernel_index,
      feature1_index, feature2_index):
    """Tests the upsampling with an indicator kernel."""
    data = np.random.uniform(size=(num_vertices,
                                   num_features)).astype(np.float32)
    pool_map = np.zeros(
        shape=(num_vertices, num_vertices * kernel_size), dtype=np.float32)
    for i in range(num_vertices):
      pool_map[i, np.arange(kernel_size * i, kernel_size *
                            (i + 1))] = (1.0 / kernel_size)
    pool_map = _dense_to_sparse(pool_map)

    selection = np.zeros(
        shape=(1, kernel_size, num_features, num_features), dtype=np.float32)
    selection[0, kernel_index, feature1_index, feature2_index] = 1.
    initializer = tf.constant_initializer(value=selection)
    transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
        filters=num_features,
        kernel_size=(1, kernel_size),
        strides=(1, kernel_size),
        padding='valid',
        use_bias=False,
        kernel_initializer=initializer)

    true = np.zeros(
        shape=(num_vertices * kernel_size, num_features), dtype=np.float32)
    input_column = feature2_index
    output_column = feature1_index
    output_row_start = kernel_index
    true[output_row_start::kernel_size, output_column] = (data[:, input_column])
    upsampled = gp.upsample_transposed_convolution(
        data,
        pool_map,
        sizes=None,
        kernel_size=kernel_size,
        transposed_convolution_op=transposed_convolution_op)

    # Initializes variables of the transpose conv layer.
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllEqual(upsampled, true)

  def test_upsample_transposed_convolution_preset_padded(self):
    """Tests upsampling with presets."""
    data = np.reshape(np.arange(12).astype(np.float32), (2, 3, 2))
    data[0, -1, :] = 0.
    sizes = ((2, 3), (3, 3))
    pool_map = _dense_to_sparse(
        np.array((((0.5, 0.5, 0.), (0., 0., 1.), (0., 0., 0.)),
                  ((1., 0., 0.), (0., 1., 0.), (0., 0., 1.))),
                 dtype=np.float32))

    kernel = np.ones(shape=(1, 2, 2, 2), dtype=np.float32)
    initializer = tf.constant_initializer(value=kernel)
    transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
        filters=2,
        kernel_size=(1, 2),
        strides=(1, 2),
        padding='valid',
        use_bias=False,
        kernel_initializer=initializer)

    # Convolving with an all-ones kernel is equal to summation of the input.
    data_sum = np.tile(np.sum(data, axis=-1, keepdims=True), (1, 1, 2))
    true = np.zeros(shape=(2, 3, 2), dtype=np.float32)
    true[0, :, :] = data_sum[0, (0, 0, 1), :]
    true[1, :, :] = data_sum[1, :, :]
    upsampled = gp.upsample_transposed_convolution(
        data,
        pool_map,
        sizes=sizes,
        kernel_size=2,
        transposed_convolution_op=transposed_convolution_op)

    # Initializes variables of the transpose conv layer.
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assertAllEqual(upsampled.shape, (2, 3, 2))
    self.assertAllClose(upsampled, true)

  def test_upsample_transposed_convolution_jacobian_random(self):
    """Tests the jacobian is correct."""
    num_filters = 6
    kernel_size = 1
    data_init = np.random.uniform(size=(2, 5, num_filters))
    pool_map = _batch_sparse_eye((2,), 5, np.float64)
    transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=(1, kernel_size),
        strides=(1, kernel_size),
        padding='valid',
        dtype='float64')

    # Calling the upsample_transposed_convolution to create the variables
    # in the transposed_convoution.
    gp.upsample_transposed_convolution(
        data_init,
        pool_map,
        sizes=None,
        kernel_size=kernel_size,
        transposed_convolution_op=transposed_convolution_op)

    def gp_upsample_transposed_convolution(data):
      return gp.upsample_transposed_convolution(
          data,
          pool_map,
          sizes=None,
          kernel_size=kernel_size,
          transposed_convolution_op=transposed_convolution_op)

    # Initializes variables of the transpose conv layer.
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assert_jacobian_is_correct_fn(gp_upsample_transposed_convolution,
                                       [data_init])

  def test_upsample_transposed_convolution_jacobian_random_padding(self):
    """Tests the jacobian is correct with padded data."""
    num_filters = 6
    sizes = ((2, 4), (3, 5))
    data_init = np.random.uniform(size=(2, 3, num_filters))
    data_init[0, -1, :] = 0.
    pool_map = np.array(
        (((0.5, 0.5, 0., 0., 0.), (0., 0., 0.5, 0.5, 0.), (0., 0., 0., 0., 0.)),
         ((1., 0., 0., 0., 0.), (0., 1. / 3., 1. / 3., 1. / 3., 0.),
          (0., 0., 0., 0., 1.))),
        dtype=data_init.dtype)
    pool_map = _dense_to_sparse(pool_map)
    kernel_size = 2
    transposed_convolution_op = tf.keras.layers.Conv2DTranspose(
        filters=num_filters,
        kernel_size=(1, kernel_size),
        strides=(1, kernel_size),
        padding='valid',
        dtype='float64')

    # Calling the upsample_transposed_convolution to create the variables
    # in the transposed_convoution.
    gp.upsample_transposed_convolution(
        data_init,
        pool_map,
        sizes=sizes,
        kernel_size=kernel_size,
        transposed_convolution_op=transposed_convolution_op)

    def gp_upsample_transposed_convolution(data):
      return gp.upsample_transposed_convolution(
          data,
          pool_map,
          sizes=sizes,
          kernel_size=kernel_size,
          transposed_convolution_op=transposed_convolution_op)

    # Initializes variables of the transpose conv layer.
    self.evaluate(tf.compat.v1.global_variables_initializer())

    self.assert_jacobian_is_correct_fn(gp_upsample_transposed_convolution,
                                       [data_init])


if __name__ == '__main__':
  test_case.main()
