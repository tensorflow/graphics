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
"""Tests for graph convolution ops."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import itertools

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

import tensorflow_graphics.geometry.convolution.graph_convolution as gc
from tensorflow_graphics.util import test_case


def _dense_to_sparse(data):
  """Convert a numpy array to a tf.SparseTensor."""
  indices = np.where(data)
  return tf.SparseTensor(
      np.stack(indices, axis=-1), data[indices], dense_shape=data.shape)


def _dummy_data(batch_size, num_vertices, num_channels):
  """Create inputs for feature_steered_convolution."""
  if batch_size > 0:
    data = np.zeros(
        shape=(batch_size, num_vertices, num_channels), dtype=np.float32)
    neighbors = _dense_to_sparse(
        np.tile(np.eye(num_vertices, dtype=np.float32), (batch_size, 1, 1)))
  else:
    data = np.zeros(shape=(num_vertices, num_channels), dtype=np.float32)
    neighbors = _dense_to_sparse(np.eye(num_vertices, dtype=np.float32))
  return data, neighbors


def _dummy_variables(in_channels, out_channels, num_weight_matrices):
  """Create variable substitutes for feature_steered_convolution."""
  var_u = tf.zeros(shape=(in_channels, num_weight_matrices))
  var_v = tf.zeros(shape=(in_channels, num_weight_matrices))
  var_c = tf.zeros(shape=(num_weight_matrices))
  var_w = tf.zeros(shape=(num_weight_matrices, in_channels, out_channels))
  var_b = tf.zeros(shape=(out_channels))
  return var_u, var_v, var_c, var_w, var_b


def _random_data(batch_size,
                 num_vertices,
                 num_channels,
                 padding,
                 only_self_edges,
                 data_type=np.float32,
                 neighbors_type=np.float32,
                 sizes_type=np.int32):
  """Create random inputs for feature_steered_convolution."""

  def _random_data_2d(padding):
    size = num_vertices if not padding else np.random.randint(
        low=1, high=num_vertices + 1)
    data = np.random.uniform(size=(size, num_channels)).astype(data_type)
    if only_self_edges:
      neighbors = np.eye(size, dtype=neighbors_type)
    else:
      random = np.random.uniform(size=(size, size)).astype(neighbors_type)
      neighbors = np.maximum(
          np.where(random > 0.75, np.ones_like(random), np.zeros_like(random)),
          np.eye(size, dtype=neighbors_type))
      neighbors = neighbors / np.sum(neighbors, axis=1, keepdims=True)
    if padding:
      data = np.pad(data, ((0, num_vertices - size), (0, 0)), "constant")
      neighbors = np.pad(neighbors,
                         ((0, num_vertices - size), (0, num_vertices - size)),
                         "constant")
      return data, neighbors, size
    else:
      return data, neighbors

  if batch_size > 0:
    list_2d = [_random_data_2d(padding=padding) for _ in range(batch_size)]
    data = np.stack([i[0] for i in list_2d], 0).astype(data_type)
    neighbors = np.stack([i[1] for i in list_2d], 0).astype(neighbors_type)
    if padding:
      sizes = np.stack([i[2] for i in list_2d], 0).astype(sizes_type)
      return data, _dense_to_sparse(neighbors), sizes
    else:
      return data, _dense_to_sparse(neighbors)
  else:
    if padding:
      raise ValueError("Padding only allowed with batched data.")
    data, neighbors = _random_data_2d(padding=False)
    return data.astype(data_type), _dense_to_sparse(
        neighbors.astype(neighbors_type))


def _random_variables(in_channels,
                      out_channels,
                      num_weight_matrices,
                      dtype=np.float32):
  """Create random variables for feature_steered_convolution."""

  def _random_constant(shape, dtype):
    return tf.constant(np.random.uniform(size=shape).astype(dtype))

  var_u = _random_constant([in_channels, num_weight_matrices], dtype)
  var_v = _random_constant([in_channels, num_weight_matrices], dtype)
  var_c = _random_constant([num_weight_matrices], dtype)
  var_w = _random_constant([num_weight_matrices, in_channels, out_channels],
                           dtype)
  var_b = _random_constant([out_channels], dtype)
  return var_u, var_v, var_c, var_w, var_b


class GraphConvolutionTestFeatureSteeredConvolutionTests(test_case.TestCase):

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32,
       np.float32),
      ("'neighbors' and 'data' must have the same type.", np.float32,
       np.float64, np.int32, np.float32),
  )
  def test_feature_steered_convolution_exception_raised_types(
      self, err_msg, data_type, neighbors_type, sizes_type, var_type):
    """Check the type errors for invalid input types."""
    data, neighbors, sizes = _random_data(1, 5, 3, True, False, data_type,
                                          neighbors_type, sizes_type)
    u, v, c, w, b = _random_variables(3, 3, 1, var_type)
    with self.assertRaisesRegexp(TypeError, err_msg):
      _ = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

  @parameterized.parameters(
      (np.float32, np.float32, np.int32, np.float32),
      (np.float64, np.float64, np.int32, np.float64),
      (np.float32, np.float32, np.int64, np.float32),
      (np.float64, np.float64, np.int64, np.float64),
  )
  def test_feature_steered_convolution_exception_not_raised_types(
      self, data_type, neighbors_type, sizes_type, var_type):
    """Check there are no exceptions for valid input types."""
    data, neighbors, sizes = _random_data(1, 5, 3, True, False, data_type,
                                          neighbors_type, sizes_type)
    u, v, c, w, b = _random_variables(3, 3, 1, var_type)
    try:
      gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_feature_steered_convolution_exception_raised_shapes(self):
    """Check that invalid input shapes trigger the right exceptions."""
    with self.assertRaisesRegexp(ValueError, "must have a rank of 2"):
      data, neighbors = _dummy_data(1, 5, 2)
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      data = data[0, :]
      _ = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

    with self.assertRaisesRegexp(ValueError, "must have a rank greater than 1"):
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      data = np.ones(shape=(5), dtype=np.float32)
      neighbors = _dense_to_sparse(np.ones(shape=(5), dtype=np.float32))
      _ = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

    with self.assertRaisesRegexp(ValueError,
                                 "Not all batch dimensions are identical."):
      data, neighbors = _dummy_data(1, 5, 2)
      u, v, c, w, b = _dummy_variables(2, 2, 1)
      _ = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=(1, 1),
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

  @parameterized.parameters(
      (1, 1, 1, 1, 1),
      (4, 2, 3, 6, 5),
      (0, 1, 1, 1, 1),
      (0, 2, 3, 6, 5),
  )
  def test_feature_steered_convolution_output_shape(self, batch_size,
                                                    num_vertices, in_channels,
                                                    out_channels,
                                                    num_weight_matrices):
    """Check that the output of convolution has the correct shape."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)
    u, v, c, w, b = _dummy_variables(in_channels, out_channels,
                                     num_weight_matrices)

    y = gc.feature_steered_convolution(
        data=data,
        neighbors=neighbors,
        sizes=None,
        var_u=u,
        var_v=v,
        var_c=c,
        var_w=w,
        var_b=b)
    y_shape = y.shape.as_list()

    self.assertEqual(y_shape[-1], out_channels)
    self.assertAllEqual(y_shape[:-1], data.shape[:-1])

  @parameterized.parameters(
      (1, 1, 1, 1, 1),
      (4, 2, 3, 6, 5),
      (0, 1, 1, 1, 1),
      (0, 2, 3, 6, 5),
  )
  def test_feature_steered_convolution_only_self_edges(self, batch_size,
                                                       num_vertices,
                                                       in_channels,
                                                       out_channels,
                                                       num_weight_matrices):
    """Test convolution when the graph only has self edges."""
    data, neighbors = _random_data(
        batch_size,
        num_vertices,
        in_channels,
        padding=False,
        only_self_edges=True)
    u, v, c, w, b = _random_variables(in_channels, out_channels,
                                      num_weight_matrices)

    with self.subTest(name="w=0_expect_output=b"):
      y = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=tf.zeros_like(w),
          var_b=b)
      y_expected = tf.broadcast_to(b, y.shape)

      self.assertAllEqual(y, y_expected)

    with self.subTest(name="translation_invariant_self_edges"):
      y = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=-u,
          var_c=c,
          var_w=w,
          var_b=b)
      q = tf.reshape(
          tf.exp(c) / tf.reduce_sum(input_tensor=tf.exp(c)),
          (num_weight_matrices, 1, 1))
      if batch_size > 0:
        q_times_w = tf.reduce_sum(input_tensor=q * w, axis=0, keepdims=True)
        q_times_w = tf.tile(q_times_w, (batch_size, 1, 1))
      else:
        q_times_w = tf.reduce_sum(input_tensor=q * w, axis=0)
      y_expected = tf.matmul(data, q_times_w) + tf.broadcast_to(b, y.shape)

      self.assertAllClose(y, y_expected)

    with self.subTest(name="constant_signal"):
      if batch_size > 0:
        constant_data = np.tile(
            np.random.uniform(size=(batch_size, 1,
                                    in_channels)).astype(np.float32),
            (1, num_vertices, 1))
      else:
        constant_data = np.tile(
            np.random.uniform(size=(1, in_channels)).astype(np.float32),
            (num_vertices, 1))
      y = gc.feature_steered_convolution(
          data=constant_data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)
      if batch_size > 0:
        y_expected = tf.tile(y[:, :1, :], (1, num_vertices, 1))
      else:
        y_expected = tf.tile(y[:1, :], (num_vertices, 1))

      self.assertAllClose(y, y_expected)

  @parameterized.parameters(
      (((1.0,), (2.0,), (3.0,)), np.ones(shape=(3, 3)) / 3.0, ((0.5,),),
       ((1.3,),), (-0.7,), (((0.8,),),), (3.0,), ((4.6,), (4.6,), (4.6,))),
      (((1.0,), (2.0,), (3.0,)), np.ones(shape=(3, 3)) / 3.0, ((0.5, 0.2),),
       ((0.3, 0.4),), (-0.7, 0.15), (((0.8,),), ((1.1,),)), (3.0,),
       ((5.011706928844621,), (4.971030281984818,), (4.927388658982911,))),
  )
  def test_feature_steered_convolution_padding_preset(self, data, neighbors, u,
                                                      v, c, w, b, expected):
    """Test expected result for preset data and filter values."""
    array = (np.array(i) for i in (data, neighbors, expected))
    data, neighbors, expected = array
    tensors = (tf.convert_to_tensor(value=np.array(i).astype(data.dtype)) \
               for i in (u, v, c, w, b))
    u, v, c, w, b = tensors
    y = gc.feature_steered_convolution(
        data=data,
        neighbors=_dense_to_sparse(neighbors),
        sizes=None,
        var_u=u,
        var_v=v,
        var_c=c,
        var_w=w,
        var_b=b)
    self.assertAllClose(y, expected)

  @parameterized.parameters(
      (1, 5, 1, 1, 1),
      (2, 6, 3, 6, 5),
      (5, 15, 6, 12, 8),
  )
  def test_feature_steered_convolution_padding_random(self, batch_size,
                                                      num_vertices, in_channels,
                                                      out_channels,
                                                      num_weight_matrices):
    """Test mixed topology batches (random vertices and neighbors)."""
    data, neighbors, sizes = _random_data(
        batch_size,
        num_vertices,
        in_channels,
        padding=True,
        only_self_edges=False)
    u, v, c, w, b = _random_variables(in_channels, out_channels,
                                      num_weight_matrices)

    with self.subTest(name="if_w_is_0_then_y_is_b"):
      y = gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=tf.zeros_like(w),
          var_b=b)
      for k in range(batch_size):
        y_crop = y[k, :sizes[k], :]
        y_expected = tf.broadcast_to(b, y_crop.shape)

        self.assertAllEqual(y_crop, y_expected)
        # Check for zeros in the padded region.
        self.assertAllEqual(y[k, sizes[k]:, :],
                            tf.zeros((num_vertices - sizes[k], out_channels)))

    with self.subTest(name="convolve_with_constant"):
      constant_data = data
      for k in range(batch_size):
        constant_data[k, :sizes[k], :] = np.tile(data[k, 0, :], (sizes[k], 1))

      y = gc.feature_steered_convolution(
          data=constant_data,
          neighbors=neighbors,
          sizes=sizes,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)
      for k in range(batch_size):
        y_crop = y[k, :sizes[k], :]
        y_const = tf.broadcast_to(y_crop[0, :], y_crop.shape)

        self.assertAllClose(y_crop, y_const)
        # Check for zeros in the padded region.
        self.assertAllEqual(y[k, sizes[k]:, :],
                            tf.zeros([num_vertices - sizes[k], out_channels]))

  @parameterized.parameters(
      (1, 10, 3, 1, True),
      (3, 6, 1, 4, True),
      (0, 10, 5, 2, False),
      (1, 10, 3, 1, False),
      (3, 6, 1, 4, False),
      (0, 10, 5, 2, False),
  )
  def test_feature_steered_convolution_jacobian_random(self, batch_size,
                                                       num_vertices,
                                                       in_channels,
                                                       num_weight_matrices,
                                                       padding):
    """Test the jacobian for random input data."""
    random_data = _random_data(
        batch_size,
        num_vertices,
        in_channels,
        padding,
        only_self_edges=False,
        data_type=np.float64,
        neighbors_type=np.float64)
    data_init = random_data[0]
    neighbors = random_data[1]
    sizes = None if not padding else random_data[2]
    u, v, c, w, b = _random_variables(
        in_channels, in_channels, num_weight_matrices, dtype=np.float64)

    def feature_steered_convolution(data):
      return gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

    self.assert_jacobian_is_correct_fn(feature_steered_convolution, [data_init])

  @parameterized.parameters(
      (1, 1, 0.0),
      (5, 1, 0.0),
      (1, 3, 0.0),
      (5, 3, 0.0),
      (1, 1, 1.0),
      (5, 1, 1.0),
      (1, 3, 1.0),
      (5, 3, 1.0),
  )
  def test_feature_steered_convolution_jacobian_preset(self, num_vertices,
                                                       num_channels,
                                                       data_multiplier):
    """Test the jacobian is correct for preset inputs."""
    # Corner cases include one vertex, one channel, and all-zero features.
    data_init = data_multiplier * np.random.uniform(
        size=(num_vertices, num_channels)).astype(np.float64)
    neighbors = tf.sparse.eye(num_vertices, dtype=tf.float64)
    u, v, c, w, b = _random_variables(
        num_channels, num_channels, 1, dtype=np.float64)

    def feature_steered_convolution(data):
      return gc.feature_steered_convolution(
          data=data,
          neighbors=neighbors,
          sizes=None,
          var_u=u,
          var_v=v,
          var_c=c,
          var_w=w,
          var_b=b)

    self.assert_jacobian_is_correct_fn(feature_steered_convolution, [data_init])


class EdgeConvolutionTemplateTests(test_case.TestCase):

  def _zeros(self, vertex_features, _, out_dimensions=None):
    """A callable for `edge_convolution_template`."""
    if out_dimensions is None:
      return tf.zeros_like(vertex_features)
    else:
      return tf.zeros(
          shape=(vertex_features.shape.as_list()[0], out_dimensions),
          dtype=vertex_features.dtype)

  def _pass_through(self, _, neighbor_features):
    """A callable for `edge_convolution_template`."""
    return neighbor_features

  def _circular_2d_data(self, num_vertices, include_normals=False):
    """Create data for a circle graph."""
    # Vertices are points distributed uniformly on a circle, with each point
    # connected to its closest neighbor on either side.
    theta = np.linspace(0.0, np.pi * 2.0, num=num_vertices, endpoint=False)
    data = np.stack((np.cos(theta), np.sin(theta)), axis=-1)
    if include_normals:
      data = np.concatenate((data, data), axis=-1)
    eye = np.eye(num_vertices)
    neighbors = np.maximum(np.roll(eye, 1, axis=1), np.roll(eye, -1,
                                                            axis=1)) * 0.5
    return data, _dense_to_sparse(neighbors)

  def _edge_curvature_2d(self, vertex_features, neighbor_features):
    """A callable for `edge_convolution_template` that computes curvature."""
    x_position, x_normal = tf.split(
        value=vertex_features, num_or_size_splits=2, axis=-1)
    y_position, y_normal = tf.split(
        value=neighbor_features, num_or_size_splits=2, axis=-1)
    yx_diff = x_position - y_position
    curvature_unscaled = tf.abs(
        tf.reduce_sum(
            input_tensor=(y_normal - x_normal) * yx_diff,
            axis=-1,
            keepdims=True))
    edge_length_squared = tf.reduce_sum(
        input_tensor=yx_diff * yx_diff, axis=-1, keepdims=True)
    return tf.compat.v1.where(
        tf.less(edge_length_squared, 1e-7), tf.zeros_like(edge_length_squared),
        curvature_unscaled / edge_length_squared)

  @parameterized.parameters(
      ("'sizes' must have an integer type.", np.float32, np.float32,
       np.float32),
      ("'data' must have a float type.", np.int32, np.float32, np.int32),
      ("'neighbors' and 'data' must have the same type.", np.float32,
       np.float64, np.int32),
  )
  def test_edge_convolution_template_exception_raised_types(
      self, err_msg, data_type, neighbors_type, sizes_type):
    """Check the type errors for invalid input types."""
    data, neighbors, sizes = _random_data(1, 5, 3, True, False, data_type,
                                          neighbors_type, sizes_type)
    with self.assertRaisesRegexp(TypeError, err_msg):
      gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          edge_function=self._zeros,
          reduction="weighted",
          edge_function_kwargs=dict())

  @parameterized.parameters(
      (np.float32, np.float32, np.int32),
      (np.float64, np.float64, np.int32),
      (np.float32, np.float32, np.int64),
      (np.float64, np.float64, np.int64),
      (np.float64, np.float64, np.int8),
      (np.float64, np.float64, np.uint8),
      (np.float64, np.float64, np.int16),
      (np.float64, np.float64, np.uint16),
  )
  def test_edge_convolution_template_exception_not_raised_types(
      self, data_type, neighbors_type, sizes_type):
    """Check there are no exceptions for valid input types."""
    data, neighbors, sizes = _random_data(1, 5, 3, True, False, data_type,
                                          neighbors_type, sizes_type)
    try:
      gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          edge_function=self._zeros,
          reduction="weighted",
          edge_function_kwargs=dict())
    except Exception as e:  # pylint: disable=broad-except
      self.fail("Exception raised: %s" % str(e))

  def test_edge_convolution_template_exception_raised_shapes(self):
    """Check that invalid input shapes trigger the right exceptions."""
    with self.assertRaisesRegexp(ValueError, "must have a rank of 2"):
      data, neighbors = _dummy_data(1, 5, 2)
      data = data[0, :]
      _ = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._zeros,
          reduction="weighted",
          edge_function_kwargs=dict())

    with self.assertRaisesRegexp(ValueError, "must have a rank greater than 1"):
      data = np.ones(shape=(5), dtype=np.float32)
      neighbors = _dense_to_sparse(np.ones(shape=(5), dtype=np.float32))
      _ = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._zeros,
          reduction="weighted",
          edge_function_kwargs=dict())

    with self.assertRaisesRegexp(ValueError, "must have a rank of 1"):
      data, neighbors = _dummy_data(1, 5, 2)
      _ = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=((1, 1), (1, 1)),
          edge_function=self._zeros,
          reduction="weighted",
          edge_function_kwargs=dict())

  @parameterized.parameters("", "invalid")
  def test_edge_convolution_template_exception_raised_reduction(
      self, reduction):
    """Check that an invalid reduction method triggers the exception."""
    with self.assertRaisesRegexp(ValueError, "reduction method"):
      data, neighbors = _dummy_data(1, 5, 2)
      gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._zeros,
          reduction=reduction,
          edge_function_kwargs=dict())

  @parameterized.parameters(
      (1, 1, 1, 1, "weighted"),
      (4, 2, 3, 6, "weighted"),
      (0, 1, 1, 1, "max"),
      (0, 2, 3, 6, "max"),
  )
  def test_edge_convolution_template_output_shape(self, batch_size,
                                                  num_vertices, in_channels,
                                                  out_channels, reduction):
    """Check that the output of convolution has the correct shape."""
    data, neighbors = _dummy_data(batch_size, num_vertices, in_channels)

    y = gc.edge_convolution_template(
        data,
        neighbors,
        None,
        self._zeros,
        reduction=reduction,
        edge_function_kwargs={"out_dimensions": out_channels})
    y_shape = y.shape.as_list()

    with self.subTest(name="out_channels"):
      self.assertEqual(y_shape[-1], out_channels)

    with self.subTest(name="shape"):
      self.assertAllEqual(y_shape[:-1], data.shape[:-1])

  def test_edge_convolution_template_zero_neighbors(self):
    """Check that vertices with no neighbors map to zeros in the output."""
    # We can reuse `self._edge_curvature_2d` as the curvature functional.
    num_vertices = 500
    data, neighbors = self._circular_2d_data(num_vertices, include_normals=True)

    # Interleave the data with rows filled with random data, these rows will
    # have no neighbors in the adjacency matrix so should map to all zeros in
    # the output.
    rows_odd = tf.expand_dims(
        tf.range(start=1, limit=(2 * num_vertices), delta=2), -1)
    rows_even = tf.expand_dims(
        tf.range(start=0, limit=(2 * num_vertices + 1), delta=2), -1)
    data_interleaved = tf.scatter_nd(
        indices=rows_odd,
        updates=data,
        shape=(2 * num_vertices + 1, tf.shape(input=data)[-1]))
    random_data = tf.random.uniform(
        shape=(data.shape[0] + 1, data.shape[-1]), dtype=data.dtype)
    random_interleaved = tf.scatter_nd(
        indices=rows_even,
        updates=random_data,
        shape=(2 * num_vertices + 1, tf.shape(input=data)[-1]))
    data_interleaved = data_interleaved + random_interleaved
    neighbors_interleaved_indices = neighbors.indices * 2 + 1
    neighbors_interleaved = tf.SparseTensor(
        indices=neighbors_interleaved_indices,
        values=neighbors.values,
        dense_shape=(2 * num_vertices + 1, 2 * num_vertices + 1))

    # Convolve the interleaved data.
    data_curvature = gc.edge_convolution_template(
        data=data_interleaved,
        neighbors=neighbors_interleaved,
        sizes=None,
        edge_function=self._edge_curvature_2d,
        reduction="weighted",
        edge_function_kwargs=dict())

    self.assertEqual(data_curvature.shape, (2 * num_vertices + 1, 1))

    # The rows corresponding to the original input data measure the curvature.
    # The curvature at any point on a circle of radius 1 should be 1.
    # The interleaved rows of random data should map to zeros in the output.
    self.assertAllClose(data_curvature[1::2, :],
                        np.ones(shape=(num_vertices, 1)))
    self.assertAllClose(data_curvature[::2, :],
                        np.zeros(shape=(num_vertices + 1, 1)))

  @parameterized.parameters(
      (1, 10, 3, True, "weighted"),
      (3, 6, 1, True, "weighted"),
      (0, 10, 5, False, "weighted"),
      (1, 10, 3, False, "max"),
      (3, 6, 1, False, "max"),
      (0, 10, 5, False, "max"),
  )
  def test_edge_convolution_template_jacobian_random(self, batch_size,
                                                     num_vertices, in_channels,
                                                     padding, reduction):
    """Test the jacobian for random input data."""
    random_data = _random_data(
        batch_size,
        num_vertices,
        in_channels,
        padding,
        only_self_edges=False,
        data_type=np.float64,
        neighbors_type=np.float64)
    data_init = random_data[0]
    neighbors = random_data[1]
    sizes = None if not padding else random_data[2]

    def edge_convolution_template(data):
      return gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=sizes,
          edge_function=self._pass_through,
          reduction=reduction,
          edge_function_kwargs=dict())

    self.assert_jacobian_is_correct_fn(edge_convolution_template, [data_init])

  def test_edge_convolution_template_preset_max(self):
    data = np.array(((1, 2), (3, 4), (5, 6), (7, 8)), np.float32)
    neighbors = np.array(
        ((0, 1, 0, 1), (0, 0, 1, 0), (1, 1, 1, 0), (0, 0, 1, 1)), np.float32)
    neighbors = _dense_to_sparse(neighbors)
    true = np.array(((8, 10), (8, 10), (10, 12), (14, 16)), np.float32)

    with self.subTest("max_sum"):
      max_sum = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=lambda x, y: x + y,
          reduction="max",
          edge_function_kwargs=dict())

      self.assertAllEqual(max_sum, true)

    with self.subTest("max_sum_scaled"):
      # Max reduction ignores the weights, so scaling the neighbors weights
      # should not change the result.
      max_sum_scaled = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors * 10.0,
          sizes=None,
          edge_function=lambda x, y: x + y,
          reduction="max",
          edge_function_kwargs=dict())

      self.assertAllEqual(max_sum_scaled, true)

  @parameterized.parameters(
      itertools.product((1, 5), (1, 3), (0.0, 1.0), ("weighted", "max")))
  def test_edge_convolution_template_jacobian_preset(self, num_vertices,
                                                     num_channels,
                                                     data_multiplier,
                                                     reduction):
    """Test the jacobian is correct for preset inputs."""
    # Corner cases include one vertex, one channel, and all-zero features.
    data_init = data_multiplier * np.random.uniform(
        size=(num_vertices, num_channels)).astype(np.float64)
    neighbors = tf.sparse.eye(num_vertices, dtype=tf.float64)

    def edge_convolution_template(data):
      return gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._pass_through,
          reduction=reduction,
          edge_function_kwargs=dict())

    self.assert_jacobian_is_correct_fn(edge_convolution_template, [data_init])

  def test_edge_convolution_template_laplacian_smoothing(self):
    r"""Test the expected result with laplacian smoothing.

      Laplacian smoothing for meshes is defined as
      $$y_i = \frac{1}{|\mathcal{N(i)}|} \sum_{j \in \mathcal{N(i)}} x_j$$

      This can be computed using `edge_convolution_template` with `f(x, y)->y`.
    """

    # We can reuse `self._pass_through(x, y)->y` as the smoothing functional.
    with self.subTest(name="only_self_edges_random"):
      num_vertices = 500
      data = np.random.uniform(size=(num_vertices, 5))
      neighbors = tf.sparse.eye(num_vertices, dtype=tf.as_dtype(data.dtype))

      data_smoothed = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._pass_through,
          reduction="weighted",
          edge_function_kwargs=dict())

      self.assertAllEqual(data, data_smoothed)

    with self.subTest(name="circular_2d"):
      num_vertices = 500
      data, neighbors = self._circular_2d_data(num_vertices)

      data_smoothed = gc.edge_convolution_template(
          data=data,
          neighbors=neighbors,
          sizes=None,
          edge_function=self._pass_through,
          reduction="weighted",
          edge_function_kwargs=dict())
      # The smoothed points should have the same direction as the originals.
      data_smoothed_normalized = tf.nn.l2_normalize(data_smoothed, axis=-1)

      self.assertAllClose(data, data_smoothed_normalized)

  def test_edge_convolution_template_curvature(self):
    r"""Test the expected result with curvature.

      (Approximate) curvature for meshes is defined as
      $$\kappa_{v_i} = \frac{1}{|\mathcal{N}(v_i)|}
        \sum_{v_j \in \mathcal{N}(v_i)}
        \frac{(\vec{v_i} - \vec{v_j})^T (\vec{n_{v_i}} -
        \vec{n_{v_j}})} {\left|\vec{v_i}-\vec{v_j}\right|^2}
      $$

      This can be computed using `edge_convolution_template` with
        $$f(x, y) = (n_x - n_y)^T (x - y) / ||x - y||^2.$$
      where $$n_x$$ and $$n_y$$ are the normals at points $$x$$ and $$y$$
      respectively.
    """
    # We can reuse `self._edge_curvature_2d` as the curvature functional.
    num_vertices = 500
    data, neighbors = self._circular_2d_data(num_vertices, include_normals=True)

    data_curvature = gc.edge_convolution_template(
        data=data,
        neighbors=neighbors,
        sizes=None,
        edge_function=self._edge_curvature_2d,
        reduction="weighted",
        edge_function_kwargs=dict())

    # The curvature at each point on a circle of radius 1 should be 1.
    self.assertAllClose(data_curvature, np.ones(shape=(num_vertices, 1)))


if __name__ == "__main__":
  test_case.main()
