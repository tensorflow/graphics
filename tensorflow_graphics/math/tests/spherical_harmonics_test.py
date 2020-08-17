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
"""Tests for spherical harmonics."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math import math_helpers
from tensorflow_graphics.math import spherical_harmonics
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import test_case


class SphericalHarmonicsTest(test_case.TestCase):

  def test_evaluate_legendre_polynomial_preset(self):
    """Tests that evaluate_legendre_polynomial generates expected results."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    x = np.random.uniform(size=tensor_shape)

    with self.subTest(name="l_0_m_0"):
      l = tf.constant(0, shape=tensor_shape)
      m = tf.constant(0, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = np.ones_like(x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_1_m_0"):
      l = tf.constant(1, shape=tensor_shape)
      m = tf.constant(0, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = x

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_1_m_1"):
      l = tf.constant(1, shape=tensor_shape)
      m = tf.constant(1, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = -tf.sqrt(1.0 - x * x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_2_m_0"):
      l = tf.constant(2, shape=tensor_shape)
      m = tf.constant(0, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = 0.5 * (3.0 * x * x - 1.0)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_2_m_1"):
      l = tf.constant(2, shape=tensor_shape)
      m = tf.constant(1, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = -3.0 * x * tf.sqrt(1.0 - x * x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_2_m_2"):
      l = tf.constant(2, shape=tensor_shape)
      m = tf.constant(2, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = 3.0 * (1.0 - x * x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_3_m_0"):
      l = tf.constant(3, shape=tensor_shape)
      m = tf.constant(0, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = 0.5 * x * (5.0 * x * x - 3.0)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_3_m_1"):
      l = tf.constant(3, shape=tensor_shape)
      m = tf.constant(1, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = 1.5 * (1.0 - 5.0 * x * x) * tf.sqrt(1.0 - x * x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_3_m_2"):
      l = tf.constant(3, shape=tensor_shape)
      m = tf.constant(2, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = 15.0 * x * (1.0 - x * x)

      self.assertAllClose(pred, gt)

    with self.subTest(name="l_3_m_3"):
      l = tf.constant(3, shape=tensor_shape)
      m = tf.constant(3, shape=tensor_shape)
      pred = spherical_harmonics.evaluate_legendre_polynomial(l, m, x)
      gt = -15.0 * tf.pow(1.0 - x * x, 1.5)

      self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ("Not all batch dimensions are broadcast-compatible", tf.int32, tf.int32,
       (3, 3), (3, 3), (2, 3)),
      ("Not all batch dimensions are broadcast-compatible", tf.int32, tf.int32,
       (None, 5, 2), (None, 1, 2), (3, 1)),
      ("must be of an integer type", tf.float32, tf.int32, (3,), (3,), (1,)),
      ("must be of an integer type", tf.int32, tf.float32, (3,), (3,), (1,)),
  )
  def test_evaluate_legendre_polynomial_raised(self, error_msg, degree_dtype,
                                               order_dtype, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        spherical_harmonics.evaluate_legendre_polynomial,
        error_msg,
        shapes,
        dtypes=(degree_dtype, order_dtype, tf.float32))

  def test_evaluate_legendre_polynomial_exceptions_l_raised(self):
    """Tests that an exception is raised when l is not in the expected range."""
    l = np.random.randint(-10, -1, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    x = np.random.uniform(size=(1,)) * 2.0

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(spherical_harmonics.evaluate_legendre_polynomial(l, m, x))

  def test_evaluate_legendre_polynomial_exceptions_m_raised(self):
    """Tests that an exception is raised when m is not in the expected range."""
    l = np.random.randint(1, 4, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    x = np.random.uniform(size=(1,)) * 2.0

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(spherical_harmonics.evaluate_legendre_polynomial(l, m, x))

  def test_evaluate_legendre_polynomial_exceptions_x_raised(self):
    """Tests that an exception is raised when x is not in the expected range."""
    l = np.random.randint(4, 10, size=(1,))
    m = np.random.randint(0, 4, size=(1,))
    x = np.random.uniform(
        1.0 + asserts.select_eps_for_addition(tf.float32), 2.0, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(spherical_harmonics.evaluate_legendre_polynomial(l, m, x))

  def test_evaluate_legendre_polynomial_jacobian_random(self):
    """Tests the Jacobian of evaluate_legendre_polynomial."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    l_init = np.random.randint(5, 10, size=tensor_shape)
    m_init = np.random.randint(0, 4, size=tensor_shape)
    x_init = np.random.uniform(-1.0, 1.0, size=tensor_shape)

    def evaluate_legendre_polynomial_fn(x):
      return spherical_harmonics.evaluate_legendre_polynomial(l_init, m_init, x)

    self.assert_jacobian_is_correct_fn(
        evaluate_legendre_polynomial_fn, [x_init], atol=1e-3)

  @parameterized.parameters(
      ((4,), (4,)),
      ((None, 9), (None, 9)),
      ((None, 3, 9), (None, 1, 9)),
  )
  def test_integration_product_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(spherical_harmonics.integration_product,
                                        shape)

  @parameterized.parameters(
      ("must have the same number of dimensions in axes", (1,), (2,)),)
  def test_integration_product_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(spherical_harmonics.integration_product,
                                    error_msg, shape)

  def test_generate_l_m_permutations_preset(self):
    """Tests that generate_l_m_permutations produces the expected results."""
    l, m = spherical_harmonics.generate_l_m_permutations(2)

    self.assertAllEqual(l, (0, 1, 1, 1, 2, 2, 2, 2, 2))
    self.assertAllEqual(m, (0, -1, 0, 1, -2, -1, 0, 1, 2))

  def test_generate_l_m_zonal_preset(self):
    """Tests that generate_l_m_zonal produces the expected results."""
    l, m = spherical_harmonics.generate_l_m_zonal(2)

    self.assertAllEqual(l, (0, 1, 2))
    self.assertAllEqual(m, (0, 0, 0))

  @parameterized.parameters(
      (((2, -1)), (0.25752,)),
      (((8, 6)), (5.5710e-06,)),
      (((8, -6)), (5.5710e-06,)),
  )
  def test_spherical_harmonics_normalization_preset(self, test_inputs,
                                                    test_outputs):
    self.assert_output_is_correct(
        spherical_harmonics._spherical_harmonics_normalization, test_inputs,
        test_outputs)

  @parameterized.parameters(
      ("must have the same number of dimensions in axes", tf.int32, tf.int32,
       (3,), (2,), (1,), (1,)),
      ("must have exactly 1 dimensions in axis -1", tf.int32, tf.int32, (3,),
       (3,), (1,), (2,)),
      ("must have exactly 1 dimensions in axis -1", tf.int32, tf.int32, (3,),
       (3,), (2, 2), (2, 2)),
      ("Not all batch dimensions are identical.", tf.int32, tf.int32, (3, 3),
       (3, 3), (2, 1), (2, 1)),
      ("Not all batch dimensions are identical.", tf.int32, tf.int32,
       (None, 5, 2), (None, 1, 2), (5, 1), (1, 1)),
      ("must be of an integer type", tf.float32, tf.int32, (3,), (3,), (1,),
       (1,)),
      ("must be of an integer type", tf.int32, tf.float32, (3,), (3,), (1,),
       (1,)),
  )
  def test_evaluate_spherical_harmonics_raised(self, error_msg, degree_dtype,
                                               order_dtype, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        spherical_harmonics.evaluate_spherical_harmonics,
        error_msg,
        shapes,
        dtypes=(degree_dtype, order_dtype, tf.float32, tf.float32))

  def test_evaluate_spherical_harmonics_exception_l_raised(self):
    """Tests that an exception is raised when l is not in the expected range."""
    l = np.random.randint(-10, -1, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    theta = np.random.uniform(0.0, np.pi, size=(1,))
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

  def test_evaluate_spherical_harmonics_exception_m_raised(self):
    """Tests that an exception is raised when m is not in the expected range."""
    l = np.random.randint(1, 4, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    theta = np.random.uniform(0.0, np.pi, size=(1,))
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

    m = np.random.randint(-10, -5, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

  def test_evaluate_spherical_harmonics_exception_theta_raised(self):
    """Tests exceptions on the values of theta."""
    l = np.random.randint(1, 4, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    theta = np.random.uniform(
        np.pi + sys.float_info.epsilon, 2.0 * np.pi, size=(1,))
    phi = np.random.uniform(0.0, 2.0 * np.pi, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

    theta = np.random.uniform(-np.pi, 0.0 - sys.float_info.epsilon, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

  def test_evaluate_spherical_harmonics_exception_phi_raised(self):
    """Tests exceptions on the values of phi."""
    l = np.random.randint(1, 4, size=(1,))
    m = np.random.randint(5, 10, size=(1,))
    theta = np.random.uniform(0.0, np.pi, size=(1,))
    phi = np.random.uniform(
        2.0 * np.pi + sys.float_info.epsilon, 4.0 * np.pi, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

    phi = np.random.uniform(
        -2.0 * np.pi, 0.0 - sys.float_info.epsilon, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi))

  @parameterized.parameters(
      ((3,), (3,), (1,), (1,)),
      ((5, 2), (5, 2), (5, 1), (5, 1)),
  )
  def test_evaluate_spherical_harmonics_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        spherical_harmonics.evaluate_spherical_harmonics,
        shape,
        dtypes=(tf.int32, tf.int32, tf.float32, tf.float32))

  def test_evaluate_spherical_harmonics_preset(self):
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    pt_3d = tf.convert_to_tensor(
        value=np.random.uniform(size=tensor_shape + [3]))
    pt_3d = tf.math.l2_normalize(pt_3d, axis=-1)
    x, y, z = tf.unstack(pt_3d, axis=-1)
    x = tf.expand_dims(x, axis=-1)
    y = tf.expand_dims(y, axis=-1)
    z = tf.expand_dims(z, axis=-1)
    pt_spherical = math_helpers.cartesian_to_spherical_coordinates(pt_3d)
    _, theta, phi = tf.unstack(pt_spherical, axis=-1)
    theta = tf.expand_dims(theta, axis=-1)
    phi = tf.expand_dims(phi, axis=-1)
    ones = tf.ones_like(z)

    with self.subTest(name="l_0_m_0"):
      l = tf.zeros_like(theta, dtype=tf.int32)
      m = tf.zeros_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = ones / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_1_m_-1"):
      l = tf.ones_like(theta, dtype=tf.int32)
      m = -tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = -tf.sqrt(3.0 * ones) * y / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_1_m_0"):
      l = tf.ones_like(theta, dtype=tf.int32)
      m = tf.zeros_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = (tf.sqrt(3.0 * ones) * z) / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_1_m_1"):
      l = tf.ones_like(theta, dtype=tf.int32)
      m = tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = -tf.sqrt(3.0 * ones) * x / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_2_m-2"):
      l = 2 * tf.ones_like(theta, dtype=tf.int32)
      m = -2 * tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = tf.sqrt(15.0 * ones) * y * x / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_2_m_-1"):
      l = 2 * tf.ones_like(theta, dtype=tf.int32)
      m = -tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = -tf.sqrt(15.0 * ones) * y * z / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_2_m_0"):
      l = 2 * tf.ones_like(theta, dtype=tf.int32)
      m = tf.zeros_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = tf.sqrt(
          5.0 * ones) * (3.0 * z * z - ones) / (4.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_2_m_1"):
      l = 2 * tf.ones_like(theta, dtype=tf.int32)
      m = tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = -tf.sqrt(15.0 * ones) * x * z / (2.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

    with self.subTest(name="l_2_m_2"):
      l = 2 * tf.ones_like(theta, dtype=tf.int32)
      m = 2 * tf.ones_like(theta, dtype=tf.int32)
      val = spherical_harmonics.evaluate_spherical_harmonics(l, m, theta, phi)
      gt = tf.sqrt(
          15.0 * ones) * (x * x - y * y) / (4.0 * tf.sqrt(ones * np.pi))

      self.assertAllClose(val, gt)

  def test_evaluate_spherical_harmonics_jacobian_random(self):
    """Test the Jacobian of the evaluate_spherical_harmonics function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    l_init = np.random.randint(5, 10, size=tensor_shape)
    l_init = np.expand_dims(l_init, axis=-1)
    m_init = np.random.randint(0, 4, size=tensor_shape)
    m_init = np.expand_dims(m_init, axis=-1)
    theta_init = np.random.uniform(0.0, np.pi, size=tensor_shape)
    theta_init = np.expand_dims(theta_init, axis=-1)
    phi_init = np.random.uniform(0.0, 2.0 * np.pi, size=tensor_shape)
    phi_init = np.expand_dims(phi_init, axis=-1)

    def evaluate_spherical_harmonics_fn(theta, phi):
      return spherical_harmonics.evaluate_spherical_harmonics(
          l_init, m_init, theta, phi)

    self.assert_jacobian_is_correct_fn(evaluate_spherical_harmonics_fn,
                                       [theta_init, phi_init])

  def test_rotate_zonal_harmonics_jacobian_random(self):
    """Tests the jacobian of rotate_zonal_harmonics."""
    dtype = tf.float64
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    theta_init = np.random.uniform(0.0, np.pi, size=tensor_shape + [1])
    phi_init = np.random.uniform(0.0, 2.0 * np.pi, size=tensor_shape + [1])
    zonal_coeffs = tf.convert_to_tensor(
        value=np.random.uniform(-1.0, 1.0, size=[3]), dtype=dtype)

    def rotate_zonal_harmonics_fn(theta, phi):
      return spherical_harmonics.rotate_zonal_harmonics(zonal_coeffs, theta,
                                                        phi)

    self.assert_jacobian_is_correct_fn(rotate_zonal_harmonics_fn,
                                       [theta_init, phi_init])

  @parameterized.parameters(
      ((4,), (3, 1), (3, 1)),
      ((4,), (3, 3, 1), (3, 3, 1)),
  )
  def test_rotate_zonal_harmonics_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        spherical_harmonics.rotate_zonal_harmonics, shape)

  @parameterized.parameters(
      ("must have exactly 1 dimensions in axis -1", (4,), (5, 4), (5, 1)),
      ("must have a rank of 1, but it has rank", (4, 1), (5, 1), (5, 1)),
      ("must have exactly 1 dimensions in axis -1", (4,), (5, 2), (5, 2)),
  )
  def test_rotate_zonal_harmonics_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(spherical_harmonics.rotate_zonal_harmonics,
                                    error_msg, shapes)

  def test_rotate_zonal_harmonics_random(self):
    """Tests the outputs of test_rotate_zonal_harmonics."""
    dtype = tf.float64
    max_band = 2
    zonal_coeffs = tf.constant(
        np.random.uniform(-1.0, 1.0, size=[3]), dtype=dtype)
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    theta = tf.constant(
        np.random.uniform(0.0, np.pi, size=tensor_shape + [1]), dtype=dtype)
    phi = tf.constant(
        np.random.uniform(0.0, 2.0 * np.pi, size=tensor_shape + [1]),
        dtype=dtype)

    rotated_zonal_coeffs = spherical_harmonics.rotate_zonal_harmonics(
        zonal_coeffs, theta, phi)
    zonal_coeffs = spherical_harmonics.tile_zonal_coefficients(zonal_coeffs)
    l, m = spherical_harmonics.generate_l_m_permutations(max_band)
    l = tf.broadcast_to(l, tensor_shape + l.shape.as_list())
    m = tf.broadcast_to(m, tensor_shape + m.shape.as_list())
    theta_zero = tf.constant(0.0, shape=tensor_shape + [1], dtype=dtype)
    phi_zero = tf.constant(0.0, shape=tensor_shape + [1], dtype=dtype)
    gt = zonal_coeffs * spherical_harmonics.evaluate_spherical_harmonics(
        l, m, theta_zero, phi_zero)
    gt = tf.reduce_sum(input_tensor=gt, axis=-1)
    pred = rotated_zonal_coeffs * spherical_harmonics.evaluate_spherical_harmonics(
        l, m, theta + theta_zero, phi + phi_zero)
    pred = tf.reduce_sum(input_tensor=pred, axis=-1)

    self.assertAllClose(gt, pred)

  def test_tile_zonal_coefficients_jacobian_random(self):
    """Tests the jacobian of tile_zonal_coefficients."""
    zonal_coeffs_init = np.random.uniform(size=(1,))

    self.assert_jacobian_is_correct_fn(
        spherical_harmonics.tile_zonal_coefficients, [zonal_coeffs_init])

  @parameterized.parameters(
      ((2,)),)
  def test_tile_zonal_coefficients_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        spherical_harmonics.tile_zonal_coefficients, shape)

  @parameterized.parameters(
      ("must have a rank of 1, but it has rank", (2, 2)),)
  def test_tile_zonal_coefficients_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(spherical_harmonics.tile_zonal_coefficients,
                                    error_msg, shapes)

  def test_tile_zonal_coefficients_preset(self):
    """Tests that tile_zonal_coefficients produces the expected results."""
    self.assertAllEqual(
        spherical_harmonics.tile_zonal_coefficients((0, 1, 2)),
        (0, 1, 1, 1, 2, 2, 2, 2, 2))


if __name__ == "__main__":
  test_case.main()
