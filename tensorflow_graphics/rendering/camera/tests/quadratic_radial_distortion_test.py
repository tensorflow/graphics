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
"""Tests for quadratic_radial_distortion."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.camera import quadratic_radial_distortion
from tensorflow_graphics.util import test_case

RANDOM_TESTS_NUM_IMAGES = 10
RANDOM_TESTS_HEIGHT = 8
RANDOM_TESTS_WIDTH = 8

RADII_SHAPE = (RANDOM_TESTS_NUM_IMAGES, RANDOM_TESTS_HEIGHT, RANDOM_TESTS_WIDTH)
COEFFICIENT_SHAPE = (RANDOM_TESTS_NUM_IMAGES,)


def _get_random_radii():
  return np.random.rand(*RADII_SHAPE).astype('float32')


def _get_zeros_radii():
  return np.zeros(shape=RADII_SHAPE).astype('float32')


def _get_ones_radii():
  return np.ones(shape=RADII_SHAPE).astype('float32')


def _get_random_coefficient():
  return np.random.rand(*COEFFICIENT_SHAPE).astype('float32')


def _get_zeros_coefficient():
  return np.zeros(shape=COEFFICIENT_SHAPE).astype('float32')


def _get_ones_coefficient():
  return np.ones(shape=COEFFICIENT_SHAPE).astype('float32')


def _make_shape_compatible(coefficients):
  return np.expand_dims(np.expand_dims(coefficients, axis=-1), axis=-1)


class QuadraticRadialDistortionTest(test_case.TestCase):

  def test_distortion_factor_random_positive_distortion_coefficient(self):
    """Tests that distortion_factor produces the expected outputs."""
    squared_radii = _get_random_radii() * 2.0
    distortion_coefficient = _get_random_coefficient() * 2.0

    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)

    distortion_coefficient = _make_shape_compatible(distortion_coefficient)
    with self.subTest(name='distortion'):
      self.assertAllClose(1.0 + distortion_coefficient * squared_radii,
                          distortion)

    # No overflow when distortion_coefficient >= 0.0.
    with self.subTest(name='mask'):
      self.assertAllInSet(mask, (False,))

  def test_distortion_factor_preset_zero_distortion_coefficient(self):
    """Tests distortion_factor at zero distortion coefficient."""
    squared_radii = _get_random_radii() * 2.0

    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, 0.0)

    with self.subTest(name='distortion'):
      self.assertAllClose(tf.ones_like(squared_radii), distortion)

    # No overflow when distortion_coefficient = 0.0.
    with self.subTest(name='mask'):
      self.assertAllInSet(mask, (False,))

  def test_distortion_factor_random_negative_distortion_coefficient(self):
    """Tests that distortion_factor produces the expected outputs."""
    squared_radii = _get_random_radii() * 2.0
    distortion_coefficient = _get_random_coefficient() * -0.2

    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)
    distortion_coefficient = _make_shape_compatible(distortion_coefficient)
    max_squared_radii = -1.0 / 3.0 / distortion_coefficient
    expected_overflow_mask = squared_radii > max_squared_radii
    valid_mask = np.logical_not(expected_overflow_mask)
    # We assert correctness of the mask, and of all the pixels that are not in
    # overflow.
    actual_distortion_when_valid = self.evaluate(distortion)[valid_mask]
    expected_distortion_when_valid = (
        1.0 + distortion_coefficient * squared_radii)[valid_mask]

    with self.subTest(name='distortion'):
      self.assertAllClose(expected_distortion_when_valid,
                          actual_distortion_when_valid)

    with self.subTest(name='mask'):
      self.assertAllEqual(expected_overflow_mask, mask)

  def test_distortion_factor_preset_zero_radius(self):
    """Tests distortion_factor at the corner case of zero radius."""
    squared_radii = _get_zeros_radii()
    distortion_coefficient = _get_random_coefficient() - 0.5

    distortion, mask = quadratic_radial_distortion.distortion_factor(
        squared_radii, distortion_coefficient)

    with self.subTest(name='distortion'):
      self.assertAllClose(np.ones_like(squared_radii), distortion)

    with self.subTest(name='mask'):
      self.assertAllInSet(mask, (False,))

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_negative_radius_exception_raised(self, distortion_function):
    """Tests that an exception is raised when the squared radius is negative."""
    squared_radii = _get_zeros_radii() - 0.5
    distortion_coefficient = _get_random_coefficient() - 0.5

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(distortion_function(squared_radii, distortion_coefficient))

  @parameterized.parameters((2, 2e-3), (3, 1e-8))
  def test_undistortion_factor_random_positive_distortion_coefficient(
      self, num_iterations, tolerance):
    """Tests that undistortion_factor produces the expected outputs."""
    distorted_squared_radii = _get_random_radii() * 2.0
    distortion_coefficient = _get_random_coefficient() * 0.2

    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        distorted_squared_radii, distortion_coefficient, num_iterations)
    distortion_coefficient = _make_shape_compatible(distortion_coefficient)
    undistorted_squared_radii = tf.square(
        undistortion) * distorted_squared_radii
    # We distort again the undistorted radii and compare to the original
    # distorted_squared_radii.
    redistorted_squared_radii = tf.square(
        1.0 + distortion_coefficient *
        undistorted_squared_radii) * undistorted_squared_radii

    with self.subTest(name='distortion'):
      self.assertAllClose(
          distorted_squared_radii, redistorted_squared_radii, atol=tolerance)

    # Positive distortion_coefficients never overflow.
    with self.subTest(name='mask'):
      self.assertAllInSet(mask, (False,))

  @parameterized.parameters((2, 2e-2), (3, 6e-3), (4, 6e-4))
  def test_undistortion_factor_random_negative_distortion_coefficient(
      self, num_iterations, tolerance):
    """Tests that undistortion_factor produces the expected outputs."""
    distorted_squared_radii = _get_random_radii() * 2.0
    distortion_coefficient = _get_random_coefficient() * -0.2

    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        distorted_squared_radii, distortion_coefficient, num_iterations)
    distortion_coefficient = _make_shape_compatible(distortion_coefficient)
    undistorted_squared_radii = tf.square(
        undistortion) * distorted_squared_radii
    # See explanation in the implementation comments for this formula.
    expected_overflow_mask = (
        distorted_squared_radii * distortion_coefficient + 4.0 / 27.0 < 0)
    redistorted_squared_radii = tf.square(
        1.0 + distortion_coefficient *
        undistorted_squared_radii) * undistorted_squared_radii
    valid_mask = np.logical_not(expected_overflow_mask)
    redistorted_squared_radii_when_valid = self.evaluate(
        redistorted_squared_radii)[valid_mask]
    distorted_squared_radii_when_valid = distorted_squared_radii[valid_mask]

    with self.subTest(name='distortion'):
      self.assertAllClose(
          distorted_squared_radii_when_valid,
          redistorted_squared_radii_when_valid,
          rtol=tolerance,
          atol=tolerance)

    # We assert correctness of the mask, and of all the pixels that are not in
    # overflow, distorting again the undistorted radii and comparing to the
    # original distorted_squared_radii.
    with self.subTest(name='mask'):
      self.assertAllEqual(expected_overflow_mask, mask)

  def test_undistortion_factor_zero_distortion_coefficient(self):
    """Tests undistortion_factor at zero distortion coefficient."""
    squared_radii = _get_random_radii() * 2.0

    undistortion, mask = quadratic_radial_distortion.undistortion_factor(
        squared_radii, 0.0)

    with self.subTest(name='distortion'):
      self.assertAllClose(tf.ones_like(squared_radii), undistortion)

    # No overflow when distortion_coefficient = 0.0.
    with self.subTest(name='mask'):
      self.assertAllEqual(np.zeros_like(squared_radii), mask)

  @parameterized.parameters(
      ('must have a rank greater than 1', (2,), (2, 1)),
      ('Not all batch dimensions are broadcast-compatible', (2, 2, 2), (3,)),
      ('Not all batch dimensions are broadcast-compatible', (2, 2, 2), (3, 3)),
  )
  def test_distortion_factor_shape_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        func=quadratic_radial_distortion.distortion_factor,
        error_msg=error_msg,
        shapes=shapes)

  @parameterized.parameters(
      ((2, 2), ()),
      ((1, 2, 2), (2,)),
      ((2, 2, 2), (2,)),
      ((2, 2), (2, 2)),
      ((2, 2, 2), (1, 2)),
      ((2, 3, 4), (1,)),
      ((2, 3, 4), (1, 1)),
      ((2, 3, 4), (2,)),
  )
  def test_distortion_factor_shape_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_not_raised(
        func=quadratic_radial_distortion.distortion_factor, shapes=shapes)

  @parameterized.parameters(
      ('must have a rank greater than 1', (2,), (2, 1)),
      ('Not all batch dimensions are broadcast-compatible', (2, 2, 2), (3,)),
      ('Not all batch dimensions are broadcast-compatible', (2, 2, 2), (3, 3)),
  )
  def test_undistortion_factor_shape_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        func=quadratic_radial_distortion.undistortion_factor,
        error_msg=error_msg,
        shapes=shapes)

  @parameterized.parameters(
      ((2, 2), ()),
      ((1, 2, 2), (2,)),
      ((2, 2, 2), (2,)),
      ((2, 2), (2, 2)),
      ((2, 2, 2), (1, 2)),
      ((2, 3, 4), (1,)),
      ((2, 3, 4), (1, 1)),
      ((2, 3, 4), (2,)),
  )
  def test_undistortion_factor_shape_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_not_raised(
        func=quadratic_radial_distortion.undistortion_factor, shapes=shapes)

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_radial_jacobian(self, distortion_function):
    """Test the Jacobians with respect to squared radii."""
    squared_radii = _get_random_radii().astype(np.float64) * 0.5
    distortion_coefficients = _get_random_coefficient().astype(np.float64) * 0.5
    distortion_coefficients -= 0.25

    def distortion_fn(squared_radii):
      distortion, _ = distortion_function(squared_radii,
                                          distortion_coefficients)
      return distortion

    self.assert_jacobian_is_correct_fn(
        distortion_fn, [squared_radii], delta=1e-7, atol=1e-3)

  @parameterized.parameters(quadratic_radial_distortion.distortion_factor,
                            quadratic_radial_distortion.undistortion_factor)
  def test_both_distortion_coefficient_jacobian(self, distortion_function):
    """Test the Jacobians with respect to distortion coefficients."""
    squared_radii = _get_random_radii().astype(np.float64) * 0.5
    distortion_coefficients = _get_random_coefficient().astype(np.float64) * 0.5
    distortion_coefficients -= 0.25

    def distortion_fn(distortion_coefficients):
      distortion, _ = distortion_function(squared_radii,
                                          distortion_coefficients)
      return distortion

    self.assert_jacobian_is_correct_fn(
        distortion_fn, [distortion_coefficients], delta=1e-7, atol=1e-3)


if __name__ == '__main__':
  test_case.main()
