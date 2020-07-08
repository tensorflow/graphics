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
"""Tests for tangential_distortion."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_graphics.rendering.camera import tangential_distortion
from tensorflow_graphics.util import test_case

RANDOM_TESTS_NUM_IMAGES = 10
RANDOM_TESTS_HEIGHT = 8
RANDOM_TESTS_WIDTH = 8

RADII_SHAPE = (RANDOM_TESTS_NUM_IMAGES, RANDOM_TESTS_HEIGHT, RANDOM_TESTS_WIDTH)
COORDINATES_SHAPE = (RANDOM_TESTS_NUM_IMAGES,
                     RANDOM_TESTS_HEIGHT,
                     RANDOM_TESTS_WIDTH)
COEFFICIENT_SHAPE = (RANDOM_TESTS_NUM_IMAGES,)


def _get_random_coordinates():
  return np.ones(shape=COORDINATES_SHAPE).astype('float32')


def _get_zeros_coordinates():
  return np.zeros(shape=COORDINATES_SHAPE).astype('float32')


def _get_ones_coordinates():
  return np.ones(shape=COORDINATES_SHAPE).astype('float32')


def _get_random_coefficient():
  return np.random.rand(*COEFFICIENT_SHAPE).astype('float32')


def _get_zeros_coefficient():
  return np.zeros(shape=COEFFICIENT_SHAPE).astype('float32')


def _get_ones_coefficient():
  return np.ones(shape=COEFFICIENT_SHAPE).astype('float32')


def _make_shape_compatible(coefficients):
  return np.expand_dims(np.expand_dims(coefficients, axis=-1), axis=-1)


def _get_squared_radii(projective_x, projective_y):
  return projective_x ** 2.0 + projective_y ** 2.0


def _get_double_projective_coordinates_squared(coordinates):
  return 2.0 * coordinates ** 2.0


def _get_projective_x_times_projective_y(projective_x, projective_y):
  return projective_x * projective_y


class TangentialDistortionTest(test_case.TestCase):

  def test_distortion_terms_random_positive_distortion_coefficients(self):
    """Tests that distortion_terms produces the expected outputs."""
    projective_x = _get_random_coordinates() * 2.0
    projective_y = _get_random_coordinates() * 2.0
    distortion_coefficient_1 = _get_random_coefficient() * 2.0
    distortion_coefficient_2 = _get_random_coefficient() * 2.0

    (distortion_x,
     distortion_y,
     mask_x,
     mask_y) = tangential_distortion.distortion_terms(projective_x,
                                                      projective_y,
                                                      distortion_coefficient_1,
                                                      distortion_coefficient_2)

    distortion_coefficient_1 = _make_shape_compatible(distortion_coefficient_1)
    distortion_coefficient_2 = _make_shape_compatible(distortion_coefficient_2)
    squared_radius = _get_squared_radii(projective_x, projective_y)
    projective_x_times_projective_y = _get_projective_x_times_projective_y(
      projective_x, projective_y)
    double_projective_x_squared = _get_double_projective_coordinates_squared(
      projective_x)
    double_projective_y_squared = _get_double_projective_coordinates_squared(
      projective_y)
    with self.subTest(name='distortion'):
      self.assertAllClose(2.0 * distortion_coefficient_1
                          * projective_x_times_projective_y
                          + distortion_coefficient_2
                          * (squared_radius + double_projective_x_squared),
                          distortion_x)
      self.assertAllClose(2.0 * distortion_coefficient_2
                          * projective_x_times_projective_y
                          + distortion_coefficient_1
                          * (squared_radius + double_projective_y_squared),
                          distortion_y)

    # No overflow when distortion coefficients >= 0.0
    with self.subTest(name='mask'):
      self.assertAllInSet(mask_x, (False,))
      self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_preset_zero_distortion_coefficient_1(self):
      """Tests distortion_terms with first distortion coefficient at zero."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_2 = _get_random_coefficient() * 2.0

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x, projective_y, 0.0, distortion_coefficient_2)

      distortion_coefficient_2 = _make_shape_compatible(
        distortion_coefficient_2)
      squared_radius = _get_squared_radii(projective_x, projective_y)
      projective_x_times_projective_y = _get_projective_x_times_projective_y(
        projective_x, projective_y)
      double_projective_x_squared = _get_double_projective_coordinates_squared(
        projective_x)
      with self.subTest(name='distortion'):
        self.assertAllClose(distortion_coefficient_2
                            * (squared_radius + double_projective_x_squared),
                            distortion_x)
        self.assertAllClose(2.0 * distortion_coefficient_2
                            * projective_x_times_projective_y,
                            distortion_y)

      with self.subTest(name="mask"):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_preset_zero_distortion_coefficient_2(self):
      """Tests distortion_terms with second distortion coefficient at zero."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() * 2.0

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x, projective_y, distortion_coefficient_1, 0.0)

      distortion_coefficient_1 = _make_shape_compatible(
        distortion_coefficient_1)
      squared_radius = _get_squared_radii(projective_x, projective_y)
      projective_x_times_projective_y = _get_projective_x_times_projective_y(
        projective_x, projective_y)
      double_projective_y_squared = _get_double_projective_coordinates_squared(
        projective_y)
      with self.subTest(name='distortion'):
        self.assertAllClose(2.0 * distortion_coefficient_1
                            * projective_x_times_projective_y,
                            distortion_x)
        self.assertAllClose(distortion_coefficient_1
                            * (squared_radius + double_projective_y_squared),
                            distortion_y)

      with self.subTest(name="mask"):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_preset_zero_distortion_coefficients(self):
      """Tests distortion_terms at zero disortion coefficients."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(projective_x,
                                                        projective_y,
                                                        0.0,
                                                        0.0)

      with self.subTest(name='distortion'):
        self.assertAllClose(tf.zeros_like(projective_x), distortion_x)
        self.assertAllClose(tf.zeros_like(projective_y), distortion_y)

      # No overflow when distortion coefficients = 0.0
      with self.subTest(name='mask'):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_random_negative_distortion_coefficient_1(self):
      """Tests that distortion_terms produces the expected outputs."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() * -0.2
      distortion_coefficient_2 = _get_random_coefficient() * 2.0

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      distortion_coefficient_1 = _make_shape_compatible(
        distortion_coefficient_1)
      distortion_coefficient_2 = _make_shape_compatible(
        distortion_coefficient_2)
      squared_radius = _get_squared_radii(projective_x, projective_y)
      projective_x_times_projective_y = _get_projective_x_times_projective_y(
        projective_x, projective_y)
      double_projective_x_squared = _get_double_projective_coordinates_squared(
        projective_x)
      double_projective_y_squared = _get_double_projective_coordinates_squared(
        projective_y)
      max_projective_x = ((-1.0 - 2.0 * distortion_coefficient_1 * projective_y)
                          / (6.0 * distortion_coefficient_2))
      max_projective_y = ((-1.0 - 2.0 * distortion_coefficient_2 * projective_x)
                          / (6.0 * distortion_coefficient_1))
      expected_overflow_mask_x = projective_x > max_projective_x
      expected_overflow_mask_y = projective_y > max_projective_y
      valid_mask_x = np.logical_not(expected_overflow_mask_x)
      valid_mask_y = np.logical_not(expected_overflow_mask_y)
      # We assert correctness of the masks, and of all the pixels that are not
      # in overflow.
      actual_x_distortion_when_valid = self.evaluate(distortion_x)[valid_mask_x]
      actual_y_distortion_when_valid = self.evaluate(distortion_y)[valid_mask_y]
      expected_x_distortion_when_valid = (
        2.0 * distortion_coefficient_1 * projective_x_times_projective_y
        + distortion_coefficient_2
        * (squared_radius + double_projective_x_squared))[valid_mask_x]
      expected_y_distortion_when_valid = (
        2.0 * distortion_coefficient_2 * projective_x_times_projective_y
        + distortion_coefficient_1
        * (squared_radius + double_projective_y_squared))[valid_mask_y]

      with self.subTest(name='distortion'):
        self.assertAllClose(expected_x_distortion_when_valid,
                            actual_x_distortion_when_valid)
        self.assertAllClose(expected_y_distortion_when_valid,
                            actual_y_distortion_when_valid)

      with self.subTest(name='mask'):
        self.assertAllEqual(expected_overflow_mask_x, mask_x)
        self.assertAllEqual(expected_overflow_mask_y, mask_y)

    def test_distortion_terms_random_negative_distortion_coefficient_2(self):
      """Tests that distortion_terms produces the expected outputs."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() * 2.0
      distortion_coefficient_2 = _get_random_coefficient() * -0.2

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      distortion_coefficient_1 = _make_shape_compatible(
        distortion_coefficient_1)
      distortion_coefficient_2 = _make_shape_compatible(
        distortion_coefficient_2)
      squared_radius = _get_squared_radii(projective_x, projective_y)
      projective_x_times_projective_y = _get_projective_x_times_projective_y(
        projective_x, projective_y)
      double_projective_x_squared = _get_double_projective_coordinates_squared(
        projective_x)
      double_projective_y_squared = _get_double_projective_coordinates_squared(
        projective_y)
      max_projective_x = ((-1.0 - 2.0 * distortion_coefficient_1 * projective_y)
                          / (6.0 * distortion_coefficient_2))
      max_projective_y = ((-1.0 - 2.0 * distortion_coefficient_2 * projective_x)
                          / (6.0 * distortion_coefficient_1))
      expected_overflow_mask_x = projective_x > max_projective_x
      expected_overflow_mask_y = projective_y > max_projective_y
      valid_mask_x = np.logical_not(expected_overflow_mask_x)
      valid_mask_y = np.logical_not(expected_overflow_mask_y)
      # We assert correctness of the masks, and of all the pixels that are not
      # in overflow.
      actual_x_distortion_when_valid = self.evaluate(distortion_x)[valid_mask_x]
      actual_y_distortion_when_valid = self.evaluate(distortion_y)[valid_mask_y]
      expected_x_distortion_when_valid = (
        2.0 * distortion_coefficient_1 * projective_x_times_projective_y
        + distortion_coefficient_2
        * (squared_radius + double_projective_x_squared))[valid_mask_x]
      expected_y_distortion_when_valid = (
        2.0 * distortion_coefficient_2 * projective_x_times_projective_y
        + distortion_coefficient_1
        * (squared_radius + double_projective_y_squared))[valid_mask_y]

      with self.subTest(name='distortion'):
        self.assertAllClose(expected_x_distortion_when_valid,
                            actual_x_distortion_when_valid)
        self.assertAllClose(expected_y_distortion_when_valid,
                            actual_y_distortion_when_valid)

      with self.subTest(name='mask'):
        self.assertAllEqual(expected_overflow_mask_x, mask_x)
        self.assertAllEqual(expected_overflow_mask_y, mask_y)

    def test_distortion_terms_random_negative_distortion_coefficients(self):
      """Tests that distortion_terms produces the expected outputs."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() * -0.2
      distortion_coefficient_2 = _get_random_coefficient() * -0.2

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      distortion_coefficient_1 = _make_shape_compatible(
        distortion_coefficient_1)
      distortion_coefficient_2 = _make_shape_compatible(
        distortion_coefficient_2)
      squared_radius = _get_squared_radii(projective_x, projective_y)
      projective_x_times_projective_y = _get_projective_x_times_projective_y(
        projective_x, projective_y)
      double_projective_x_squared = _get_double_projective_coordinates_squared(
        projective_x)
      double_projective_y_squared = _get_double_projective_coordinates_squared(
        projective_y)
      max_projective_x = ((-1.0 - 2.0 * distortion_coefficient_1 * projective_y)
                          / (6.0 * distortion_coefficient_2))
      max_projective_y = ((-1.0 - 2.0 * distortion_coefficient_2 * projective_x)
                          / (6.0 * distortion_coefficient_1))
      expected_overflow_mask_x = projective_x > max_projective_x
      expected_overflow_mask_y = projective_y > max_projective_y
      valid_mask_x = np.logical_not(expected_overflow_mask_x)
      valid_mask_y = np.logical_not(expected_overflow_mask_y)
      # We assert correctness of the masks, and of all the pixels that are not
      # in overflow.
      actual_x_distortion_when_valid = self.evaluate(distortion_x)[valid_mask_x]
      actual_y_distortion_when_valid = self.evaluate(distortion_y)[valid_mask_y]
      expected_x_distortion_when_valid = (
        2.0 * distortion_coefficient_1 * projective_x_times_projective_y
        + distortion_coefficient_2
        * (squared_radius + double_projective_x_squared))[valid_mask_x]
      expected_y_distortion_when_valid = (
        2.0 * distortion_coefficient_2 * projective_x_times_projective_y
        + distortion_coefficient_1
        * (squared_radius + double_projective_y_squared))[valid_mask_y]

      with self.subTest(name='distortion'):
        self.assertAllClose(expected_x_distortion_when_valid,
                            actual_x_distortion_when_valid)
        self.assertAllClose(expected_y_distortion_when_valid,
                            actual_y_distortion_when_valid)

      with self.subTest(name='mask'):
        self.assertAllEqual(expected_overflow_mask_x, mask_x)
        self.assertAllEqual(expected_overflow_mask_y, mask_y)

    def test_distortion_terms_preset_zero_x_coordinates(self):
      """Tests distortion_terms at the corner case of x coordinates at zero."""
      projective_x = _get_zeros_coordinates()
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() - 0.5
      distortion_coefficient_2 = _get_random_coefficient() - 0.5

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      with self.subTest(name='distortion'):
        self.assertAllClose(distortion_coefficient_2 * projective_y ** 2.0,
                            distortion_x)
        self.assertAllClose(distortion_coefficient_1
                            * 3.0 * projective_y ** 2.0,
                            distortion_y)

      with self.subTest(name='mask'):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_preset_zero_y_coordinates(self):
      """Tests distortion_terms at the corner case of x coordinates at zero."""
      projective_x = _get_random_coordinates() * 2.0
      projective_y = _get_zeros_coordinates()
      distortion_coefficient_1 = _get_random_coefficient() - 0.5
      distortion_coefficient_2 = _get_random_coefficient() - 0.5

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      with self.subTest(name='distortion'):
        self.assertAllClose(distortion_coefficient_2
                            * 3.0 * projective_x ** 2.0,
                            distortion_x)
        self.assertAllClose(distortion_coefficient_1 * projective_x ** 2.0,
                            distortion_y)

      with self.subTest(name='mask'):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    def test_distortion_terms_preset_zero_coordinates(self):
      """Tests distortion_terms at corner case of zero-valued coordinates."""
      projective_x = _get_zeros_coordinates()
      projective_y = _get_zeros_coordinates()
      distortion_coefficient_1 = _get_random_coefficient() - 0.5
      distortion_coefficient_2 = _get_random_coefficient() - 0.5

      (distortion_x,
       distortion_y,
       mask_x,
       mask_y) = tangential_distortion.distortion_terms(
         projective_x,
         projective_y,
         distortion_coefficient_1,
         distortion_coefficient_2)

      with self.subTest(name='distortion'):
        self.assertAllClose(tf.zeros_like(projective_x), distortion_x)
        self.assertAllClose(tf.zeros_like(projective_y), distortion_y)

      with self.subTest(name='mask'):
        self.assertAllInSet(mask_x, (False,))
        self.assertAllInSet(mask_y, (False,))

    # TODO: add undistortion_terms
    # @parameterized.parameters(tangential_distortion.distortion_terms)
    def test_both_negative_x_exception_raised(self):
      """Tests that an exception is raised for negative projective_x."""
      projective_x = _get_zeros_coordinates() - 0.5
      projective_y = _get_random_coordinates() * 2.0
      distortion_coefficient_1 = _get_random_coefficient() - 0.5
      distortion_coefficient_2 = _get_random_coefficient() - 0.5

      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(tangential_distortion.distortion_terms(
          projective_x,
          projective_y,
          distortion_coefficient_1,
          distortion_coefficient_2))

    @parameterized.parameters(
      ('must have a rank greater than 1', (2,), (2,), (2, 1), (2, 1)),
      ('Not all batch dimensions are broadcast-compatible',
       (2, 2, 2), (2, 2, 2), (3,), (3,)),
      ('Not all batch dimensions are broadcast-compatible',
       (2, 2, 2), (2, 2, 2), (3, 3), (3, 3)),
    )
    def test_distortion_terms_shape_exception_raised(self, error_msg, *shapes):
      """Tests that the shape exceptions are raised."""
      self.assert_exception_is_raised(
        func=tangential_distortion.distortion_terms,
        error_msg=error_msg,
        shapes=shapes)

    @parameterized.parameters(
      ((2, 2), (2, 2), (), ()),
      ((1, 2, 2), (1, 2, 2), (2,), (2,)),
      ((2, 2, 2), (2, 2, 2), (2,), (2,)),
      ((2, 2), (2, 2), (2, 2), (2, 2)),
      ((2, 2, 2), (2, 2, 2), (1, 2), (1, 2)),
      ((2, 3, 4), (2, 3, 4), (1,), (1,)),
      ((2, 3, 4), (2, 3, 4), (1, 1), (1, 1)),
      ((2, 3, 4), (2, 3, 4), (2,), (2,)),
    )
    def test_distortion_terms_shape_exception_not_raised(self, *shapes):
      """Tests that the shape exceptions are not raised."""
      self.assert_exception_is_not_raised(
        func=tangential_distortion.distortion_terms, shapes=shapes)

    # TODO: add undistortion_terms
    def test_both_projective_x_jacobian(self):
      """Test the Jacobians with respect to projective_x."""
      projective_x = _get_random_coordinates().astype(np.float64) * 0.5
      projective_y = _get_random_coordinates().astype(np.float64) * 0.5
      distortion_coefficient_1 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_2 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_1 -= 0.25
      distortion_coefficient_2 -= 0.25

      def distortion_fn(distortion):
        (distortion_x,
         distortion_y,
         _, _) = tangential_distortion.distortion_terms(
           projective_x,
           projective_y,
           distortion_coefficient_1,
           distortion_coefficient_2)

        return distortion_x, distortion_y

      self.assert_jacobian_is_correct_fn(
        distortion_fn, [projective_x], delta=1e-7, atol=1e-3)

    # TODO: add undistortion_terms
    def test_both_projective_y_jacobian(self):
      """Test the Jacobians with respect to projective_y."""
      projective_x = _get_random_coordinates().astype(np.float64) * 0.5
      projective_y = _get_random_coordinates().astype(np.float64) * 0.5
      distortion_coefficient_1 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_2 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_1 -= 0.25
      distortion_coefficient_2 -= 0.25

      def distortion_fn(distortion):
        (distortion_x,
         distortion_y,
         _, _) = tangential_distortion.distortion_terms(
           projective_x,
           projective_y,
           distortion_coefficient_1,
           distortion_coefficient_2)

        return distortion_x, distortion_y

      self.assert_jacobian_is_correct_fn(
        distortion_fn, [projective_y], delta=1e-7, atol=1e-3)

    # TODO: add undistortion_terms
    def test_both_distortion_coefficient_1_jacobian(self):
      """Test the Jacobians with respect to projective_y."""
      projective_x = _get_random_coordinates().astype(np.float64) * 0.5
      projective_y = _get_random_coordinates().astype(np.float64) * 0.5
      distortion_coefficient_1 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_2 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_1 -= 0.25
      distortion_coefficient_2 -= 0.25

      def distortion_fn(distortion):
        (distortion_x,
         distortion_y,
         _, _) = tangential_distortion.distortion_terms(
           projective_x,
           projective_y,
           distortion_coefficient_1,
           distortion_coefficient_2)

        return distortion_x, distortion_y

      self.assert_jacobian_is_correct_fn(
        distortion_fn, [distortion_coefficient_1], delta=1e-7, atol=1e-3)

    # TODO: add undistortion_terms
    def test_both_distortion_coefficient_2_jacobian(self):
      """Test the Jacobians with respect to projective_y."""
      projective_x = _get_random_coordinates().astype(np.float64) * 0.5
      projective_y = _get_random_coordinates().astype(np.float64) * 0.5
      distortion_coefficient_1 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_2 = _get_random_coefficient().astype(float64) * 0.5
      distortion_coefficient_1 -= 0.25
      distortion_coefficient_2 -= 0.25

      def distortion_fn(distortion):
        (distortion_x,
         distortion_y,
         _, _) = tangential_distortion.distortion_terms(
           projective_x,
           projective_y,
           distortion_coefficient_1,
           distortion_coefficient_2)

        return distortion_x, distortion_y

      self.assert_jacobian_is_correct_fn(
        distortion_fn, [distortion_coefficient_2], delta=1e-7, atol=1e-3)


if __name__ == '__main__':
  test_case.main()
