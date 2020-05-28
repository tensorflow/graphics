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
"""Tests for math_helpers."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import sys

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.math import math_helpers
from tensorflow_graphics.util import asserts
from tensorflow_graphics.util import test_case


class MathTest(test_case.TestCase):

  @parameterized.parameters(
      (((0.0, 0.0, 0.0),), ((0.0, np.pi / 2.0, 0.0),)),
      (((2.0, 0.0, 0.0),), ((2.0, np.pi / 2.0, 0.0),)),
      (((0.0, 1.0, 0.0),), ((1.0, np.pi / 2.0, np.pi / 2.0),)),
      (((0.0, 0.0, 1.0),), ((1.0, 0.0, 0.0),)),
      (((-1.0, 0.0, 0.0),), ((1.0, np.pi / 2.0, np.pi),)),
      (((0.0, -1.0, 0.0),), ((1.0, np.pi / 2.0, -np.pi / 2.0),)),
      (((0.0, 0.0, -1.0),), ((1.0, np.pi, 0.0),)),
  )
  def test_cartesian_to_spherical_coordinates_preset(self, test_inputs,
                                                     test_outputs):
    """Tests that cartesian_to_spherical_coordinates behaves as expected."""
    self.assert_output_is_correct(
        math_helpers.cartesian_to_spherical_coordinates, test_inputs,
        test_outputs)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_cartesian_to_spherical_coordinates_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        math_helpers.cartesian_to_spherical_coordinates, shape)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (1,)),)
  def test_cartesian_to_spherical_coordinates_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(
        math_helpers.cartesian_to_spherical_coordinates, error_msg, shape)

  def test_cartesian_to_spherical_coordinates_jacobian_random(self):
    """Test the Jacobian of the spherical_to_cartesian_coordinates function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_init = np.random.uniform(-10.0, 10.0, size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(
        math_helpers.cartesian_to_spherical_coordinates, [point_init])

  @parameterized.parameters(
      (((1.0, 1.0, 1.0),),),
      (((1.0, 0.0, 0.0),),),
      (((0.0, 1.0, 0.0),),),
  )
  def test_cartesian_to_spherical_coordinates_jacobian_preset(self, cartesian):
    """Test the Jacobian of the spherical_to_cartesian_coordinates function."""
    point_init = np.asarray(cartesian)

    self.assert_jacobian_is_correct_fn(
        math_helpers.cartesian_to_spherical_coordinates, [point_init])

  @parameterized.parameters(
      (((1.0, 1.0, 0.0),), ((np.sqrt(2.0), np.pi / 2.0, np.pi / 4.0),)),
      (((1.0, 0.0, 0.0),), ((1.0, np.pi / 2.0, 0.0),)),
      (((0.0, 1.0, 0.0),), ((1.0, np.pi / 2.0, np.pi / 2.0),)),
      (((0.0, 0.0, 1.0),), ((1.0, 0.0, 0.0),)),
      (((0.0, 0.0, 0.0),), ((0.0, np.pi / 2.0, 0.0),)),
  )
  def test_cartesian_to_spherical_coordinates_values_preset(
      self, test_inputs, test_outputs):
    """Test the Jacobian of the spherical_to_cartesian_coordinates function."""
    self.assert_output_is_correct(
        math_helpers.cartesian_to_spherical_coordinates, test_inputs,
        test_outputs)

  @parameterized.parameters(
      (((0, 1, 5, 6, 15.0),), ((1, 1, 15, 48, 2027025.0),)),)
  def test_double_factorial_preset(self, test_inputs, test_outputs):
    """Tests that double_factorial generates expected results."""
    self.assert_output_is_correct(math_helpers.double_factorial, test_inputs,
                                  test_outputs)

  @parameterized.parameters(
      (((0, 1, 2, 3, 4.0),), ((1, 1, 2, 6, 24.0),)),)
  def test_factorial_preset(self, test_inputs, test_outputs):
    """Tests that double_factorial generates expected results."""
    self.assert_output_is_correct(math_helpers.factorial, test_inputs,
                                  test_outputs)

  @parameterized.parameters(
      (((2.0, np.pi / 2.0, 0.0),), ((2.0, 0.0, 0.0),)),
      (((2.0, -3.0 * np.pi / 2.0, 0.0),), ((2.0, 0.0, 0.0),)),
      (((1.0, np.pi / 2.0, np.pi / 2.0),), ((0.0, 1.0, 0.0),)),
      (((1.0, 0.0, 0.0),), ((0.0, 0.0, 1.0),)),
  )
  def test_spherical_to_cartesian_coordinates_preset(self, test_inputs,
                                                     test_outputs):
    """Tests that spherical_to_cartesian_coordinates behaves as expected."""
    self.assert_output_is_correct(
        math_helpers.spherical_to_cartesian_coordinates, test_inputs,
        test_outputs)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_spherical_to_cartesian_coordinates_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        math_helpers.spherical_to_cartesian_coordinates, shape)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (1,)),)
  def test_spherical_to_cartesian_coordinates_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(
        math_helpers.spherical_to_cartesian_coordinates, error_msg, shape)

  def test_spherical_to_cartesian_coordinates_jacobian_random(self):
    """Test the Jacobian of the spherical_to_cartesian_coordinates function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    r_init = np.random.uniform(0.0, 10.0, size=tensor_shape + [1])
    theta_init = np.random.uniform(
        -np.pi / 2.0, np.pi / 2.0, size=tensor_shape + [1])
    phi_init = np.random.uniform(-np.pi, np.pi, size=tensor_shape + [1])
    data_init = np.stack((r_init, theta_init, phi_init), axis=-1)

    self.assert_jacobian_is_correct_fn(
        math_helpers.spherical_to_cartesian_coordinates, [data_init])

  @parameterized.parameters(
      (((0.0, 0.0),), ((1.0, 0.0, 0.0),)),
      (((1.0, 0.0),), ((1.0, np.pi, 0.0),)),
      (((0.0, 1.0),), ((1.0, 0.0, 2.0 * np.pi),)),
      (((1.0, 1.0),), ((1.0, np.pi, 2.0 * np.pi),)),
  )
  def test_square_to_spherical_coordinates_preset(self, test_inputs,
                                                  test_outputs):
    """Tests that square_to_spherical_coordinates generates expected results."""
    self.assert_output_is_correct(math_helpers.square_to_spherical_coordinates,
                                  test_inputs, test_outputs)

  def test_square_to_spherical_coordinates_jacobian_random(self):
    """Tests the Jacobian of square_to_spherical_coordinates."""
    epsilon = 1e-3
    point_2d_init = np.random.uniform(epsilon, 1.0 - epsilon, size=(10, 2))

    self.assert_jacobian_is_correct_fn(
        math_helpers.square_to_spherical_coordinates, [point_2d_init],
        atol=1e-3)

  def test_square_to_spherical_coordinates_range_exception_raised(self):
    """Tests that the exceptions are raised correctly."""
    point_2d_below = np.random.uniform(-1.0, -sys.float_info.epsilon, size=(2,))
    point_2d_above = np.random.uniform(
        1.0 + asserts.select_eps_for_addition(tf.float32), 2.0, size=(2,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          math_helpers.square_to_spherical_coordinates(point_2d_below))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          math_helpers.square_to_spherical_coordinates(point_2d_above))

  @parameterized.parameters(
      ((2,),),
      ((None, 2),),
  )
  def test_square_to_spherical_coordinates_shape_exception_not_raised(
      self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        math_helpers.square_to_spherical_coordinates, shape)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (1,)),
      ("must have exactly 2 dimensions in axis -1", (3,)),
  )
  def test_square_to_spherical_coordinates_shape_exception_raised(
      self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(
        math_helpers.square_to_spherical_coordinates, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
