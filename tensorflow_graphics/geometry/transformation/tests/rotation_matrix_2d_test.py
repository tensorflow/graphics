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
"""Tests for 2d rotation matrix."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.geometry.transformation import rotation_matrix_2d
from tensorflow_graphics.geometry.transformation.tests import test_data as td
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class RotationMatrix2dTest(test_case.TestCase):

  @parameterized.parameters(
      ((1,)),
      ((None, 1),),
  )
  def test_from_euler_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_2d.from_euler, shapes)

  @parameterized.parameters(
      ("must have exactly 1 dimensions in axis -1", (None,)),)
  def test_from_euler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_2d.from_euler, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_jacobian_preset(self):
    """Test the Jacobian of the from_euler function."""
    x_init = test_helpers.generate_preset_test_euler_angles(dimensions=1)

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.from_euler, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_jacobian_random(self):
    """Test the Jacobian of the from_euler function."""
    x_init = test_helpers.generate_random_test_euler_angles(dimensions=1)

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.from_euler, [x_init])

  def test_from_euler_normalized_preset(self):
    """Tests that an angle maps to correct matrix."""
    euler_angles = test_helpers.generate_preset_test_euler_angles(dimensions=1)

    matrix = rotation_matrix_2d.from_euler(euler_angles)

    self.assertAllEqual(
        rotation_matrix_2d.is_valid(matrix),
        np.ones(euler_angles.shape[0:-1] + (1,), dtype=bool))

  @parameterized.parameters(
      ((td.ANGLE_0,), (td.MAT_2D_ID,)),
      ((td.ANGLE_45,), (td.MAT_2D_45,)),
      ((td.ANGLE_90,), (td.MAT_2D_90,)),
      ((td.ANGLE_180,), (td.MAT_2D_180,)),
  )
  def test_from_euler_preset(self, test_inputs, test_outputs):
    """Tests that an angle maps to correct matrix."""
    self.assert_output_is_correct(rotation_matrix_2d.from_euler, test_inputs,
                                  test_outputs)

  @parameterized.parameters(
      ((1,),),
      ((None, 1),),
  )
  def test_from_euler_with_small_angles_approximation_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        rotation_matrix_2d.from_euler_with_small_angles_approximation, shapes)

  @parameterized.parameters(
      ("must have exactly 1 dimensions in axis -1", (None,)),)
  def test_from_euler_with_small_angles_approximation_exception_raised(
      self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        rotation_matrix_2d.from_euler_with_small_angles_approximation,
        error_msg, shape)

  def test_from_euler_with_small_angles_approximation_random(self):
    """Tests small_angles approximation by comparing to exact calculation."""
    # Only generate small angles. For a test tolerance of 1e-3, 0.17 was found
    # empirically to be the range where the small angle approximation works.
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.17, max_angle=0.17, dimensions=1)

    exact_matrix = rotation_matrix_2d.from_euler(random_euler_angles)
    approximate_matrix = (
        rotation_matrix_2d.from_euler_with_small_angles_approximation(
            random_euler_angles))

    self.assertAllClose(exact_matrix, approximate_matrix, atol=1e-3)

  @parameterized.parameters(
      ((2, 2),),
      ((None, 2, 2),),
  )
  def test_inverse_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_2d.inverse, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (2,)),
      ("must have exactly 2 dimensions in axis -1", (2, None)),
      ("must have exactly 2 dimensions in axis -2", (None, 2)),
  )
  def test_inverse_exception_raised(self, error_msg, *shapes):
    """Checks the inputs of the inverse function."""
    self.assert_exception_is_raised(rotation_matrix_2d.inverse, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_preset_test_rotation_matrices_2d()

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.inverse, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_random_test_rotation_matrix_2d()

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.inverse, [x_init])

  def test_inverse_random(self):
    """Checks that inverting rotated points results in no transformation."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        dimensions=1)
    tensor_shape = random_euler_angles.shape[:-1]

    random_matrix = rotation_matrix_2d.from_euler(random_euler_angles)
    random_point = np.random.normal(size=tensor_shape + (2,))
    rotated_random_points = rotation_matrix_2d.rotate(random_point,
                                                      random_matrix)
    predicted_invert_random_matrix = rotation_matrix_2d.inverse(random_matrix)
    predicted_invert_rotated_random_points = rotation_matrix_2d.rotate(
        rotated_random_points, predicted_invert_random_matrix)

    self.assertAllClose(
        random_point, predicted_invert_rotated_random_points, rtol=1e-6)

  @parameterized.parameters(
      ((2, 2),),
      ((None, 2, 2),),
  )
  def test_is_valid_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_2d.inverse, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (2,)),
      ("must have exactly 2 dimensions in axis -1", (2, None)),
      ("must have exactly 2 dimensions in axis -2", (None, 2)),
  )
  def test_is_valid_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rotation_matrix_2d.is_valid, error_msg,
                                    shape)

  @parameterized.parameters(
      ((2,), (2, 2)),
      ((None, 2), (None, 2, 2)),
      ((1, 2), (1, 2, 2)),
      ((2, 2), (2, 2, 2)),
      ((2,), (1, 2, 2)),
      ((1, 2), (2, 2)),
  )
  def test_rotate_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_2d.rotate, shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (None,), (2, 2)),
      ("must have a rank greater than 1", (2,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (2, None)),
      ("must have exactly 2 dimensions in axis -2", (2,), (None, 2)),
  )
  def test_rotate_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_2d.rotate, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_preset(self):
    """Test the Jacobian of the rotate function."""
    x_matrix_init = test_helpers.generate_preset_test_rotation_matrices_2d()
    tensor_shape = x_matrix_init.shape[:-2] + (2,)
    x_point_init = np.random.uniform(size=tensor_shape)

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.rotate,
                                       [x_point_init, x_matrix_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_random(self):
    """Test the Jacobian of the rotate function."""
    x_matrix_init = test_helpers.generate_random_test_rotation_matrix_2d()
    tensor_shape = x_matrix_init.shape[:-2] + (2,)
    x_point_init = np.random.uniform(size=tensor_shape)

    self.assert_jacobian_is_correct_fn(rotation_matrix_2d.rotate,
                                       [x_point_init, x_matrix_init])

  @parameterized.parameters(
      ((td.AXIS_2D_0, td.ANGLE_90), (td.AXIS_2D_0,)),
      ((td.AXIS_2D_X, td.ANGLE_90), (td.AXIS_2D_Y,)),
  )
  def test_rotate_preset(self, test_inputs, test_outputs):
    """Tests that the rotate function correctly rotates points."""

    def func(test_point, test_angle):
      random_matrix = rotation_matrix_2d.from_euler(test_angle)
      return rotation_matrix_2d.rotate(test_point, random_matrix)

    self.assert_output_is_correct(func, test_inputs, test_outputs)


if __name__ == "__main__":
  test_case.main()
