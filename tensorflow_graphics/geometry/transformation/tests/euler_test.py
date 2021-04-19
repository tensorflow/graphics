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
"""Tests for euler-related utiliy functions."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import euler
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.geometry.transformation.tests import test_data as td
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class EulerTest(test_case.TestCase):

  @parameterized.parameters(
      ((3,), (1,)),
      ((None, 3), (None, 1)),
  )
  def test_from_axis_angle_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(euler.from_axis_angle, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions", (None,), (1,)),
      ("must have exactly 1 dimensions", (3,), (None,)),
  )
  def test_from_axis_angle_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(euler.from_axis_angle, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_axis_angle_jacobian_preset(self):
    """Test the Jacobian of the from_axis_angle function."""
    x_axis_init, x_angle_init = test_helpers.generate_preset_test_axis_angle()

    self.assert_jacobian_is_finite_fn(euler.from_axis_angle,
                                      [x_axis_init, x_angle_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_axis_angle_jacobian_random(self):
    """Test the Jacobian of the from_axis_angle function."""
    x_axis_init, x_angle_init = test_helpers.generate_random_test_axis_angle()

    self.assert_jacobian_is_finite_fn(euler.from_axis_angle,
                                      [x_axis_init, x_angle_init])

  def test_from_axis_angle_random(self):
    """Checks that Euler angles can be retrieved from an axis-angle."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    random_axis, random_angle = axis_angle.from_euler(random_euler_angles)
    predicted_matrix = rotation_matrix_3d.from_axis_angle(
        random_axis, random_angle)

    self.assertAllClose(random_matrix, predicted_matrix, atol=1e-3)

  def test_from_axis_angle_preset(self):
    """Checks that Euler angles can be retrieved from axis-angle."""
    preset_euler_angles = test_helpers.generate_preset_test_euler_angles()

    random_matrix = rotation_matrix_3d.from_euler(preset_euler_angles)
    random_axis, random_angle = axis_angle.from_euler(preset_euler_angles)
    predicted_matrix = rotation_matrix_3d.from_axis_angle(
        random_axis, random_angle)

    self.assertAllClose(random_matrix, predicted_matrix, atol=1e-3)

  @parameterized.parameters(
      (td.ANGLE_90,),
      (-td.ANGLE_90,),
  )
  def test_from_axis_angle_gimbal(self, gimbal_configuration):
    """Checks that from_axis_angle works when Ry = pi/2 or -pi/2."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()
    random_euler_angles[..., 1] = gimbal_configuration

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    random_axis, random_angle = axis_angle.from_euler(random_euler_angles)
    predicted_random_angles = euler.from_axis_angle(random_axis, random_angle)
    reconstructed_random_matrices = rotation_matrix_3d.from_euler(
        predicted_random_angles)

    self.assertAllClose(reconstructed_random_matrices, random_matrix, atol=1e-3)

  @parameterized.parameters(
      ((4,),),
      ((None, 4),),
  )
  def test_from_quaternion_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(euler.from_quaternion, shape)

  @parameterized.parameters(
      ("must have exactly 4 dimensions", (None,)),)
  def test_from_quaternion_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(euler.from_quaternion, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_quaternion_jacobian_preset(self):
    """Test the Jacobian of the from_quaternion function."""
    x_init = test_helpers.generate_preset_test_quaternions()

    self.assert_jacobian_is_finite_fn(euler.from_quaternion, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_quaternion_jacobian_random(self):
    """Test the Jacobian of the from_quaternion function."""
    x_init = test_helpers.generate_random_test_quaternions()

    self.assert_jacobian_is_finite_fn(euler.from_quaternion, [x_init])

  @parameterized.parameters(
      (td.ANGLE_90,),
      (-td.ANGLE_90,),
  )
  def test_from_quaternion_gimbal(self, gimbal_configuration):
    """Checks that from_quaternion works when Ry = pi/2 or -pi/2."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()
    random_euler_angles[..., 1] = gimbal_configuration

    random_quaternion = quaternion.from_euler(random_euler_angles)
    random_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    reconstructed_random_matrices = rotation_matrix_3d.from_quaternion(
        random_quaternion)

    self.assertAllClose(reconstructed_random_matrices, random_matrix, atol=2e-3)

  def test_from_quaternion_preset(self):
    """Checks that Euler angles can be retrieved from quaternions."""
    preset_euler_angles = test_helpers.generate_preset_test_euler_angles()

    preset_matrix = rotation_matrix_3d.from_euler(preset_euler_angles)
    preset_quaternion = quaternion.from_euler(preset_euler_angles)
    predicted_matrix = rotation_matrix_3d.from_quaternion(preset_quaternion)

    self.assertAllClose(preset_matrix, predicted_matrix, atol=2e-3)

  def test_from_quaternion_random(self):
    """Checks that Euler angles can be retrieved from quaternions."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    random_quaternion = quaternion.from_rotation_matrix(random_matrix)
    predicted_angles = euler.from_quaternion(random_quaternion)
    predicted_matrix = rotation_matrix_3d.from_euler(predicted_angles)

    self.assertAllClose(random_matrix, predicted_matrix, atol=2e-3)

  @parameterized.parameters(
      ((3, 3),),
      ((None, 3, 3),),
  )
  def test_from_rotation_matrix_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(euler.from_rotation_matrix, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions", (None, 3)),
      ("must have exactly 3 dimensions", (3, None)),
  )
  def test_from_rotation_matrix_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(euler.from_rotation_matrix, error_msg,
                                    shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_rotation_matrix_jacobian_preset(self):
    """Test the Jacobian of the from_rotation_matrix function."""
    if tf.executing_eagerly():
      self.skipTest(reason="Graph mode only test")
    with tf.compat.v1.Session() as sess:
      x_init = np.array(
          sess.run(test_helpers.generate_preset_test_rotation_matrices_3d()))
      x = tf.convert_to_tensor(value=x_init)

      y = euler.from_rotation_matrix(x)

      self.assert_jacobian_is_finite(x, x_init, y)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_rotation_matrix_jacobian_random(self):
    """Test the Jacobian of the from_rotation_matrix function."""
    x_init = test_helpers.generate_random_test_rotation_matrix_3d()

    self.assert_jacobian_is_finite_fn(euler.from_rotation_matrix, [x_init])

  def test_from_rotation_matrix_gimbal(self):
    """Testing that Euler angles can be retrieved in Gimbal lock."""
    angles = test_helpers.generate_random_test_euler_angles()

    angles[..., 1] = np.pi / 2.
    matrix = rotation_matrix_3d.from_euler(angles)
    predicted_angles = euler.from_rotation_matrix(matrix)
    reconstructed_matrices = rotation_matrix_3d.from_euler(predicted_angles)

    self.assertAllClose(reconstructed_matrices, matrix, rtol=1e-3)

    angles[..., 1] = -np.pi / 2.
    matrix = rotation_matrix_3d.from_euler(angles)
    predicted_angles = euler.from_rotation_matrix(matrix)
    reconstructed_matrices = rotation_matrix_3d.from_euler(predicted_angles)

    self.assertAllClose(reconstructed_matrices, matrix, rtol=1e-3)

  def test_from_rotation_matrix_preset(self):
    """Tests that Euler angles can be retrieved from rotation matrices."""
    matrix = test_helpers.generate_preset_test_rotation_matrices_3d()

    predicted_angles = euler.from_rotation_matrix(matrix)
    reconstructed_matrices = rotation_matrix_3d.from_euler(predicted_angles)

    self.assertAllClose(reconstructed_matrices, matrix, rtol=1e-3)

  def test_from_rotation_matrix_random(self):
    """Tests that Euler angles can be retrieved from rotation matrices."""
    matrix = test_helpers.generate_random_test_rotation_matrix_3d()

    predicted_angles = euler.from_rotation_matrix(matrix)
    # There is not a unique mapping from rotation matrices to Euler angles. The
    # following constructs the rotation matrices from the `predicted_angles` and
    # compares them with `matrix`.
    reconstructed_matrices = rotation_matrix_3d.from_euler(predicted_angles)

    self.assertAllClose(reconstructed_matrices, matrix, rtol=1e-3)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_inverse_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(euler.inverse, shape)

  @parameterized.parameters(
      ("must have exactly 3 dimensions", (None,)),)
  def test_inverse_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(euler.inverse, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_preset_test_euler_angles()

    self.assert_jacobian_is_correct_fn(euler.inverse, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_random_test_euler_angles()

    self.assert_jacobian_is_correct_fn(euler.inverse, [x_init])

  def test_inverse_preset(self):
    """Checks that inverse works as intended."""
    preset_euler_angles = test_helpers.generate_preset_test_euler_angles()

    prediction = euler.inverse(preset_euler_angles)
    groundtruth = -preset_euler_angles

    self.assertAllClose(prediction, groundtruth, rtol=1e-3)

  def test_inverse_random(self):
    """Checks that inverse works as intended."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    prediction = euler.inverse(random_euler_angles)
    groundtruth = -random_euler_angles

    self.assertAllClose(prediction, groundtruth, rtol=1e-3)


if __name__ == "__main__":
  test_case.main()
