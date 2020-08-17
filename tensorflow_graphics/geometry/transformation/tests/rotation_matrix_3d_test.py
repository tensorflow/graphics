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
"""Tests for 3d rotation matrix."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.geometry.transformation.tests import test_data as td
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class RotationMatrix3dTest(test_case.TestCase):

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_assert_rotation_matrix_normalized_passthrough(self):
    """Checks that the assert is a passthrough when the flag is False."""
    angles = test_helpers.generate_preset_test_euler_angles()

    matrix_input = rotation_matrix_3d.from_euler(angles)
    matrix_output = rotation_matrix_3d.assert_rotation_matrix_normalized(
        matrix_input)

    self.assertTrue(matrix_input is matrix_output)  # pylint: disable=g-generic-assert

  @parameterized.parameters((np.float32), (np.float64))
  def test_assert_rotation_matrix_normalized_preset(self, dtype):
    """Checks that assert_normalized function works as expected."""
    angles = test_helpers.generate_preset_test_euler_angles().astype(dtype)

    matrix = rotation_matrix_3d.from_euler(angles)
    matrix_rescaled = matrix * 1.01
    matrix_normalized = rotation_matrix_3d.assert_rotation_matrix_normalized(
        matrix)
    self.evaluate(matrix_normalized)

    with self.assertRaises(tf.errors.InvalidArgumentError):  # pylint: disable=g-error-prone-assert-raises
      self.evaluate(rotation_matrix_3d.assert_rotation_matrix_normalized(
          matrix_rescaled))

  @parameterized.parameters(
      ((3, 3),),
      ((None, 3, 3),),
  )
  def test_assert_rotation_matrix_normalized_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        rotation_matrix_3d.assert_rotation_matrix_normalized, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, None)),
      ("must have exactly 3 dimensions in axis -2", (None, 3)),
  )
  def test_assert_rotation_matrix_normalized_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(
        rotation_matrix_3d.assert_rotation_matrix_normalized, error_msg, shapes)

  @parameterized.parameters(
      ((3,), (1,)),
      ((None, 3), (None, 1)),
      ((1, 3), (1, 1)),
      ((2, 3), (2, 1)),
      ((1, 3), (1,)),
      ((3,), (1, 1)),
  )
  def test_from_axis_angle_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.from_axis_angle,
                                        shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,), (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (None,)),
  )
  def test_from_axis_angle_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_3d.from_axis_angle,
                                    error_msg, shapes)

  def test_from_axis_angle_normalized_preset(self):
    """Tests that axis-angles can be converted to rotation matrices."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    axis, angle = axis_angle.from_euler(euler_angles)
    matrix_axis_angle = rotation_matrix_3d.from_axis_angle(axis, angle)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix_axis_angle),
        np.ones(euler_angles.shape[0:-1] + (1,)))

  def test_from_axis_angle_normalized_random(self):
    """Tests that axis-angles can be converted to rotation matrices."""
    tensor_shape = np.random.randint(1, 10, size=np.random.randint(3)).tolist()
    random_axis = np.random.normal(size=tensor_shape + [3])
    random_axis /= np.linalg.norm(random_axis, axis=-1, keepdims=True)
    random_angle = np.random.normal(size=tensor_shape + [1])

    matrix_axis_angle = rotation_matrix_3d.from_axis_angle(
        random_axis, random_angle)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix_axis_angle),
        np.ones(tensor_shape + [1]))

  @parameterized.parameters(
      ((td.AXIS_3D_X, td.ANGLE_45), (td.MAT_3D_X_45,)),
      ((td.AXIS_3D_Y, td.ANGLE_45), (td.MAT_3D_Y_45,)),
      ((td.AXIS_3D_Z, td.ANGLE_45), (td.MAT_3D_Z_45,)),
      ((td.AXIS_3D_X, td.ANGLE_90), (td.MAT_3D_X_90,)),
      ((td.AXIS_3D_Y, td.ANGLE_90), (td.MAT_3D_Y_90,)),
      ((td.AXIS_3D_Z, td.ANGLE_90), (td.MAT_3D_Z_90,)),
      ((td.AXIS_3D_X, td.ANGLE_180), (td.MAT_3D_X_180,)),
      ((td.AXIS_3D_Y, td.ANGLE_180), (td.MAT_3D_Y_180,)),
      ((td.AXIS_3D_Z, td.ANGLE_180), (td.MAT_3D_Z_180,)),
  )
  def test_from_axis_angle_preset(self, test_inputs, test_outputs):
    """Tests that an axis-angle maps to correct matrix."""
    self.assert_output_is_correct(rotation_matrix_3d.from_axis_angle,
                                  test_inputs, test_outputs)

  def test_from_axis_angle_random(self):
    """Tests conversion to matrix."""
    tensor_shape = np.random.randint(1, 10, size=np.random.randint(3)).tolist()
    random_axis = np.random.normal(size=tensor_shape + [3])
    random_axis /= np.linalg.norm(random_axis, axis=-1, keepdims=True)
    random_angle = np.random.normal(size=tensor_shape + [1])

    matrix_axis_angle = rotation_matrix_3d.from_axis_angle(
        random_axis, random_angle)
    random_quaternion = quaternion.from_axis_angle(random_axis, random_angle)
    matrix_quaternion = rotation_matrix_3d.from_quaternion(random_quaternion)

    self.assertAllClose(matrix_axis_angle, matrix_quaternion, rtol=1e-3)
    # Checks that resulting rotation matrices are normalized.
    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix_axis_angle),
        np.ones(tensor_shape + [1]))

  @parameterized.parameters(
      ((td.AXIS_3D_X, td.ANGLE_90, td.AXIS_3D_X), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_X, td.ANGLE_90, td.AXIS_3D_Y), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_X, -td.ANGLE_90, td.AXIS_3D_Z), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, -td.ANGLE_90, td.AXIS_3D_X), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_Y, td.ANGLE_90, td.AXIS_3D_Y), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, td.ANGLE_90, td.AXIS_3D_Z), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.ANGLE_90, td.AXIS_3D_X), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Z, -td.ANGLE_90, td.AXIS_3D_Y), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.ANGLE_90, td.AXIS_3D_Z), (td.AXIS_3D_Z,)),
  )
  def test_from_axis_angle_rotate_vector_preset(self, test_inputs,
                                                test_outputs):
    """Tests the directionality of axis-angle rotations."""

    def func(axis, angle, point):
      matrix = rotation_matrix_3d.from_axis_angle(axis, angle)
      return rotation_matrix_3d.rotate(point, matrix)

    self.assert_output_is_correct(func, test_inputs, test_outputs)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
      ((2, 3),),
  )
  def test_from_euler_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.from_euler, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,)),)
  def test_from_euler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_3d.from_euler, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_jacobian_preset(self):
    """Test the Jacobian of the from_euler function."""
    x_init = test_helpers.generate_preset_test_euler_angles()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.from_euler, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_jacobian_random(self):
    """Test the Jacobian of the from_euler function."""
    x_init = test_helpers.generate_random_test_euler_angles()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.from_euler, [x_init])

  def test_from_euler_normalized_preset(self):
    """Tests that euler angles can be converted to rotation matrices."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    matrix = rotation_matrix_3d.from_euler(euler_angles)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix),
        np.ones(euler_angles.shape[0:-1] + (1,)))

  def test_from_euler_normalized_random(self):
    """Tests that euler angles can be converted to rotation matrices."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    matrix = rotation_matrix_3d.from_euler(random_euler_angles)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix),
        np.ones(random_euler_angles.shape[0:-1] + (1,)))

  @parameterized.parameters(
      ((td.AXIS_3D_0,), (td.MAT_3D_ID,)),
      ((td.ANGLE_45 * td.AXIS_3D_X,), (td.MAT_3D_X_45,)),
      ((td.ANGLE_45 * td.AXIS_3D_Y,), (td.MAT_3D_Y_45,)),
      ((td.ANGLE_45 * td.AXIS_3D_Z,), (td.MAT_3D_Z_45,)),
      ((td.ANGLE_90 * td.AXIS_3D_X,), (td.MAT_3D_X_90,)),
      ((td.ANGLE_90 * td.AXIS_3D_Y,), (td.MAT_3D_Y_90,)),
      ((td.ANGLE_90 * td.AXIS_3D_Z,), (td.MAT_3D_Z_90,)),
      ((td.ANGLE_180 * td.AXIS_3D_X,), (td.MAT_3D_X_180,)),
      ((td.ANGLE_180 * td.AXIS_3D_Y,), (td.MAT_3D_Y_180,)),
      ((td.ANGLE_180 * td.AXIS_3D_Z,), (td.MAT_3D_Z_180,)),
  )
  def test_from_euler_preset(self, test_inputs, test_outputs):
    """Tests that Euler angles create the expected matrix."""
    self.assert_output_is_correct(rotation_matrix_3d.from_euler, test_inputs,
                                  test_outputs)

  def test_from_euler_random(self):
    """Tests that Euler angles produce the same result as axis-angle."""
    angles = test_helpers.generate_random_test_euler_angles()
    matrix = rotation_matrix_3d.from_euler(angles)
    tensor_tile = angles.shape[:-1]

    x_axis = np.tile(td.AXIS_3D_X, tensor_tile + (1,))
    y_axis = np.tile(td.AXIS_3D_Y, tensor_tile + (1,))
    z_axis = np.tile(td.AXIS_3D_Z, tensor_tile + (1,))
    x_angle = np.expand_dims(angles[..., 0], axis=-1)
    y_angle = np.expand_dims(angles[..., 1], axis=-1)
    z_angle = np.expand_dims(angles[..., 2], axis=-1)
    x_rotation = rotation_matrix_3d.from_axis_angle(x_axis, x_angle)
    y_rotation = rotation_matrix_3d.from_axis_angle(y_axis, y_angle)
    z_rotation = rotation_matrix_3d.from_axis_angle(z_axis, z_angle)
    expected_matrix = tf.matmul(z_rotation, tf.matmul(y_rotation, x_rotation))

    self.assertAllClose(expected_matrix, matrix, rtol=1e-3)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_from_euler_with_small_angles_approximation_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        rotation_matrix_3d.from_euler_with_small_angles_approximation, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,)),)
  def test_from_euler_with_small_angles_approximation_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        rotation_matrix_3d.from_euler_with_small_angles_approximation,
        error_msg, shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_with_small_angles_approximation_jacobian_random(self):
    """Test the Jacobian of from_euler_with_small_angles_approximation."""
    x_init = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.17, max_angle=0.17)

    self.assert_jacobian_is_correct_fn(
        rotation_matrix_3d.from_euler_with_small_angles_approximation, [x_init])

  def test_from_euler_with_small_angles_approximation_random(self):
    """Tests small_angles approximation by comparing to exact calculation."""
    # Only generate small angles. For a test tolerance of 1e-3, 0.16 was found
    # empirically to be the range where the small angle approximation works.
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.16, max_angle=0.16)

    exact_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    approximate_matrix = (
        rotation_matrix_3d.from_euler_with_small_angles_approximation(
            random_euler_angles))

    self.assertAllClose(exact_matrix, approximate_matrix, atol=1e-3)

  @parameterized.parameters(
      ((4,),),
      ((None, 4),),
  )
  def test_from_quaternion_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.from_quaternion,
                                        shapes)

  @parameterized.parameters(
      ("must have exactly 4 dimensions in axis -1", (None,)),)
  def test_from_quaternion_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_3d.from_quaternion,
                                    error_msg, shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_quaternion_jacobian_preset(self):
    """Test the Jacobian of the from_quaternion function."""
    x_init = test_helpers.generate_preset_test_quaternions()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.from_quaternion,
                                       [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_quaternion_jacobian_random(self):
    """Test the Jacobian of the from_quaternion function."""
    x_init = test_helpers.generate_random_test_quaternions()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.from_quaternion,
                                       [x_init])

  def test_from_quaternion_normalized_preset(self):
    """Tests that quaternions can be converted to rotation matrices."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    quat = quaternion.from_euler(euler_angles)
    matrix_quat = rotation_matrix_3d.from_quaternion(quat)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(matrix_quat),
        np.ones(euler_angles.shape[0:-1] + (1,)))

  def test_from_quaternion_normalized_random(self):
    """Tests that random quaternions can be converted to rotation matrices."""
    random_quaternion = test_helpers.generate_random_test_quaternions()
    tensor_shape = random_quaternion.shape[:-1]

    random_matrix = rotation_matrix_3d.from_quaternion(random_quaternion)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(random_matrix),
        np.ones(tensor_shape + (1,)))

  def test_from_quaternion_preset(self):
    """Tests that a quaternion maps to correct matrix."""
    preset_quaternions = test_helpers.generate_preset_test_quaternions()

    preset_matrices = test_helpers.generate_preset_test_rotation_matrices_3d()

    self.assertAllClose(preset_matrices,
                        rotation_matrix_3d.from_quaternion(preset_quaternions))

  def test_from_quaternion_random(self):
    """Tests conversion to matrix."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    random_quaternions = quaternion.from_euler(random_euler_angles)
    random_rotation_matrices = rotation_matrix_3d.from_euler(
        random_euler_angles)

    self.assertAllClose(random_rotation_matrices,
                        rotation_matrix_3d.from_quaternion(random_quaternions))

  @parameterized.parameters(
      ((3, 3),),
      ((None, 3, 3),),
      ((2, 3, 3),),
  )
  def test_inverse_exception_not_raised(self, *shapes):
    """Checks the inputs of the rotate function."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.inverse, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, None)),
      ("must have exactly 3 dimensions in axis -2", (None, 3)),
  )
  def test_inverse_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(rotation_matrix_3d.inverse, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_preset_test_rotation_matrices_3d()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.inverse, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_random_test_rotation_matrix_3d()

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.inverse, [x_init])

  def test_inverse_normalized_random(self):
    """Checks that inverted rotation matrices are valid rotations."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angle)
    predicted_invert_random_matrix = rotation_matrix_3d.inverse(random_matrix)

    self.assertAllEqual(
        rotation_matrix_3d.is_valid(predicted_invert_random_matrix),
        np.ones(tensor_tile + (1,)))

  def test_inverse_random(self):
    """Checks that inverting rotated points results in no transformation."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]
    random_matrix = rotation_matrix_3d.from_euler(random_euler_angle)
    random_point = np.random.normal(size=tensor_tile + (3,))

    rotated_random_points = rotation_matrix_3d.rotate(random_point,
                                                      random_matrix)
    predicted_invert_random_matrix = rotation_matrix_3d.inverse(random_matrix)
    predicted_invert_rotated_random_points = rotation_matrix_3d.rotate(
        rotated_random_points, predicted_invert_random_matrix)

    self.assertAllClose(
        random_point, predicted_invert_rotated_random_points, rtol=1e-6)

  @parameterized.parameters(
      ((3, 3),),
      ((None, 3, 3),),
      ((2, 3, 3),),
  )
  def test_is_valid_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.is_valid, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, None)),
      ("must have exactly 3 dimensions in axis -2", (None, 3)),
  )
  def test_is_valid_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(rotation_matrix_3d.is_valid, error_msg,
                                    shape)

  def test_is_valid_random(self):
    """Tests that is_valid works as intended."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]

    rotation_matrix = rotation_matrix_3d.from_euler(random_euler_angle)
    pred_normalized = rotation_matrix_3d.is_valid(rotation_matrix)

    with self.subTest(name="all_normalized"):
      self.assertAllEqual(pred_normalized,
                          np.ones(shape=tensor_tile + (1,), dtype=bool))

    with self.subTest(name="non_orthonormal"):
      test_matrix = np.array([[2., 0., 0.], [0., 0.5, 0], [0., 0., 1.]])
      pred_normalized = rotation_matrix_3d.is_valid(test_matrix)

      self.assertAllEqual(pred_normalized, np.zeros(shape=(1,), dtype=bool))

    with self.subTest(name="negative_orthonormal"):
      test_matrix = np.array([[1., 0., 0.], [0., -1., 0.], [0., 0., 1.]])
      pred_normalized = rotation_matrix_3d.is_valid(test_matrix)

      self.assertAllEqual(pred_normalized, np.zeros(shape=(1,), dtype=bool))

  @parameterized.parameters(
      ((3,), (3, 3)),
      ((None, 3), (None, 3, 3)),
      ((1, 3), (1, 3, 3)),
      ((2, 3), (2, 3, 3)),
      ((3,), (1, 3, 3)),
      ((1, 3), (3, 3)),
  )
  def test_rotate_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(rotation_matrix_3d.rotate, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,), (3, 3)),
      ("must have a rank greater than 1", (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3, None)),
      ("must have exactly 3 dimensions in axis -2", (3,), (None, 3)),
  )
  def test_rotate_exception_raised(self, error_msg, *shapes):
    """Checks the inputs of the rotate function."""
    self.assert_exception_is_raised(rotation_matrix_3d.rotate, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_preset(self):
    """Test the Jacobian of the rotate function."""
    x_matrix_init = test_helpers.generate_preset_test_rotation_matrices_3d()
    tensor_shape = x_matrix_init.shape[:-1]
    x_point_init = np.random.uniform(size=tensor_shape)

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.rotate,
                                       [x_point_init, x_matrix_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_random(self):
    """Test the Jacobian of the rotate function."""
    x_matrix_init = test_helpers.generate_random_test_rotation_matrix_3d()
    tensor_shape = x_matrix_init.shape[:-1]
    x_point_init = np.random.uniform(size=tensor_shape)

    self.assert_jacobian_is_correct_fn(rotation_matrix_3d.rotate,
                                       [x_point_init, x_matrix_init])

  @parameterized.parameters(
      ((td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_X), (td.AXIS_3D_X,)),
      ((td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_Y), (td.AXIS_3D_Z,)),
      ((-td.ANGLE_90 * td.AXIS_3D_X, td.AXIS_3D_Z), (td.AXIS_3D_Y,)),
      ((-td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_X), (td.AXIS_3D_Z,)),
      ((td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_Y), (td.AXIS_3D_Y,)),
      ((td.ANGLE_90 * td.AXIS_3D_Y, td.AXIS_3D_Z), (td.AXIS_3D_X,)),
      ((td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_X), (td.AXIS_3D_Y,)),
      ((-td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_Y), (td.AXIS_3D_X,)),
      ((td.ANGLE_90 * td.AXIS_3D_Z, td.AXIS_3D_Z), (td.AXIS_3D_Z,)),
  )
  def test_rotate_vector_preset(self, test_inputs, test_outputs):
    """Tests that the rotate function produces the expected results."""

    def func(angles, point):
      matrix = rotation_matrix_3d.from_euler(angles)
      return rotation_matrix_3d.rotate(point, matrix)

    self.assert_output_is_correct(func, test_inputs, test_outputs)

  def test_rotate_vs_rotate_quaternion_random(self):
    """Tests that the rotate provide the same results as quaternion.rotate."""
    random_euler_angle = test_helpers.generate_random_test_euler_angles()
    tensor_tile = random_euler_angle.shape[:-1]

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angle)
    random_quaternion = quaternion.from_rotation_matrix(random_matrix)
    random_point = np.random.normal(size=tensor_tile + (3,))
    ground_truth = quaternion.rotate(random_point, random_quaternion)
    prediction = rotation_matrix_3d.rotate(random_point, random_matrix)

    self.assertAllClose(ground_truth, prediction, rtol=1e-6)


if __name__ == "__main__":
  test_case.main()
