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
"""Tests for axis-angle."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
from six.moves import range
from six.moves import zip
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import axis_angle
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class AxisAngleTest(test_case.TestCase):

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_from_euler_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(axis_angle.from_euler, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,)),)
  def test_from_euler_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(axis_angle.from_euler, error_msg, shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_euler_jacobian_random(self):
    """Test the Jacobian of the from_euler function.

    Note:
      Preset angles are not tested as the gradient of tf.norm is NaN at 0.
    """
    x_init = test_helpers.generate_random_test_euler_angles()

    self.assert_jacobian_is_finite_fn(lambda x: axis_angle.from_euler(x)[0],
                                      [x_init])
    self.assert_jacobian_is_finite_fn(lambda x: axis_angle.from_euler(x)[1],
                                      [x_init])

  def test_from_euler_random(self):
    """Tests that from_euler allows to perform the expect rotation of points."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()
    tensor_shape = random_euler_angles.shape[:-1]
    random_point = np.random.normal(size=tensor_shape + (3,))

    random_matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    random_axis, random_angle = axis_angle.from_euler(random_euler_angles)
    rotated_with_matrix = rotation_matrix_3d.rotate(random_point, random_matrix)
    rotated_with_axis_angle = axis_angle.rotate(random_point, random_axis,
                                                random_angle)

    self.assertAllClose(rotated_with_matrix, rotated_with_axis_angle)

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
      ((2, 3),),
  )
  def test_from_euler_with_small_angles_approximation_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        axis_angle.from_euler_with_small_angles_approximation, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,)),)
  def test_from_euler_with_small_angles_approximation_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        axis_angle.from_euler_with_small_angles_approximation, error_msg,
        shapes)

  def test_from_euler_normalized_preset(self):
    """Tests that from_euler allows build normalized axis-angles."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    axis, angle = axis_angle.from_euler(euler_angles)

    self.assertAllEqual(
        axis_angle.is_normalized(axis, angle), np.ones(angle.shape, dtype=bool))

  def test_from_euler_normalized_random(self):
    """Tests that from_euler allows build normalized axis-angles."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    random_axis, random_angle = axis_angle.from_euler(random_euler_angles)

    self.assertAllEqual(
        axis_angle.is_normalized(random_axis, random_angle),
        np.ones(shape=random_angle.shape))

  def test_from_euler_with_small_angles_approximation_random(self):
    # Only generate small angles. For a test tolerance of 1e-3, 0.23 was found
    # empirically to be the range where the small angle approximation works.
    random_euler_angles = test_helpers.generate_random_test_euler_angles(
        min_angle=-0.23, max_angle=0.23)

    exact_axis_angle = axis_angle.from_euler(random_euler_angles)
    approximate_axis_angle = (
        axis_angle.from_euler_with_small_angles_approximation(
            random_euler_angles))

    self.assertAllClose(exact_axis_angle, approximate_axis_angle, atol=1e-3)

  @parameterized.parameters(
      ((4,),),
      ((None, 4),),
      ((2, 4),),
  )
  def test_from_quaternion_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(axis_angle.from_quaternion, shape)

  @parameterized.parameters(
      ("must have exactly 4 dimensions in axis -1", (None,)),)
  def test_from_quaternion_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(axis_angle.from_quaternion, error_msg,
                                    shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_quaternion_jacobian_random(self):
    """Test the Jacobian of the from_quaternion function.

    Note:
      Preset angles are not tested as the gradient of tf.norm is NaN a 0.
    """
    x_init = test_helpers.generate_random_test_quaternions()

    self.assert_jacobian_is_finite_fn(
        lambda x: axis_angle.from_quaternion(x)[0], [x_init])
    self.assert_jacobian_is_finite_fn(
        lambda x: axis_angle.from_quaternion(x)[1], [x_init])

  def test_from_quaternion_normalized_preset(self):
    """Tests that from_quaternion returns normalized axis-angles."""
    euler_angles = test_helpers.generate_preset_test_euler_angles()

    quat = quaternion.from_euler(euler_angles)
    axis, angle = axis_angle.from_quaternion(quat)

    self.assertAllEqual(
        axis_angle.is_normalized(axis, angle), np.ones(angle.shape, dtype=bool))

  def test_from_quaternion_normalized_random(self):
    """Tests that from_quaternion returns normalized axis-angles."""
    random_quaternions = test_helpers.generate_random_test_quaternions()

    random_axis, random_angle = axis_angle.from_quaternion(random_quaternions)

    self.assertAllEqual(
        axis_angle.is_normalized(random_axis, random_angle),
        np.ones(random_angle.shape))

  def test_from_quaternion_preset(self):
    """Tests that axis_angle.from_quaternion produces the expected result."""
    preset_euler_angles = test_helpers.generate_preset_test_euler_angles()

    preset_quaternions = quaternion.from_euler(preset_euler_angles)
    preset_axis_angle = axis_angle.from_euler(preset_euler_angles)

    self.assertAllClose(
        preset_axis_angle,
        axis_angle.from_quaternion(preset_quaternions),
        rtol=1e-3)

  def test_from_quaternion_random(self):
    """Tests that axis_angle.from_quaternion produces the expected result."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    random_quaternions = quaternion.from_euler(random_euler_angles)
    random_axis_angle = axis_angle.from_euler(random_euler_angles)

    self.assertAllClose(
        random_axis_angle,
        axis_angle.from_quaternion(random_quaternions),
        rtol=1e-3)

  @parameterized.parameters(
      ((3, 3),),
      ((None, 3, 3),),
  )
  def test_from_rotation_matrix_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(axis_angle.from_rotation_matrix, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, None)),
      ("must have exactly 3 dimensions in axis -2", (None, 3)),
  )
  def test_from_rotation_matrix_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(axis_angle.from_rotation_matrix, error_msg,
                                    shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_rotation_matrix_jacobian_random(self):
    """Test the Jacobian of the from_rotation_matrix function.

    Note:
      Preset angles are not tested as the gradient of tf.norm is NaN a 0.
    """
    x_init = test_helpers.generate_random_test_rotation_matrix_3d()

    self.assert_jacobian_is_finite_fn(
        lambda x: axis_angle.from_rotation_matrix(x)[0], [x_init])
    self.assert_jacobian_is_finite_fn(
        lambda x: axis_angle.from_rotation_matrix(x)[1], [x_init])

  def test_from_rotation_matrix_normalized_preset(self):
    """Tests that from_rotation_matrix returns normalized axis-angles."""
    preset_euler_angles = test_helpers.generate_preset_test_euler_angles()

    matrix = rotation_matrix_3d.from_euler(preset_euler_angles)
    axis, angle = axis_angle.from_rotation_matrix(matrix)

    self.assertAllEqual(
        axis_angle.is_normalized(axis, angle), np.ones(angle.shape, dtype=bool))

  def test_from_rotation_matrix_normalized_random(self):
    """Tests that from_rotation_matrix returns normalized axis-angles."""
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    matrix = rotation_matrix_3d.from_euler(random_euler_angles)
    axis, angle = axis_angle.from_rotation_matrix(matrix)

    self.assertAllEqual(
        axis_angle.is_normalized(axis, angle), np.ones(angle.shape, dtype=bool))

  def test_from_rotation_matrix_random(self):
    """Tests rotation around Z axis."""

    def get_rotation_matrix_around_z(angle_rad):
      return np.array([
          [np.cos(angle_rad), -np.sin(angle_rad), 0],
          [np.sin(angle_rad), np.cos(angle_rad), 0],
          [0, 0, 1],
      ])

    tensor_size = np.random.randint(10)
    angle = (
        np.array([
            np.deg2rad(np.random.randint(720) - 360) for _ in range(tensor_size)
        ]).reshape((tensor_size, 1)))
    rotation_matrix = [get_rotation_matrix_around_z(i[0]) for i in angle]
    rotation_matrix = np.array(rotation_matrix).reshape((tensor_size, 3, 3))
    tf_axis, tf_angle = axis_angle.from_rotation_matrix(rotation_matrix)
    axis = np.tile([[0., 0., 1.]], (angle.shape[0], 1))
    tf_quat_gt = quaternion.from_axis_angle(axis, angle)
    tf_quat = quaternion.from_axis_angle(tf_axis, tf_angle)
    # Compare quaternions since axis orientation and angle ambiguity will
    # lead to more complex comparisons.
    for quat_gt, quat in zip(self.evaluate(tf_quat_gt), self.evaluate(tf_quat)):
      # Remember that q=-q for any quaternion.
      pos = np.allclose(quat_gt, quat)
      neg = np.allclose(quat_gt, -quat)

      self.assertTrue(pos or neg)

  @parameterized.parameters(
      ((3,), (1,)),
      ((None, 3), (None, 1)),
      ((2, 3), (2, 1)),
      ((1, 3), (1,)),
      ((3,), (1, 1)),
  )
  def test_inverse_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(axis_angle.inverse, shape)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,), (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (None,)),
  )
  def test_inverse_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(axis_angle.inverse, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_axis_init, x_angle_init = test_helpers.generate_preset_test_axis_angle()

    if tf.executing_eagerly():
      # Because axis is returned as is, gradient calculation fails in graph mode
      # but not in eager mode. This is a side effect of having a graph rather
      # than a problem of the function.
      with self.subTest("axis"):
        self.assert_jacobian_is_correct_fn(
            lambda x: axis_angle.inverse(x, x_angle_init)[0], [x_axis_init])

    with self.subTest("angle"):
      self.assert_jacobian_is_correct_fn(
          lambda x: axis_angle.inverse(x_axis_init, x)[1], [x_angle_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_axis_init, x_angle_init = test_helpers.generate_random_test_axis_angle()

    if tf.executing_eagerly():
      # Because axis is returned as is, gradient calculation fails in graph mode
      # but not in eager mode. This is a side effect of having a graph rather
      # than a problem of the function.
      with self.subTest("axis"):
        self.assert_jacobian_is_correct_fn(
            lambda x: axis_angle.inverse(1.0 * x, x_angle_init)[0],
            [x_axis_init])

    with self.subTest("angle"):
      self.assert_jacobian_is_correct_fn(
          lambda x: axis_angle.inverse(x_axis_init, x)[1], [x_angle_init])

  def test_inverse_normalized_random(self):
    """Tests that axis-angle inversion return a normalized axis-angle."""
    random_axis, random_angle = test_helpers.generate_random_test_axis_angle()

    inverse_axis, inverse_angle = axis_angle.inverse(random_axis, random_angle)

    self.assertAllEqual(
        axis_angle.is_normalized(inverse_axis, inverse_angle),
        np.ones(random_angle.shape))

  def test_inverse_random(self):
    """Tests axis-angle inversion."""
    random_axis, random_angle = test_helpers.generate_random_test_axis_angle()

    inverse_axis, inverse_angle = axis_angle.inverse(random_axis, random_angle)

    self.assertAllClose(inverse_axis, random_axis, rtol=1e-3)
    self.assertAllClose(inverse_angle, -random_angle, rtol=1e-3)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,), (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (None,)),
  )
  def test_is_normalized_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(axis_angle.is_normalized, error_msg, shape)

  def test_is_normalized_random(self):
    """Tests that is_normalized works as intended."""
    # Samples normalized axis-angles.
    random_euler_angles = test_helpers.generate_random_test_euler_angles()

    with self.subTest(name=("is_normalized")):
      random_axis, random_angle = axis_angle.from_euler(random_euler_angles)
      pred = axis_angle.is_normalized(random_axis, random_angle)

      self.assertAllEqual(np.ones(shape=random_angle.shape, dtype=bool), pred)

    with self.subTest(name=("is_not_normalized")):
      random_axis *= 1.01
      pred = axis_angle.is_normalized(random_axis, random_angle)

      self.assertAllEqual(np.zeros(shape=random_angle.shape, dtype=bool), pred)

  @parameterized.parameters(
      ((3,), (3,), (1,)),
      ((None, 3), (None, 3), (None, 1)),
      ((2, 3), (2, 3), (2, 1)),
      ((3,), (1, 3), (1, 2, 1)),
      ((1, 2, 3), (1, 3), (1,)),
      ((3,), (1, 3), (1,)),
  )
  def test_rotate_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(axis_angle.rotate, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (1,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (3,), (2,)),
  )
  def test_rotate_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(axis_angle.rotate, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_preset(self):
    """Test the Jacobian of the rotate function."""
    x_axis_init, x_angle_init = test_helpers.generate_preset_test_axis_angle()
    x_point_init = np.random.uniform(size=x_axis_init.shape)

    self.assert_jacobian_is_correct_fn(
        axis_angle.rotate, [x_point_init, x_axis_init, x_angle_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_rotate_jacobian_random(self):
    """Test the Jacobian of the rotate function."""
    x_axis_init, x_angle_init = test_helpers.generate_random_test_axis_angle()
    x_point_init = np.random.uniform(size=x_axis_init.shape)

    self.assert_jacobian_is_correct_fn(
        axis_angle.rotate, [x_point_init, x_axis_init, x_angle_init])

  def test_rotate_random(self):
    """Tests that the rotate provide the same results as quaternion.rotate."""
    random_axis, random_angle = test_helpers.generate_random_test_axis_angle()
    tensor_shape = random_angle.shape[:-1]
    random_point = np.random.normal(size=tensor_shape + (3,))

    random_quaternion = quaternion.from_axis_angle(random_axis, random_angle)
    ground_truth = quaternion.rotate(random_point, random_quaternion)
    prediction = axis_angle.rotate(random_point, random_axis, random_angle)

    self.assertAllClose(ground_truth, prediction, rtol=1e-6)


if __name__ == "__main__":
  test_case.main()
