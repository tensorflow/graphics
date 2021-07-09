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
"""Tests for dual quaternion."""

from absl.testing import flagsaver
from absl.testing import parameterized

import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import dual_quaternion
from tensorflow_graphics.geometry.transformation import quaternion
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class DualQuaternionTest(test_case.TestCase):

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_conjugate_exception_not_raised(self, *shape):
    self.assert_exception_is_not_raised(dual_quaternion.conjugate, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_conjugate_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.conjugate, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_random(self):
    x_init = test_helpers.generate_random_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    x = tf.convert_to_tensor(value=x_init)
    y = tf.convert_to_tensor(value=x_init)

    x = dual_quaternion.conjugate(x)
    x_real, x_dual = tf.split(x, (4, 4), axis=-1)

    y_real, y_dual = tf.split(y, (4, 4), axis=-1)
    xyz_y_real, w_y_real = tf.split(y_real, (3, 1), axis=-1)
    xyz_y_dual, w_y_dual = tf.split(y_dual, (3, 1), axis=-1)
    y_real = tf.concat((-xyz_y_real, w_y_real), axis=-1)
    y_dual = tf.concat((-xyz_y_dual, w_y_dual), axis=-1)

    self.assertAllEqual(x_real, y_real)
    self.assertAllEqual(x_dual, y_dual)

  @parameterized.parameters(
      ((8,), (8,)),
      ((None, 8), (None, 8)),
  )
  def test_multiply_exception_not_raised(self, *shapes):
    self.assert_exception_is_not_raised(dual_quaternion.multiply, shapes)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (5,), (6,)),
      ("must have exactly 8 dimensions", (7,), (8,)),
  )
  def test_multiply_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.multiply, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_multiply_jacobian_preset(self):
    x_1_init = test_helpers.generate_preset_test_dual_quaternions()
    x_2_init = test_helpers.generate_preset_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.multiply,
                                       [x_1_init, x_2_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_multiply_jacobian_random(self):
    x_1_init = test_helpers.generate_random_test_dual_quaternions()
    x_2_init = test_helpers.generate_random_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.multiply,
                                       [x_1_init, x_2_init])

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_inverse_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_not_raised(dual_quaternion.inverse, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_inverse_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(dual_quaternion.inverse, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_preset(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_preset_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.inverse, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_inverse_jacobian_random(self):
    """Test the Jacobian of the inverse function."""
    x_init = test_helpers.generate_random_test_dual_quaternions()

    self.assert_jacobian_is_correct_fn(dual_quaternion.inverse, [x_init])

  def test_inverse_random(self):
    """Tests that multiplying with the inverse gives identity."""
    rand_dual_quaternion = test_helpers.generate_random_test_dual_quaternions()

    inverse_dual_quaternion = dual_quaternion.inverse(rand_dual_quaternion)
    final_dual_quaternion = dual_quaternion.multiply(rand_dual_quaternion,
                                                     inverse_dual_quaternion)
    tensor_shape = rand_dual_quaternion.shape[:-1]
    identity_dual_quaternion = np.array(
        (0.0, 0.0, 0.0, 1.0, 0.0, 0.0, 0.0, 0.0), dtype=np.float32)
    identity_dual_quaternion = np.tile(identity_dual_quaternion,
                                       tensor_shape + (1,))

    self.assertAllClose(
        final_dual_quaternion, identity_dual_quaternion, rtol=1e-3)

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_norm_exception_not_raised(self, *shape):
    self.assert_exception_is_not_raised(dual_quaternion.norm, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_norm_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.norm, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_norm_jacobian_preset(self):
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.norm, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_norm_jacobian_random(self):
    x_init = test_helpers.generate_random_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.norm, [x_init])

  def test_norm_correct_random(self):
    rand_dual_quaternion = test_helpers.generate_random_test_dual_quaternions()
    norms = dual_quaternion.norm(rand_dual_quaternion)
    tensor_shape = rand_dual_quaternion.shape[:-1]
    norms_gt = np.array((1.0, 0.0), dtype=np.float32)
    norms_gt = np.tile(norms_gt, tensor_shape + (1,))

    self.assertAllClose(norms, norms_gt, rtol=1e-3)

  def test_norm_correct_preset(self):
    pre_dual_quaternion = test_helpers.generate_preset_test_dual_quaternions()
    norms = dual_quaternion.norm(pre_dual_quaternion)
    tensor_shape = pre_dual_quaternion.shape[:-1]
    norms_gt = np.array((1.0, 0.0), dtype=np.float32)
    norms_gt = np.tile(norms_gt, tensor_shape + (1,))

    self.assertAllClose(norms, norms_gt, rtol=1e-3)

  def test_norm_correct_preset_non_unit(self):
    pre_dual_quaternion = test_helpers.generate_preset_test_dual_quaternions()
    pre_dual_quaternion = tf.concat(
        (pre_dual_quaternion[..., :4], pre_dual_quaternion[..., :4],), -1)
    norms = dual_quaternion.norm(pre_dual_quaternion)
    tensor_shape = pre_dual_quaternion.shape[:-1]
    norms_gt = np.array((1.0, 1.0), dtype=np.float32)
    norms_gt = np.tile(norms_gt, tensor_shape + (1,))

    self.assertAllClose(norms, norms_gt, rtol=1e-3)

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_is_normalized_exception_not_raised(self, *shape):
    self.assert_exception_is_not_raised(dual_quaternion.is_normalized, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (1, 5)),)
  def test_is_normalized_exception_raised(self, error_msg, *shape):
    self.assert_exception_is_raised(dual_quaternion.is_normalized,
                                    error_msg,
                                    shape)

  def test_is_normalized_random(self):
    rnd_dual_quaternion = test_helpers.generate_random_test_dual_quaternions()
    tensor_shape = rnd_dual_quaternion.shape[:-1]

    unnormalized_rnd_dual_quaternion = rnd_dual_quaternion * 1.01
    rnd_dual_quaternion = tf.convert_to_tensor(rnd_dual_quaternion)
    unnormalized_rnd_dual_quaternion = tf.convert_to_tensor(
        unnormalized_rnd_dual_quaternion)
    dual_quaternions = tf.concat(
        (rnd_dual_quaternion, unnormalized_rnd_dual_quaternion), axis=0)
    mask = tf.concat(
        (tf.ones(shape=tensor_shape + (1,), dtype=bool),
         tf.zeros(shape=tensor_shape + (1,), dtype=bool)),
        axis=0)
    is_normalized = dual_quaternion.is_normalized(dual_quaternions)

    self.assertAllEqual(mask, is_normalized)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_rotation_translation_jacobian_random(self):
    (euler_angles_init, translation_init
    ) = test_helpers.generate_random_test_euler_angles_translations()
    rotation_init = rotation_matrix_3d.from_quaternion(
        quaternion.from_euler(euler_angles_init))

    self.assert_jacobian_is_finite_fn(dual_quaternion.from_rotation_translation,
                                      [rotation_init, translation_init])

  def test_from_rotation_matrix_normalized_random(self):
    (euler_angles, translation
    ) = test_helpers.generate_random_test_euler_angles_translations()
    rotation = rotation_matrix_3d.from_quaternion(
        quaternion.from_euler(euler_angles))

    random_dual_quaternion = dual_quaternion.from_rotation_translation(
        rotation, translation)

    self.assertAllEqual(
        dual_quaternion.is_normalized(random_dual_quaternion),
        np.ones(shape=rotation.shape[:-2] + (1,), dtype=bool))

  def test_from_rotation_matrix_random(self):
    (euler_angles_gt, translation_gt
    ) = test_helpers.generate_random_test_euler_angles_translations()
    rotation_gt = rotation_matrix_3d.from_quaternion(
        quaternion.from_euler(euler_angles_gt))

    dual_quaternion_output = dual_quaternion.from_rotation_translation(
        rotation_gt, translation_gt)
    dual_quaternion_real = dual_quaternion_output[..., 0:4]
    dual_quaternion_dual = dual_quaternion_output[..., 4:8]
    rotation = rotation_matrix_3d.from_quaternion(dual_quaternion_real)
    translation = 2.0 * quaternion.multiply(
        dual_quaternion_dual, quaternion.inverse(dual_quaternion_real))
    translation = translation[..., 0:3]

    self.assertAllClose(rotation_gt, rotation)
    self.assertAllClose(translation_gt, translation)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_to_rotation_translation_jacobian_preset(self):
    pre_dual_quaternion = test_helpers.generate_preset_test_dual_quaternions()

    def to_rotation(input_dual_quaternion):
      rotation, _ = dual_quaternion.to_rotation_translation(
          input_dual_quaternion)
      return rotation

    self.assert_jacobian_is_finite_fn(to_rotation, [pre_dual_quaternion])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_to_rotation_translation_jacobian_random(self):
    rnd_dual_quaternion = test_helpers.generate_random_test_dual_quaternions()

    def to_translation(input_dual_quaternion):
      _, translation = dual_quaternion.to_rotation_translation(
          input_dual_quaternion)
      return translation

    self.assert_jacobian_is_finite_fn(to_translation, [rnd_dual_quaternion])

  def test_to_rotation_matrix_random(self):
    (euler_angles_gt, translation_gt
    ) = test_helpers.generate_random_test_euler_angles_translations()
    rotation_gt = rotation_matrix_3d.from_quaternion(
        quaternion.from_euler(euler_angles_gt))

    dual_quaternion_output = dual_quaternion.from_rotation_translation(
        rotation_gt, translation_gt)
    rotation, translation = dual_quaternion.to_rotation_translation(
        dual_quaternion_output)

    self.assertAllClose(rotation_gt,
                        rotation_matrix_3d.from_quaternion(rotation))
    self.assertAllClose(translation_gt, translation)


if __name__ == "__main__":
  test_case.main()
