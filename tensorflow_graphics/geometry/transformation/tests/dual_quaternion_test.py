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
import tensorflow.compat.v2 as tf

from tensorflow_graphics.geometry.transformation import dual_quaternion
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class DualQuaternionTest(test_case.TestCase):

  @parameterized.parameters(
      ((8,),),
      ((None, 8),),
  )
  def test_conjugate_exception_not_raised(self, *shape):
    """Tests that the shape exceptions of conjugate are not raised."""
    self.assert_exception_is_not_raised(dual_quaternion.conjugate, shape)

  @parameterized.parameters(
      ("must have exactly 8 dimensions", (3,)),)
  def test_conjugate_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are raised."""
    self.assert_exception_is_raised(dual_quaternion.conjugate, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_preset(self):
    """Tests the Jacobian of the conjugate function."""
    x_init = test_helpers.generate_preset_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_jacobian_random(self):
    """Tests the Jacobian of the conjugate function."""
    x_init = test_helpers.generate_random_test_dual_quaternions()
    self.assert_jacobian_is_correct_fn(dual_quaternion.conjugate, [x_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_conjugate_preset(self):
    """Tests if the conjugate function is providing correct results."""
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
