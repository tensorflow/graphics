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
"""Tests for vector."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation.tests import test_data as td
from tensorflow_graphics.math import vector
from tensorflow_graphics.util import test_case


class VectorTest(test_case.TestCase):

  @parameterized.parameters(
      ((None, 3), (None, 3)),)
  def test_cross_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(vector.cross, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis", (1,), (3,)),
      ("must have exactly 3 dimensions in axis", (3,), (2,)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3)),
  )
  def test_cross_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(vector.cross, error_msg, shapes)

  @parameterized.parameters(
      (td.AXIS_3D_0, td.AXIS_3D_0),
      (td.AXIS_3D_0, td.AXIS_3D_X),
      (td.AXIS_3D_0, td.AXIS_3D_Y),
      (td.AXIS_3D_0, td.AXIS_3D_Z),
      (td.AXIS_3D_X, td.AXIS_3D_X),
      (td.AXIS_3D_X, td.AXIS_3D_Y),
      (td.AXIS_3D_X, td.AXIS_3D_Z),
      (td.AXIS_3D_Y, td.AXIS_3D_X),
      (td.AXIS_3D_Y, td.AXIS_3D_Y),
      (td.AXIS_3D_Y, td.AXIS_3D_Z),
      (td.AXIS_3D_Z, td.AXIS_3D_X),
      (td.AXIS_3D_Z, td.AXIS_3D_Y),
      (td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_cross_jacobian_preset(self, u_init, v_init):
    """Tests the Jacobian of the dot product."""
    self.assert_jacobian_is_correct_fn(vector.cross, [u_init, v_init])

  def test_cross_jacobian_random(self):
    """Test the Jacobian of the dot product."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    u_init = np.random.random(size=tensor_shape + [3])
    v_init = np.random.random(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(vector.cross, [u_init, v_init])

  @parameterized.parameters(
      ((td.AXIS_3D_0, td.AXIS_3D_0), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_X), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Y), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Z), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_X, td.AXIS_3D_X), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Y), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Z), (-td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_X), (-td.AXIS_3D_Z,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Y), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Z), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_X), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Y), (-td.AXIS_3D_X,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Z), (td.AXIS_3D_0,)),
  )
  def test_cross_preset(self, test_inputs, test_outputs):
    """Tests the cross product of predefined axes."""
    self.assert_output_is_correct(vector.cross, test_inputs, test_outputs)

  def test_cross_random(self):
    """Tests the cross product function."""
    tensor_size = np.random.randint(1, 4)
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    axis = np.random.randint(tensor_size)
    tensor_shape[axis] = 3  # pylint: disable=invalid-sequence-index
    u = np.random.random(size=tensor_shape)
    v = np.random.random(size=tensor_shape)

    self.assertAllClose(
        vector.cross(u, v, axis=axis), np.cross(u, v, axis=axis))

  @parameterized.parameters(
      ((None,), (None,)),
      ((None, None), (None, None)),
  )
  def test_dot_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(vector.dot, shapes)

  @parameterized.parameters(
      ("must have the same number of dimensions", (None, 1), (None, 2)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 3), (3, 3)),
  )
  def test_dot_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(vector.dot, error_msg, shapes)

  @parameterized.parameters(
      (td.AXIS_3D_0, td.AXIS_3D_0),
      (td.AXIS_3D_0, td.AXIS_3D_X),
      (td.AXIS_3D_0, td.AXIS_3D_Y),
      (td.AXIS_3D_0, td.AXIS_3D_Z),
      (td.AXIS_3D_X, td.AXIS_3D_X),
      (td.AXIS_3D_X, td.AXIS_3D_Y),
      (td.AXIS_3D_X, td.AXIS_3D_Z),
      (td.AXIS_3D_Y, td.AXIS_3D_X),
      (td.AXIS_3D_Y, td.AXIS_3D_Y),
      (td.AXIS_3D_Y, td.AXIS_3D_Z),
      (td.AXIS_3D_Z, td.AXIS_3D_X),
      (td.AXIS_3D_Z, td.AXIS_3D_Y),
      (td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  def test_dot_jacobian_preset(self, u_init, v_init):
    """Tests the Jacobian of the dot product."""
    self.assert_jacobian_is_correct_fn(vector.dot, [u_init, v_init])

  def test_dot_jacobian_random(self):
    """Tests the Jacobian of the dot product."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    u_init = np.random.random(size=tensor_shape + [3])
    v_init = np.random.random(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(vector.dot, [u_init, v_init])

  @parameterized.parameters(
      ((td.AXIS_3D_0, td.AXIS_3D_0), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_X), (1.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Y), (1.,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Z), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_X), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Y), (0.,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Z), (1.,)),
  )
  def test_dot_preset(self, test_inputs, test_outputs):
    """Tests the dot product of predefined axes."""

    def func(u, v):
      return tf.squeeze(vector.dot(u, v), axis=-1)

    self.assert_output_is_correct(func, test_inputs, test_outputs)

  def test_dot_random(self):
    """Tests the dot product function."""
    tensor_size = np.random.randint(2, 4)
    tensor_shape = np.random.randint(1, 10, size=tensor_size).tolist()
    axis = np.random.randint(tensor_size)
    u = np.random.random(size=tensor_shape)
    v = np.random.random(size=tensor_shape)

    dot = tf.linalg.tensor_diag_part(tf.tensordot(u, v, axes=[[axis], [axis]]))
    dot = tf.expand_dims(dot, axis=axis)

    self.assertAllClose(vector.dot(u, v, axis=axis), dot)

  @parameterized.parameters(
      ((None,), (None,)),
      ((None, None), (None, None)),
      ((1,), (1,)),
      ((1, 1), (1, 1)),
  )
  def test_reflect_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(vector.reflect, shapes)

  @parameterized.parameters(
      ("must have the same number of dimensions", (None, 1), (None, 2)),
      ("Not all batch dimensions are broadcast-compatible.", (2, 2), (3, 2)),
  )
  def test_reflect_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(vector.reflect, error_msg, shapes)

  @parameterized.parameters(
      (td.AXIS_3D_0, td.AXIS_3D_0),
      (td.AXIS_3D_0, td.AXIS_3D_X),
      (td.AXIS_3D_0, td.AXIS_3D_Y),
      (td.AXIS_3D_0, td.AXIS_3D_Z),
      (td.AXIS_3D_X, td.AXIS_3D_X),
      (td.AXIS_3D_X, td.AXIS_3D_Y),
      (td.AXIS_3D_X, td.AXIS_3D_Z),
      (td.AXIS_3D_Y, td.AXIS_3D_X),
      (td.AXIS_3D_Y, td.AXIS_3D_Y),
      (td.AXIS_3D_Y, td.AXIS_3D_Z),
      (td.AXIS_3D_Z, td.AXIS_3D_X),
      (td.AXIS_3D_Z, td.AXIS_3D_Y),
      (td.AXIS_3D_Z, td.AXIS_3D_Z),
  )
  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_reflect_jacobian_preset(self, u_init, v_init):
    """Tests the Jacobian of the reflect function."""
    self.assert_jacobian_is_correct_fn(vector.reflect, [u_init, v_init])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_reflect_jacobian_random(self):
    """Tests the Jacobian of the reflect function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    u_init = np.random.random(size=tensor_shape + [3])
    v_init = np.random.random(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(vector.reflect, [u_init, v_init])

  @parameterized.parameters(
      ((td.AXIS_3D_0, td.AXIS_3D_X), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Y), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_0, td.AXIS_3D_Z), (td.AXIS_3D_0,)),
      ((td.AXIS_3D_X, td.AXIS_3D_X), (-td.AXIS_3D_X,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Y), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_X, td.AXIS_3D_Z), (td.AXIS_3D_X,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_X), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Y), (-td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Y, td.AXIS_3D_Z), (td.AXIS_3D_Y,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_X), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Y), (td.AXIS_3D_Z,)),
      ((td.AXIS_3D_Z, td.AXIS_3D_Z), (-td.AXIS_3D_Z,)),
  )
  def test_reflect_preset(self, test_inputs, test_outputs):
    """Tests the reflect function of predefined axes."""
    self.assert_output_is_correct(vector.reflect, test_inputs, test_outputs)

  def test_reflect_random(self):
    """Tests that calling reflect twice give an identity transform."""
    tensor_size = np.random.randint(2, 4)
    tensor_shape = np.random.randint(2, 3, size=tensor_size).tolist()
    axis = np.random.randint(tensor_size)
    u = np.random.random(size=tensor_shape)
    v = np.random.random(size=tensor_shape)
    v /= np.linalg.norm(v, axis=axis, keepdims=True)

    u_new = vector.reflect(u, v, axis=axis)
    u_new = vector.reflect(u_new, v, axis=axis)

    self.assertAllClose(u_new, u)


if __name__ == "__main__":
  test_case.main()
