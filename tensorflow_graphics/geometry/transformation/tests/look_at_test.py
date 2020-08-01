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
"""Tests for OpenGL lookAt functions."""

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.util import test_case


class LookAtTest(test_case.TestCase):

  def test_look_at_right_handed_preset(self):
    """Tests that look_at_right_handed generates expected results."""
    camera_position = ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    look_at_point = ((0.0, 0.0, 1.0), (0.4, 0.5, 0.6))
    up_vector = ((0.0, 1.0, 0.0), (0.7, 0.8, 0.9))

    pred = look_at.right_handed(camera_position, look_at_point, up_vector)

    gt = (((-1.0, 0.0, 0.0, 0.0), (0.0, 1.0, 0.0, 0.0), (0.0, 0.0, -1.0, 0.0),
           (0.0, 0.0, 0.0, 1.0)),
          ((4.08248186e-01, -8.16496551e-01, 4.08248395e-01, -2.98023224e-08),
           (-7.07106888e-01, 1.19209290e-07, 7.07106769e-01, -1.41421378e-01),
           (-5.77350318e-01, -5.77350318e-01, -5.77350318e-01,
            3.46410215e-01), (0.0, 0.0, 0.0, 1.0)))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((3,), (3,), (3,)),
      ((None, 3), (None, 3), (None, 3)),
      ((None, 2, 3), (None, 2, 3), (None, 2, 3)),
  )
  def test_look_at_right_handed_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(look_at.right_handed, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (1,)),
      ("Not all batch dimensions are identical", (3,), (3, 3), (3, 3)),
  )
  def test_look_at_right_handed_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(look_at.right_handed, error_msg, shapes)

  def test_look_at_right_handed_jacobian_preset(self):
    """Tests the Jacobian of look_at_right_handed."""
    camera_position_init = np.array(((0.0, 0.0, 0.0), (0.1, 0.2, 0.3)))
    look_at_init = np.array(((0.0, 0.0, 1.0), (0.4, 0.5, 0.6)))
    up_vector_init = np.array(((0.0, 1.0, 0.0), (0.7, 0.8, 0.9)))

    self.assert_jacobian_is_correct_fn(
        look_at.right_handed,
        [camera_position_init, look_at_init, up_vector_init])

  def test_look_at_right_handed_jacobian_random(self):
    """Tests the Jacobian of look_at_right_handed."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    camera_position_init = np.random.uniform(size=tensor_shape + [3])
    look_at_init = np.random.uniform(size=tensor_shape + [3])
    up_vector_init = np.random.uniform(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(
        look_at.right_handed,
        [camera_position_init, look_at_init, up_vector_init])
