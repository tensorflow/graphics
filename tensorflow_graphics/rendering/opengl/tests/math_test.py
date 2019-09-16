#Copyright 2018 Google LLC
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
"""Tests for OpenGL math routines."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import math

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.util import test_case


class MathTest(test_case.TestCase):

  def test_perspective_right_handed_preset(self):
    """Tests that perspective_right_handed generates expected results.."""
    vertical_field_of_view = (60.0 * math.pi / 180.0, 50.0 * math.pi / 180.0)
    aspect_ratio = (1.5, 1.1)
    near = (1.0, 1.2)
    far = (10.0, 5.0)

    pred = glm.perspective_right_handed(vertical_field_of_view, aspect_ratio,
                                        near, far)
    gt = (((1.15470052, 0.0, 0.0, 0.0), (0.0, 1.73205066, 0.0, 0.0),
           (0.0, 0.0, -1.22222221, -2.22222233), (0.0, 0.0, -1.0, 0.0)),
          ((1.9495517, 0.0, 0.0, 0.0), (0.0, 2.14450693, 0.0, 0.0),
           (0.0, 0.0, -1.63157892, -3.15789485), (0.0, 0.0, -1.0, 0.0)))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((1,), (1,), (1,), (1,)),
      ((None, 2), (None, 2), (None, 2), (None, 2)),
  )
  def test_perspective_right_handed_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.perspective_right_handed, shapes)

  @parameterized.parameters(
      ("Not all batch dimensions are identical", (3,), (3, 3), (3, 3), (3, 3)),
      ("Not all batch dimensions are identical", (2, 3), (3, 3), (3, 3),
       (3, 3)),
  )
  def test_perspective_right_handed_shape_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.perspective_right_handed, error_msg,
                                    shapes)

  @parameterized.parameters(
      ((1.0,),
       (1.0,), np.random.uniform(-1.0, 0.0, size=(1,)).astype(np.float32),
       (1.0,)),
      ((1.0,), (1.0,), (0.0,), (1.0,)),
      ((1.0,), np.random.uniform(-1.0, 0.0, size=(1,)).astype(np.float32),
       (0.1,), (1.0,)),
      ((1.0,), (0.0,), (0.1,), (1.0,)),
      ((1.0,),
       (1.0,), np.random.uniform(1.0, 2.0, size=(1,)).astype(np.float32),
       np.random.uniform(0.1, 0.5, size=(1,)).astype(np.float32)),
      ((1.0,), (1.0,), (0.1,), (0.1,)),
      (np.random.uniform(-math.pi, 0.0, size=(1,)).astype(np.float32), (1.0,),
       (0.1,), (1.0,)),
      (np.random.uniform(math.pi, 2.0 * math.pi, size=(1,)).astype(np.float32),
       (1.0,), (0.1,), (1.0,)),
      ((0.0,), (1.0,), (0.1,), (1.0,)),
      ((math.pi,), (1.0,), (0.1,), (1.0,)),
  )
  def test_perspective_right_handed_valid_range_exception_raised(
      self, vertical_field_of_view, aspect_ratio, near, far):
    """Tests that an exception is raised with out of bounds values."""
    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          glm.perspective_right_handed(vertical_field_of_view, aspect_ratio,
                                       near, far))

  def test_perspective_right_handed_cross_jacobian_preset(self):
    """Tests the Jacobian of perspective_right_handed."""
    vertical_field_of_view_init = np.array((1.0,))
    aspect_ratio_init = np.array((1.0,))
    near_init = np.array((1.0,))
    far_init = np.array((10.0,))

    # Wrap with tf.identity because some assert_* ops look at the constant
    # tensor value and mark it as unfeedable.
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    aspect_ratio_tensor = tf.identity(
        tf.convert_to_tensor(value=aspect_ratio_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y = glm.perspective_right_handed(vertical_field_of_view_tensor,
                                     aspect_ratio_tensor, near_tensor,
                                     far_tensor)

    self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                    vertical_field_of_view_init, y)
    self.assert_jacobian_is_correct(aspect_ratio_tensor, aspect_ratio_init, y)
    self.assert_jacobian_is_correct(near_tensor, near_init, y)
    self.assert_jacobian_is_correct(far_tensor, far_init, y)

  def test_perspective_right_handed_cross_jacobian_random(self):
    """Tests the Jacobian of perspective_right_handed."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    eps = np.finfo(np.float64).eps
    vertical_field_of_view_init = np.random.uniform(
        eps, math.pi - eps, size=tensor_shape)
    aspect_ratio_init = np.random.uniform(eps, 100.0, size=tensor_shape)
    near_init = np.random.uniform(eps, 10.0, size=tensor_shape)
    far_init = np.random.uniform(10 + eps, 100.0, size=tensor_shape)

    # Wrap with tf.identity because some assert_* ops look at the constant
    # tensor value and mark it as unfeedable.
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    aspect_ratio_tensor = tf.identity(
        tf.convert_to_tensor(value=aspect_ratio_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y = glm.perspective_right_handed(vertical_field_of_view_tensor,
                                     aspect_ratio_tensor, near_tensor,
                                     far_tensor)

    self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                    vertical_field_of_view_init, y)
    self.assert_jacobian_is_correct(aspect_ratio_tensor, aspect_ratio_init, y)
    self.assert_jacobian_is_correct(near_tensor, near_init, y)
    self.assert_jacobian_is_correct(far_tensor, far_init, y)

  def test_look_at_right_handed_preset(self):
    """Tests that look_at_right_handed generates expected results.."""
    camera_position = ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    look_at = ((0.0, 0.0, 1.0), (0.4, 0.5, 0.6))
    up_vector = ((0.0, 1.0, 0.0), (0.7, 0.8, 0.9))

    pred = glm.look_at_right_handed(camera_position, look_at, up_vector)
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
    self.assert_exception_is_not_raised(glm.look_at_right_handed, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (1,)),
      ("Not all batch dimensions are identical", (3,), (3, 3), (3, 3)),
  )
  def test_look_at_right_handed_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.look_at_right_handed, error_msg, shapes)

  def test_look_at_right_handed_jacobian_preset(self):
    """Tests the Jacobian of look_at_right_handed."""
    camera_position_init = np.array(((0.0, 0.0, 0.0), (0.1, 0.2, 0.3)))
    look_at_init = np.array(((0.0, 0.0, 1.0), (0.4, 0.5, 0.6)))
    up_vector_init = np.array(((0.0, 1.0, 0.0), (0.7, 0.8, 0.9)))
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    y = glm.look_at_right_handed(camera_position_tensor, look_at_tensor,
                                 up_vector_tensor)

    self.assert_jacobian_is_correct(camera_position_tensor,
                                    camera_position_init, y)
    self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)

  def test_look_at_right_handed_jacobian_random(self):
    """Tests the Jacobian of look_at_right_handed."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    camera_position_init = np.random.uniform(size=tensor_shape + [3])
    look_at_init = np.random.uniform(size=tensor_shape + [3])
    up_vector_init = np.random.uniform(size=tensor_shape + [3])
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    y = glm.look_at_right_handed(camera_position_tensor, look_at_tensor,
                                 up_vector_tensor)

    self.assert_jacobian_is_correct(camera_position_tensor,
                                    camera_position_init, y)
    self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)

  def test_model_to_eye_preset(self):
    """Tests that model_to_eye generates expected results.."""
    point = ((2.0, 3.0, 4.0), (3.0, 4.0, 5.0))
    camera_position = ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    look_at = ((0.0, 0.0, 1.0), (0.4, 0.5, 0.6))
    up_vector = ((0.0, 1.0, 0.0), (0.7, 0.8, 0.9))

    pred = glm.model_to_eye(point, camera_position, look_at, up_vector)
    gt = ((-2.0, 3.0, -4.0), (2.08616257e-07, 1.27279234, -6.58179379))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((3,), (3,), (3,), (3,)),
      ((None, 3), (None, 3), (None, 3), (None, 3)),
      ((3,), (None, 3), (None, 3), (None, 3)),
      ((None, 2, 3), (None, 2, 3), (None, 2, 3), (None, 2, 3)),
  )
  def test_model_to_eye_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.model_to_eye, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (3,), (2,)),
      ("Not all batch dimensions are broadcast-compatible", (3,), (2, 3),
       (3, 3), (3, 3)),
  )
  def test_model_to_eye_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.model_to_eye, error_msg, shapes)

  def test_model_to_eye_jacobian_preset(self):
    """Tests the Jacobian of model_to_eye."""
    point_init = np.array(((2.0, 3.0, 4.0), (3.0, 4.0, 5.0)))
    camera_position_init = np.array(((0.0, 0.0, 0.0), (0.1, 0.2, 0.3)))
    look_at_init = np.array(((0.0, 0.0, 1.0), (0.4, 0.5, 0.6)))
    up_vector_init = np.array(((0.0, 1.0, 0.0), (0.7, 0.8, 0.9)))
    point_tensor = tf.convert_to_tensor(value=point_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    y = glm.model_to_eye(point_tensor, camera_position_tensor, look_at_tensor,
                         up_vector_tensor)

    self.assert_jacobian_is_correct(point_tensor, point_init, y)
    self.assert_jacobian_is_correct(camera_position_tensor,
                                    camera_position_init, y)
    self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)

  def test_model_to_eye_jacobian_random(self):
    """Tests the Jacobian of model_to_eye."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [3])
    camera_position_init = np.random.uniform(size=tensor_shape + [3])
    look_at_init = np.random.uniform(size=tensor_shape + [3])
    up_vector_init = np.random.uniform(size=tensor_shape + [3])
    point_tensor = tf.convert_to_tensor(value=point_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    y = glm.model_to_eye(point_tensor, camera_position_tensor, look_at_tensor,
                         up_vector_tensor)

    self.assert_jacobian_is_correct(point_tensor, point_init, y)
    self.assert_jacobian_is_correct(camera_position_tensor,
                                    camera_position_init, y)
    self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)

  def test_eye_to_clip_preset(self):
    """Tests that eye_to_clip generates expected results."""
    point = ((2.0, 3.0, 4.0), (3.0, 4.0, 5.0))
    fov = ((60.0 * math.pi / 180.0,), (50.0 * math.pi / 180.0,))
    aspect_ratio = ((1.5,), (1.6,))
    near_plane = ((1.0,), (2.0,))
    far_plane = ((10.0,), (11.0,))

    pred = glm.eye_to_clip(point, fov, aspect_ratio, near_plane, far_plane)
    gt = ((2.30940104, 5.19615173, -7.11111116, -4.0), (4.02095032, 8.57802773,
                                                        -12.11111069, -5.0))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((3,), (1,), (1,), (1,), (1,)),
      ((None, 3), (None, 1), (None, 1), (None, 1), (None, 1)),
      ((None, 5, 3), (None, 5, 1), (None, 5, 1), (None, 5, 1), (None, 5, 1)),
  )
  def test_eye_to_clip_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.eye_to_clip, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (1,), (1,), (1,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (2,), (1,), (1,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (1,), (2,), (1,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (1,), (1,), (2,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (1,), (1,), (1,),
       (2,)),
      ("Not all batch dimensions are broadcast-compatible", (3, 3), (2, 1),
       (1,), (1,), (1,)),
  )
  def test_eye_to_clip_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.eye_to_clip, error_msg, shapes)

  def test_eye_to_clip_jacobian_preset(self):
    """Tests the Jacobian of eye_to_clip."""
    point_init = np.array(((2.0, 3.0, 4.0), (3.0, 4.0, 5.0)))
    vertical_field_of_view_init = np.array(
        ((60.0 * math.pi / 180.0,), (50.0 * math.pi / 180.0,)))
    aspect_ratio_init = np.array(((1.5,), (1.6,)))
    near_init = np.array(((1.0,), (2.0,)))
    far_init = np.array(((10.0,), (11.0,)))

    point_tensor = tf.convert_to_tensor(value=point_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    aspect_ratio_tensor = tf.identity(
        tf.convert_to_tensor(value=aspect_ratio_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))
    y = glm.eye_to_clip(point_tensor, vertical_field_of_view_tensor,
                        aspect_ratio_tensor, near_tensor, far_tensor)

    self.assert_jacobian_is_correct(point_tensor, point_init, y)
    self.assert_jacobian_is_correct(
        vertical_field_of_view_tensor,
        vertical_field_of_view_init,
        y,
        atol=1e-5)
    self.assert_jacobian_is_correct(
        aspect_ratio_tensor, aspect_ratio_init, y, atol=1e-5)
    self.assert_jacobian_is_correct(near_tensor, near_init, y, atol=1e-5)
    self.assert_jacobian_is_correct(far_tensor, far_init, y, atol=1e-5)

  def test_eye_to_clip_jacobian_random(self):
    """Tests the Jacobian of eye_to_clip."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [3])
    eps = np.finfo(np.float64).eps
    vertical_field_of_view_init = np.random.uniform(
        eps, math.pi - eps, size=tensor_shape + [1])
    aspect_ratio_init = np.random.uniform(eps, 100.0, size=tensor_shape + [1])
    near_init = np.random.uniform(eps, 100.0, size=tensor_shape + [1])
    far_init = near_init + np.random.uniform(eps, 10.0, size=tensor_shape + [1])

    point_tensor = tf.convert_to_tensor(value=point_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    aspect_ratio_tensor = tf.identity(
        tf.convert_to_tensor(value=aspect_ratio_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y = glm.eye_to_clip(point_tensor, vertical_field_of_view_tensor,
                        aspect_ratio_tensor, near_tensor, far_tensor)

    self.assert_jacobian_is_correct(point_tensor, point_init, y, atol=5e-06)
    self.assert_jacobian_is_correct(
        vertical_field_of_view_tensor,
        vertical_field_of_view_init,
        y,
        atol=5e-06)
    self.assert_jacobian_is_correct(
        aspect_ratio_tensor, aspect_ratio_init, y, atol=5e-06)
    self.assert_jacobian_is_correct(near_tensor, near_init, y, atol=5e-06)
    self.assert_jacobian_is_correct(far_tensor, far_init, y, atol=5e-06)


if __name__ == "__main__":
  test_case.main()
