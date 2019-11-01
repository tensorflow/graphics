#Copyright 2019 Google LLC
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
    vertical_field_of_view = ((60.0 * math.pi / 180.0,),
                              (50.0 * math.pi / 180.0,))
    aspect_ratio = ((1.5,), (1.1,))
    near = ((1.0,), (1.2,))
    far = ((10.0,), (5.0,))

    pred = glm.perspective_right_handed(vertical_field_of_view, aspect_ratio,
                                        near, far)
    gt = (((1.15470052, 0.0, 0.0, 0.0), (0.0, 1.73205066, 0.0, 0.0),
           (0.0, 0.0, -1.22222221, -2.22222233), (0.0, 0.0, -1.0, 0.0)),
          ((1.9495517, 0.0, 0.0, 0.0), (0.0, 2.14450693, 0.0, 0.0),
           (0.0, 0.0, -1.63157892, -3.15789485), (0.0, 0.0, -1.0, 0.0)))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((1,), (1,), (1,), (1,)),
      ((None, 1), (None, 1), (None, 1), (None, 1)),
      ((None, 3, 1), (None, 3, 1), (None, 3, 1), (None, 3, 1)),
  )
  def test_perspective_right_handed_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.perspective_right_handed, shapes)

  @parameterized.parameters(
      ("Not all batch dimensions are identical", (1,), (3, 1), (3, 1), (3, 1)),
      ("Not all batch dimensions are identical", (3, 1), (None, 3, 1), (3, 1),
       (3, 1)),
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
        eps, math.pi - eps, size=tensor_shape + [1])
    aspect_ratio_init = np.random.uniform(eps, 100.0, size=tensor_shape + [1])
    near_init = np.random.uniform(eps, 10.0, size=tensor_shape + [1])
    far_init = np.random.uniform(10 + eps, 100.0, size=tensor_shape + [1])

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
      ((100, 3), (3,), (3,), (3,)),
      ((None, 1, 3), (None, 2, 3), (None, 2, 3), (None, 2, 3)),
  )
  def test_model_to_eye_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.model_to_eye, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (3,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (2,), (3,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (2,), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3,), (3,), (3,), (2,)),
      ("Not all batch dimensions are identical", (3,), (2, 3), (3, 3), (3, 3)),
      ("Not all batch dimensions are broadcast-compatible", (2, 3), (3, 3),
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
    vertical_field_of_view = ((60.0 * math.pi / 180.0,),
                              (50.0 * math.pi / 180.0,))
    aspect_ratio = ((1.5,), (1.6,))
    near_plane = ((1.0,), (2.0,))
    far_plane = ((10.0,), (11.0,))

    pred = glm.eye_to_clip(point, vertical_field_of_view, aspect_ratio,
                           near_plane, far_plane)
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

  def test_clip_to_ndc_preset(self):
    """Tests that clip_to_ndc generates expected results."""
    point = ((4.0, 8.0, 16.0, 2.0), (4.0, 8.0, 16.0, 1.0))
    pred = glm.clip_to_ndc(point)
    gt = ((2.0, 4.0, 8.0), (4.0, 8.0, 16.0))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((4,)),
      ((None, 4),),
      ((None, 5, 4),),
  )
  def test_clip_to_ndc_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.clip_to_ndc, shapes)

  def test_clip_to_ndc_exception_raised(self):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        glm.clip_to_ndc, "must have exactly 4 dimensions in axis -1", ((2,),))

  def test_clip_to_ndc_jacobian_preset(self):
    """Tests the Jacobian of clip_to_ndc."""
    point_init = np.array(((4.0, 8.0, 16.0, 2.0), (4.0, 8.0, 16.0, 1.0)))
    point_tensor = tf.convert_to_tensor(value=point_init)
    y = glm.clip_to_ndc(point_tensor)
    self.assert_jacobian_is_correct(point_tensor, point_init, y)

  def test_clip_to_ndc_jacobian_random(self):
    """Tests the Jacobian of clip_to_ndc."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [4])
    point_tensor = tf.convert_to_tensor(value=point_init)
    y = glm.clip_to_ndc(point_tensor)
    self.assert_jacobian_is_correct(point_tensor, point_init, y)

  def test_ndc_to_screen_preset(self):
    """Tests that ndc_to_screen generates expected results."""
    point = ((1.1, 2.2, 3.3), (5.1, 5.2, 5.3))
    lower_left_corner = ((6.4, 4.8), (0.0, 0.0))
    screen_dimensions = ((640.0, 480.0), (300.0, 400.0))
    near = ((1.0,), (11.0,))
    far = ((10.0,), (100.0,))
    pred = glm.ndc_to_screen(point, lower_left_corner, screen_dimensions, near,
                             far)
    gt = ((678.40002441, 772.79998779, 20.34999847), (915.0, 1240.0,
                                                      291.3500061))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((3,), (2,), (2,), (1,), (1,)),
      ((None, 3), (None, 2), (None, 2), (None, 1), (None, 1)),
      ((None, 5, 3), (None, 5, 2), (None, 5, 2), (None, 5, 1), (None, 5, 1)),
  )
  def test_ndc_to_screen_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.ndc_to_screen, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (2,), (2,), (2,), (1,),
       (1,)),
      ("must have exactly 2 dimensions in axis -1", (3,), (1,), (2,), (1,),
       (1,)),
      ("must have exactly 2 dimensions in axis -1", (3,), (2,), (3,), (1,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (2,), (2,), (2,),
       (1,)),
      ("must have exactly 1 dimensions in axis -1", (3,), (2,), (2,), (1,),
       (3,)),
      ("Not all batch dimensions are identical", (3,), (2, 2), (3, 2), (3, 1),
       (3, 1)),
      ("Not all batch dimensions are broadcast-compatible", (4, 3), (3, 2),
       (3, 2), (3, 1), (3, 1)),
  )
  def test_ndc_to_screen_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.ndc_to_screen, error_msg, shapes)

  def test_ndc_to_screen_exception_near_raised(self):
    """Tests that an exception is raised when `near` is not strictly positive."""

    point = np.random.uniform(size=(3,))
    lower_left_corner = np.random.uniform(size=(2,))
    screen_dimensions = np.random.uniform(1.0, 2.0, size=(2,))
    near = np.random.uniform(-1.0, 0.0, size=(1,))
    far = np.random.uniform(1.0, 2.0, size=(1,))

    with self.subTest("negative_near"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            glm.ndc_to_screen(point, lower_left_corner, screen_dimensions, near,
                              far))

    with self.subTest("zero_near"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            glm.ndc_to_screen(point, lower_left_corner, screen_dimensions,
                              np.array((0.0,)), far))

  def test_ndc_to_screen_exception_far_raised(self):
    """Tests that an exception is raised if `far` is not greater than `near`."""
    point = np.random.uniform(size=(3,))
    lower_left_corner = np.random.uniform(size=(2,))
    screen_dimensions = np.random.uniform(1.0, 2.0, size=(2,))
    near = np.random.uniform(1.0, 10.0, size=(1,))
    far = near + np.random.uniform(-1.0, 0.0, size=(1,))

    with self.assertRaises(tf.errors.InvalidArgumentError):
      self.evaluate(
          glm.ndc_to_screen(point, lower_left_corner, screen_dimensions, near,
                            far))

  def test_ndc_to_screen_exception_screen_dimensions_raised(self):
    """Tests that an exception is raised when `screen_dimensions` is not strictly positive."""
    point = np.random.uniform(size=(3,))
    lower_left_corner = np.random.uniform(size=(2,))
    screen_dimensions = np.random.uniform(-1.0, 0.0, size=(2,))
    near = np.random.uniform(1.0, 10.0, size=(1,))
    far = near + np.random.uniform(0.1, 1.0, size=(1,))

    with self.subTest("negative_screen_dimensions"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            glm.ndc_to_screen(point, lower_left_corner, screen_dimensions, near,
                              far))

    with self.subTest("zero_screen_dimensions"):
      with self.assertRaises(tf.errors.InvalidArgumentError):
        self.evaluate(
            glm.ndc_to_screen(point, lower_left_corner, np.array((0.0, 0.0)),
                              near, far))

  def test_ndc_to_screen_jacobian_preset(self):
    """Tests the Jacobian of ndc_to_screen."""
    point_init = np.array(((1.1, 2.2, 3.3), (5.1, 5.2, 5.3)))
    lower_left_corner_init = np.array(((6.4, 4.8), (0.0, 0.0)))
    screen_dimensions_init = np.array(((640.0, 480.0), (300.0, 400.0)))
    near_init = np.array(((1.0,), (11.0,)))
    far_init = np.array(((10.0,), (100.0,)))

    point_tensor = tf.convert_to_tensor(value=point_init)
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y = glm.ndc_to_screen(point_tensor, lower_left_corner_tensor,
                          screen_dimensions_tensor, near_tensor, far_tensor)
    self.assert_jacobian_is_correct(point_tensor, point_init, y)
    self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                    lower_left_corner_init, y)
    self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                    screen_dimensions_init, y)
    self.assert_jacobian_is_correct(near_tensor, near_init, y)
    self.assert_jacobian_is_correct(far_tensor, far_init, y)

  def test_ndc_to_screen_jacobian_random(self):
    """Tests the Jacobian of ndc_to_screen."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [3])
    lower_left_corner_init = np.random.uniform(size=tensor_shape + [2])
    screen_dimensions_init = np.random.uniform(
        1.0, 1000.0, size=tensor_shape + [2])
    near_init = np.random.uniform(1.0, 10.0, size=tensor_shape + [1])
    far_init = near_init + np.random.uniform(0.1, 1.0, size=(1,))

    point_tensor = tf.convert_to_tensor(value=point_init)
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y = glm.ndc_to_screen(point_tensor, lower_left_corner_tensor,
                          screen_dimensions_tensor, near_tensor, far_tensor)
    self.assert_jacobian_is_correct(point_tensor, point_init, y)
    self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                    lower_left_corner_init, y)
    self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                    screen_dimensions_init, y)
    self.assert_jacobian_is_correct(near_tensor, near_init, y)
    self.assert_jacobian_is_correct(far_tensor, far_init, y)

  def test_model_to_screen_preset(self):
    """Tests that model_to_screen generates expected results."""
    point_world_space = ((3.1, 4.1, 5.1), (-1.1, 2.2, -3.1))
    camera_position = ((0.0, 0.0, 0.0), (0.4, -0.8, 0.1))
    camera_up = ((0.0, 1.0, 0.0), (0.0, 0.0, 1.0))
    look_at = ((0.0, 0.0, 1.0), (0.0, 1.0, 0.0))
    vertical_field_of_view = ((60.0 * math.pi / 180.0,), (65 * math.pi / 180,))
    lower_left_corner = ((0.0, 0.0), (10.0, 20.0))
    screen_dimensions = ((501.0, 501.0), (400.0, 600.0))
    near = ((0.01,), (1.0,))
    far = ((4.0,), (3.0,))
    pred_screen, pred_w = glm.model_to_screen(point_world_space,
                                              camera_position, look_at,
                                              camera_up, vertical_field_of_view,
                                              screen_dimensions, near, far,
                                              lower_left_corner)
    gt_screen = ((-13.23016357, 599.30444336, 4.00215721),
                 (98.07017517, -95.40383911, 3.1234405))
    gt_w = ((5.1,), (3.42247,))
    self.assertAllClose(pred_screen, gt_screen, atol=1e-5, rtol=1e-5)
    self.assertAllClose(pred_w, gt_w)

  @parameterized.parameters(
      ((3,), (3,), (3,), (3,), (1,), (2,), (1,), (1,), (2,)),
      ((640, 480, 3), (3,), (3,), (3,), (1,), (2,), (1,), (1,), (2,)),
      ((None, 3), (None, 3), (None, 3), (None, 3), (None, 1), (None, 2),
       (None, 1), (None, 1), (None, 2)),
      ((3,), (None, 1, 3), (None, 1, 3), (None, 1, 3), (None, 1, 1),
       (None, 1, 2), (None, 1, 1), (None, 1, 1), (None, 1, 2)),
  )
  def test_model_to_screen_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.model_to_screen, shapes)

  @parameterized.parameters(
      ("point_model_space must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (2,), (3,), (3,), (3,)),
      ("camera_position must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3,), (2,), (3,), (3,)),
      ("look_at must have exactly 3 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0,), (2.0,), (0.0, 0.0), (3,), (3,), (2,), (3,)),
      ("up_vector must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3,), (3,), (3,), (2,)),
      ("vertical_field_of_view must have exactly 1 dimensions in axis -1",
       (1.0, 1.0), (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3,), (3,), (3,),
       (3,)),
      ("screen_dimensions must have exactly 2 dimensions in axis -1", (1.0,),
       (1.0,), (1.0,), (2.0,), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("near must have exactly 1 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0, 1.0), (2.0,), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("far must have exactly 1 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0,), (2.0, 2.0), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("lower_left_corner must have exactly 2 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0,), (3,), (3,), (3,), (3,)),
      ("Not all batch dimensions are broadcast-compatible", ((1.0,), (1.0,)),
       ((1.0, 1.0), (1.0, 1.0)), ((1.0,), (1.0,)), ((2.0,), (2.0,)),
       ((0.0, 0.0), (0.0, 0.0)), (5, 3), (2, 3), (2, 3), (2, 3)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (3,), (2, 3), (3,), (3,)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (3,), (3,), (2, 3), (3,)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (3,), (3,), (3,), (2, 3)),
      ("Not all batch dimensions are identical", ((1.0,),), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("Not all batch dimensions are identical", (1.0,), ((1.0, 1.0),), (1.0,),
       (2.0,), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), ((1.0,),),
       (2.0,), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), (1.0,),
       ((2.0,),), (0.0, 0.0), (3,), (3,), (3,), (3,)),
      ("Not all batch dimensions are identical", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), ((0.0, 0.0),), (3,), (3,), (3,), (3,)),
  )
  def test_model_to_screen_exception_raised(self, error_msg,
                                            vertical_field_of_view,
                                            screen_dimensions, near, far,
                                            lower_left_corner, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        func=glm.model_to_screen,
        error_msg=error_msg,
        shapes=shapes,
        vertical_field_of_view=vertical_field_of_view,
        screen_dimensions=screen_dimensions,
        near=near,
        far=far,
        lower_left_corner=lower_left_corner)

  def test_model_to_screen_jacobian_preset(self):
    """Tests the Jacobian of model_to_screen."""
    point_world_space_init = np.array(((3.1, 4.1, 5.1), (-1.1, 2.2, -3.1)))
    camera_position_init = np.array(((0.0, 0.0, 0.0), (0.4, -0.8, 0.1)))
    camera_up_init = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    look_at_init = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    vertical_field_of_view_init = np.array(
        ((60.0 * math.pi / 180.0,), (65 * math.pi / 180,)))
    lower_left_corner_init = np.array(((0.0, 0.0), (10.0, 20.0)))
    screen_dimensions_init = np.array(((501.0, 501.0), (400.0, 600.0)))
    near_init = np.array(((0.01,), (1.0,)))
    far_init = np.array(((4.0,), (3.0,)))

    point_world_space_tensor = tf.convert_to_tensor(
        value=point_world_space_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    camera_up_tensor = tf.convert_to_tensor(value=camera_up_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y_projection, y_w = glm.model_to_screen(
        point_world_space_tensor, camera_position_tensor, look_at_tensor,
        camera_up_tensor, vertical_field_of_view_tensor,
        screen_dimensions_tensor, near_tensor, far_tensor,
        lower_left_corner_tensor)
    with self.subTest(name="jacobian_point_world_space"):
      self.assert_jacobian_is_correct(point_world_space_tensor,
                                      point_world_space_init, y_projection)
      self.assert_jacobian_is_correct(point_world_space_tensor,
                                      point_world_space_init, y_w)
    with self.subTest(name="jacobian_camera_position"):
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y_projection)
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y_w)
    with self.subTest(name="jacobian_look_at"):
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init,
                                      y_projection)
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y_w)
    with self.subTest(name="jacobian_camera_up"):
      self.assert_jacobian_is_correct(camera_up_tensor, camera_up_init,
                                      y_projection)
      self.assert_jacobian_is_correct(camera_up_tensor, camera_up_init, y_w)
    with self.subTest(name="jacobian_vertical_field_of_view"):
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y_projection)
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y_w)
    with self.subTest(name="jacobian_screen_dimensions"):
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y_projection)
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y_w)
    with self.subTest(name="jacobian_near"):
      self.assert_jacobian_is_correct(near_tensor, near_init, y_projection)
      self.assert_jacobian_is_correct(near_tensor, near_init, y_w)
    with self.subTest(name="jacobian_far"):
      self.assert_jacobian_is_correct(far_tensor, far_init, y_projection)
      self.assert_jacobian_is_correct(far_tensor, far_init, y_w)
    with self.subTest(name="jacobian_lower_left_corner"):
      self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                      lower_left_corner_init, y_projection)

  def test_model_to_screen_jacobian_random(self):
    """Tests the Jacobian of model_to_screen."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_world_space_init = np.random.uniform(size=tensor_shape + [3])
    camera_position_init = np.random.uniform(size=tensor_shape + [3])
    camera_up_init = np.random.uniform(size=tensor_shape + [3])
    look_at_init = np.random.uniform(size=tensor_shape + [3])
    vertical_field_of_view_init = np.random.uniform(
        0.1, 1.0, size=tensor_shape + [1])
    lower_left_corner_init = np.random.uniform(size=tensor_shape + [2])
    screen_dimensions_init = np.random.uniform(
        0.1, 1.0, size=tensor_shape + [2])
    near_init = np.random.uniform(0.1, 1.0, size=tensor_shape + [1])
    far_init = near_init + np.random.uniform(0.1, 1.0, size=tensor_shape + [1])

    point_world_space_tensor = tf.convert_to_tensor(
        value=point_world_space_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    camera_up_tensor = tf.convert_to_tensor(value=camera_up_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))

    y_projection, y_w = glm.model_to_screen(
        point_world_space_tensor, camera_position_tensor, look_at_tensor,
        camera_up_tensor, vertical_field_of_view_tensor,
        screen_dimensions_tensor, near_tensor, far_tensor,
        lower_left_corner_tensor)
    with self.subTest(name="jacobian_point_world_space"):
      self.assert_jacobian_is_correct(point_world_space_tensor,
                                      point_world_space_init, y_projection)
      self.assert_jacobian_is_correct(point_world_space_tensor,
                                      point_world_space_init, y_w)
    with self.subTest(name="jacobian_camera_position"):
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y_projection)
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y_w)
    with self.subTest(name="jacobian_look_at"):
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init,
                                      y_projection)
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y_w)
    with self.subTest(name="jacobian_camera_up"):
      self.assert_jacobian_is_correct(camera_up_tensor, camera_up_init,
                                      y_projection)
      self.assert_jacobian_is_correct(camera_up_tensor, camera_up_init, y_w)
    with self.subTest(name="jacobian_vertical_field_of_view"):
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y_projection)
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y_w)
    with self.subTest(name="jacobian_screen_dimensions"):
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y_projection)
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y_w)
    with self.subTest(name="jacobian_near"):
      self.assert_jacobian_is_correct(near_tensor, near_init, y_projection)
      self.assert_jacobian_is_correct(near_tensor, near_init, y_w)
    with self.subTest(name="jacobian_far"):
      self.assert_jacobian_is_correct(far_tensor, far_init, y_projection)
      self.assert_jacobian_is_correct(far_tensor, far_init, y_w)
    with self.subTest(name="jacobian_lower_left_corner"):
      self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                      lower_left_corner_init, y_projection)

  def test_perspective_correct_interpolation_preset(self):
    """Tests that perspective_correct_interpolation generates expected results."""
    vertices = np.tile(((-0.2857143, 0.2857143, 5.0),
                        (0.2857143, 0.2857143, 0.5), (0.0, -0.2857143, 1.0)),
                       (2, 1, 1))
    attributes = np.tile((((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))),
                         (2, 1, 1))
    pixel_position = np.array(((125.5, 375.5), (250.5, 250.5)))
    camera_position = np.tile((0.0, 0.0, 0.0), (2, 3, 1))
    look_at = np.tile((0.0, 0.0, 1.0), (2, 3, 1))
    up_vector = np.tile((0.0, 1.0, 0.0), (2, 3, 1))
    vertical_field_of_view = np.tile((1.0471975511965976,), (2, 3, 1))
    screen_dimensions = np.tile((501.0, 501.0), (2, 3, 1))
    near = np.tile((0.01,), (2, 3, 1))
    far = np.tile((10.0,), (2, 3, 1))
    lower_left_corner = np.tile((0.0, 0.0), (2, 3, 1))

    pred = glm.perspective_correct_interpolation(
        vertices, attributes, pixel_position, camera_position, look_at,
        up_vector, vertical_field_of_view, screen_dimensions, near, far,
        lower_left_corner)
    gt = ((0.051941, 0.84417719, 0.10388184), (0.25, 0.25, 0.5))
    self.assertAllClose(pred, gt)

  @parameterized.parameters(
      ((500, 400, 3, 3), (3, 7), (2,), (3,), (3,), (3,), (1,), (2,), (1,), (1,),
       (2,)),
      ((3, 3), (3, 7), (2,), (3,), (3,), (3,), (1,), (2,), (1,), (1,), (2,)),
      ((None, 3, 3), (None, 3, 7), (None, 2), (None, 3), (None, 3), (None, 3),
       (None, 1), (None, 2), (None, 1), (None, 1), (None, 2)),
  )
  def test_perspective_correct_interpolation_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.perspective_correct_interpolation,
                                        shapes)

  @parameterized.parameters(
      ("point_model_space must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3, 2), (3, 7), (2,), (3,), (3,),
       (3,)),
      ("must have exactly 3 dimensions in axis -2", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (2, 3), (3, 7), (2,), (3,), (3,), (3,)),
      ("attribute must have exactly 3 dimensions in axis -2", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3, 3), (2, 7), (2,), (3,), (3,),
       (3,)),
      ("must have exactly 2 dimensions in axis -1", (1.0,), (1.0, 1.0), (1.0,),
       (2.0,), (0.0, 0.0), (3, 3), (3, 7), (1,), (3,), (3,), (3,)),
      ("camera_position must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,), (4,), (3,),
       (3,)),
      ("look_at must have exactly 3 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0,), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,), (3,), (1,), (3,)),
      ("up_vector must have exactly 3 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,), (3,), (3,),
       (2,)),
      ("vertical_field_of_view must have exactly 1 dimensions in axis -1",
       (1.0, 1.0), (1.0, 1.0), (1.0,), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,),
       (3,), (3,), (3,)),
      ("screen_dimensions must have exactly 2 dimensions in axis -1", (1.0,),
       (1.0,), (1.0,), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,), (3,), (3,),
       (3,)),
      ("near must have exactly 1 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0, 1.0), (2.0,), (0.0, 0.0), (3, 3), (3, 7), (2,), (3,), (3,), (3,)),
      ("far must have exactly 1 dimensions in axis -1", (1.0,), (1.0, 1.0),
       (1.0,), (2.0, 2.0), (0.0, 0.0), (3, 3), (3, 7), (2,), (3,), (3,), (3,)),
      ("lower_left_corner must have exactly 2 dimensions in axis -1", (1.0,),
       (1.0, 1.0), (1.0,), (2.0,), (0.0,), (3, 3), (3, 7), (2,), (3,), (3,),
       (3,)),
  )
  def test_perspective_correct_interpolation_exception_raised(
      self, error_msg, vertical_field_of_view, screen_dimensions, near, far,
      lower_left_corner, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        func=glm.perspective_correct_interpolation,
        error_msg=error_msg,
        shapes=shapes,
        vertical_field_of_view=vertical_field_of_view,
        screen_dimensions=screen_dimensions,
        near=near,
        far=far,
        lower_left_corner=lower_left_corner)

  def test_perspective_correct_interpolation_jacobian_preset(self):
    """Tests the Jacobian of perspective_correct_interpolation."""
    vertices_init = np.tile(
        ((-0.2857143, 0.2857143, 5.0), (0.2857143, 0.2857143, 0.5),
         (0.0, -0.2857143, 1.0)), (2, 1, 1))
    attributes_init = np.tile(
        (((1.0, 0.0, 0.0), (0.0, 1.0, 0.0), (0.0, 0.0, 1.0))), (2, 1, 1))
    pixel_position_init = np.array(((125.5, 375.5), (250.5, 250.5)))
    camera_position_init = np.tile((0.0, 0.0, 0.0), (2, 3, 1))
    look_at_init = np.tile((0.0, 0.0, 1.0), (2, 3, 1))
    up_vector_init = np.tile((0.0, 1.0, 0.0), (2, 3, 1))
    vertical_field_of_view_init = np.tile((1.0471975511965976,), (2, 3, 1))
    screen_dimensions_init = np.tile((501.0, 501.0), (2, 3, 1))
    near_init = np.tile((0.01,), (2, 3, 1))
    far_init = np.tile((10.0,), (2, 3, 1))
    lower_left_corner_init = np.tile((0.0, 0.0), (2, 3, 1))

    vertices_tensor = tf.convert_to_tensor(value=vertices_init)
    attributes_tensor = tf.convert_to_tensor(value=attributes_init)
    pixel_position_tensor = tf.convert_to_tensor(value=pixel_position_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)

    y = glm.perspective_correct_interpolation(
        vertices_tensor, attributes_tensor, pixel_position_tensor,
        camera_position_tensor, look_at_tensor, up_vector_tensor,
        vertical_field_of_view_tensor, screen_dimensions_tensor, near_tensor,
        far_tensor, lower_left_corner_tensor)
    with self.subTest(name="jacobian_vertices"):
      self.assert_jacobian_is_correct(vertices_tensor, vertices_init, y)
    with self.subTest(name="jacobian_attributes"):
      self.assert_jacobian_is_correct(attributes_tensor, attributes_init, y)
    with self.subTest(name="jacobian_pixel_potision"):
      self.assert_jacobian_is_correct(pixel_position_tensor,
                                      pixel_position_init, y)
    with self.subTest(name="jacobian_camera_position"):
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y)
    with self.subTest(name="jacobian_look_at"):
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    with self.subTest(name="jacobian_up_vector"):
      self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)
    with self.subTest(name="jacobian_vertical_field_of_view"):
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y)
    with self.subTest(name="jacobian_screen_dimensions"):
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y)
    with self.subTest(name="jacobian_near"):
      self.assert_jacobian_is_correct(near_tensor, near_init, y)
    with self.subTest(name="jacobian_far"):
      self.assert_jacobian_is_correct(far_tensor, far_init, y)
    with self.subTest(name="jacobian_lower_left_corner"):
      self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                      lower_left_corner_init, y)

  def test_perspective_correct_interpolation_jacobian_random(self):
    """Tests the Jacobian of perspective_correct_interpolation."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    vertices_init = np.random.uniform(size=tensor_shape + [3, 3])
    num_attributes = np.random.randint(1, 10)
    attributes_init = np.random.uniform(size=tensor_shape + [3, num_attributes])
    pixel_position_init = np.random.uniform(size=tensor_shape + [2])
    camera_position_init = np.random.uniform(size=tensor_shape + [3, 3])
    look_at_init = np.random.uniform(size=tensor_shape + [3, 3])
    up_vector_init = np.random.uniform(size=tensor_shape + [3, 3])
    vertical_field_of_view_init = np.random.uniform(
        0.1, 1.0, size=tensor_shape + [3, 1])
    screen_dimensions_init = np.random.uniform(
        1.0, 10.0, size=tensor_shape + [3, 2])
    near_init = np.random.uniform(1.0, 10.0, size=tensor_shape + [3, 1])
    far_init = near_init + np.random.uniform(
        0.1, 1.0, size=tensor_shape + [3, 1])
    lower_left_corner_init = np.random.uniform(size=tensor_shape + [3, 2])

    vertices_tensor = tf.convert_to_tensor(value=vertices_init)
    attributes_tensor = tf.convert_to_tensor(value=attributes_init)
    pixel_position_tensor = tf.convert_to_tensor(value=pixel_position_init)
    camera_position_tensor = tf.convert_to_tensor(value=camera_position_init)
    look_at_tensor = tf.convert_to_tensor(value=look_at_init)
    up_vector_tensor = tf.convert_to_tensor(value=up_vector_init)
    vertical_field_of_view_tensor = tf.identity(
        tf.convert_to_tensor(value=vertical_field_of_view_init))
    screen_dimensions_tensor = tf.identity(
        tf.convert_to_tensor(value=screen_dimensions_init))
    near_tensor = tf.identity(tf.convert_to_tensor(value=near_init))
    far_tensor = tf.identity(tf.convert_to_tensor(value=far_init))
    lower_left_corner_tensor = tf.convert_to_tensor(
        value=lower_left_corner_init)

    y = glm.perspective_correct_interpolation(
        vertices_tensor, attributes_tensor, pixel_position_tensor,
        camera_position_tensor, look_at_tensor, up_vector_tensor,
        vertical_field_of_view_tensor, screen_dimensions_tensor, near_tensor,
        far_tensor, lower_left_corner_tensor)
    with self.subTest(name="jacobian_vertices"):
      self.assert_jacobian_is_correct(vertices_tensor, vertices_init, y)
    with self.subTest(name="jacobian_attributes"):
      self.assert_jacobian_is_correct(attributes_tensor, attributes_init, y)
    with self.subTest(name="jacobian_pixel_potision"):
      self.assert_jacobian_is_correct(pixel_position_tensor,
                                      pixel_position_init, y)
    with self.subTest(name="jacobian_camera_position"):
      self.assert_jacobian_is_correct(camera_position_tensor,
                                      camera_position_init, y)
    with self.subTest(name="jacobian_look_at"):
      self.assert_jacobian_is_correct(look_at_tensor, look_at_init, y)
    with self.subTest(name="jacobian_up_vector"):
      self.assert_jacobian_is_correct(up_vector_tensor, up_vector_init, y)
    with self.subTest(name="jacobian_vertical_field_of_view"):
      self.assert_jacobian_is_correct(vertical_field_of_view_tensor,
                                      vertical_field_of_view_init, y)
    with self.subTest(name="jacobian_screen_dimensions"):
      self.assert_jacobian_is_correct(screen_dimensions_tensor,
                                      screen_dimensions_init, y)
    with self.subTest(name="jacobian_near"):
      self.assert_jacobian_is_correct(near_tensor, near_init, y)
    with self.subTest(name="jacobian_far"):
      self.assert_jacobian_is_correct(far_tensor, far_init, y)
    with self.subTest(name="jacobian_lower_left_corner"):
      self.assert_jacobian_is_correct(lower_left_corner_tensor,
                                      lower_left_corner_init, y)


if __name__ == "__main__":
  test_case.main()
