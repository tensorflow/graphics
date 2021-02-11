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
"""Tests for OpenGL math routines."""

import math

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.rendering.opengl import math as glm
from tensorflow_graphics.util import test_case


class MathTest(test_case.TestCase):

  def test_model_to_eye_preset(self):
    """Tests that model_to_eye generates expected results.."""
    point = ((2.0, 3.0, 4.0), (3.0, 4.0, 5.0))
    camera_position = ((0.0, 0.0, 0.0), (0.1, 0.2, 0.3))
    look_at_point = ((0.0, 0.0, 1.0), (0.4, 0.5, 0.6))
    up_vector = ((0.0, 1.0, 0.0), (0.7, 0.8, 0.9))

    pred = glm.model_to_eye(point, camera_position, look_at_point, up_vector)

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

    self.assert_jacobian_is_correct_fn(
        glm.model_to_eye,
        [point_init, camera_position_init, look_at_init, up_vector_init])

  def test_model_to_eye_jacobian_random(self):
    """Tests the Jacobian of model_to_eye."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [3])
    camera_position_init = np.random.uniform(size=tensor_shape + [3])
    look_at_init = np.random.uniform(size=tensor_shape + [3])
    up_vector_init = np.random.uniform(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(
        glm.model_to_eye,
        [point_init, camera_position_init, look_at_init, up_vector_init])

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

    self.assert_jacobian_is_correct_fn(
        glm.eye_to_clip, [
            point_init, vertical_field_of_view_init, aspect_ratio_init,
            near_init, far_init
        ],
        atol=1e-5)

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

    self.assert_jacobian_is_correct_fn(
        glm.eye_to_clip, [
            point_init, vertical_field_of_view_init, aspect_ratio_init,
            near_init, far_init
        ],
        atol=1e-03)

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

    self.assert_jacobian_is_correct_fn(glm.clip_to_ndc, [point_init])

  def test_clip_to_ndc_jacobian_random(self):
    """Tests the Jacobian of clip_to_ndc."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    point_init = np.random.uniform(size=tensor_shape + [4])

    self.assert_jacobian_is_correct_fn(
        glm.clip_to_ndc, [point_init], atol=1e-04)

  def test_ndc_to_screen_preset(self):
    """Tests that ndc_to_screen generates expected results."""
    point = ((0.0, 0.0, 0.0), (1.0, 1.0, 0.0), (-1.0, -1.0, 1.0), (1.0, -1.0,
                                                                   -1.0))
    lower_left_corner = ((0.0, 0.0), (0.0, 0.0), (3.0, 4.0), (4.0, 3.0))
    screen_dimensions = ((10.0, 10.0), (5.0, 10.0), (20.0, 20.0), (15.0, 35.0))
    near = ((1.0,), (1.0,), (2.0,), (3.0,))
    far = ((11.0,), (11.0,), (20.0,), (4.0,))

    pred = glm.ndc_to_screen(point, lower_left_corner, screen_dimensions, near,
                             far)

    gt = ((5.0, 5.0, 6.0), (5.0, 10.0, 6.0), (3.0, 4.0, 20.0), (19.0, 3.0, 3.0))
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

    self.assert_jacobian_is_correct_fn(glm.ndc_to_screen, [
        point_init, lower_left_corner_init, screen_dimensions_init, near_init,
        far_init
    ])

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

    self.assert_jacobian_is_correct_fn(glm.ndc_to_screen, [
        point_init, lower_left_corner_init, screen_dimensions_init, near_init,
        far_init
    ])

  def test_model_to_screen_preset(self):
    """Tests that model_to_screen generates expected results."""
    point_world_space = np.array(((3.1, 4.1, 5.1), (-1.1, 2.2, -3.1)))
    camera_position = np.array(((0.0, 0.0, 0.0), (0.4, -0.8, 0.1)))
    camera_up = np.array(((0.0, 1.0, 0.0), (0.0, 0.0, 1.0)))
    look_at_point = np.array(((0.0, 0.0, 1.0), (0.0, 1.0, 0.0)))
    vertical_field_of_view = np.array(
        ((60.0 * math.pi / 180.0,), (65 * math.pi / 180,)))
    lower_left_corner = np.array(((0.0, 0.0), (10.0, 20.0)))
    screen_dimensions = np.array(((501.0, 501.0), (400.0, 600.0)))
    near = np.array(((0.01,), (1.0,)))
    far = np.array(((4.0,), (3.0,)))

    # Build matrices.
    model_to_eye_matrix = look_at.right_handed(camera_position, look_at_point,
                                               camera_up)
    perspective_matrix = perspective.right_handed(
        vertical_field_of_view,
        screen_dimensions[..., 0:1] / screen_dimensions[..., 1:2], near, far)

    pred_screen, pred_w = glm.model_to_screen(point_world_space,
                                              model_to_eye_matrix,
                                              perspective_matrix,
                                              screen_dimensions,
                                              lower_left_corner)

    gt_screen = ((-13.23016357, 599.30444336, 4.00215721),
                 (98.07017517, -95.40383911, 3.1234405))
    gt_w = ((5.1,), (3.42247,))
    self.assertAllClose(pred_screen, gt_screen, atol=1e-5, rtol=1e-5)
    self.assertAllClose(pred_w, gt_w)

  @parameterized.parameters(
      ((3,), (4, 4), (4, 4), (2,), (2,)),
      ((640, 480, 3), (4, 4), (4, 4), (2,), (2,)),
      ((None, 3), (None, 4, 4), (None, 4, 4), (None, 2), (None, 2)),
      ((3,), (None, 1, 4, 4), (None, 1, 4, 4), (None, 1, 2), (None, 1, 2)),
  )
  def test_model_to_screen_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.model_to_screen, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (9.0, 12.0), (0.0, 0.0),
       (2,), (4, 4), (4, 4)),
      ("must have exactly 4 dimensions in axis -1", (9.0, 12.0), (0.0, 0.0),
       (3,), (4, 3), (4, 4)),
      ("must have exactly 4 dimensions in axis -2", (9.0, 12.0), (0.0, 0.0),
       (3,), (3, 4), (4, 4)),
      ("must have exactly 4 dimensions in axis -1", (9.0, 12.0), (0.0, 0.0),
       (3,), (4, 4), (4, 3)),
      ("must have exactly 4 dimensions in axis -2", (9.0, 12.0), (0.0, 0.0),
       (3,), (4, 4), (3, 4)),
      ("Not all batch dimensions are broadcast-compatible", (9.0, 12.0),
       (0.0, 0.0), (2, 3), (3, 4, 4), (3, 4, 4)),
  )
  def test_model_to_screen_exception_raised(self, error_msg, screen_dimensions,
                                            lower_left_corner, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(
        func=glm.model_to_screen,
        error_msg=error_msg,
        shapes=shapes,
        screen_dimensions=screen_dimensions,
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

    # Build matrices.
    model_to_eye_matrix = look_at.right_handed(camera_position_init,
                                               look_at_init, camera_up_init)
    perspective_matrix = perspective.right_handed(
        vertical_field_of_view_init,
        screen_dimensions_init[..., 0:1] / screen_dimensions_init[..., 1:2],
        near_init, far_init)

    args = [
        point_world_space_init, model_to_eye_matrix, perspective_matrix,
        screen_dimensions_init, lower_left_corner_init
    ]

    with self.subTest(name="jacobian_y_projection"):
      self.assert_jacobian_is_correct_fn(
          lambda *args: glm.model_to_screen(*args)[0], args, atol=1e-4)

    # TODO(julienvalentin): will be fixed before submission
    # with self.subTest(name="jacobian_w"):
    #   self.assert_jacobian_is_correct_fn(
    #       lambda *args: glm.model_to_screen(*args)[1], args)

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

    # Build matrices.
    model_to_eye_matrix = look_at.right_handed(camera_position_init,
                                               look_at_init, camera_up_init)
    perspective_matrix = perspective.right_handed(
        vertical_field_of_view_init,
        screen_dimensions_init[..., 0:1] / screen_dimensions_init[..., 1:2],
        near_init, far_init)

    args = [
        point_world_space_init, model_to_eye_matrix, perspective_matrix,
        screen_dimensions_init, lower_left_corner_init
    ]

    with self.subTest(name="jacobian_y_projection"):
      self.assert_jacobian_is_correct_fn(
          lambda *args: glm.model_to_screen(*args)[0], args, atol=1e-4)

    # TODO(julienvalentin): will be fixed before submission
    # with self.subTest(name="jacobian_w"):
    #   self.assert_jacobian_is_correct_fn(
    #       lambda *args: glm.model_to_screen(*args)[1], args)

  def test_perspective_correct_interpolation_preset(self):
    """Tests that perspective_correct_interpolation generates expected results."""
    camera_origin = np.array((0.0, 0.0, 0.0))
    camera_up = np.array((0.0, 1.0, 0.0))
    look_at_point = np.array((0.0, 0.0, 1.0))
    fov = np.array((90.0 * np.math.pi / 180.0,))
    bottom_left = np.array((0.0, 0.0))
    image_size = np.array((501.0, 501.0))
    near_plane = np.array((0.01,))
    far_plane = np.array((10.0,))
    batch_size = np.random.randint(1, 5)
    triangle_x_y = np.random.uniform(-10.0, 10.0, (batch_size, 3, 2))
    triangle_z = np.random.uniform(2.0, 10.0, (batch_size, 3, 1))
    triangles = np.concatenate((triangle_x_y, triangle_z), axis=-1)
    # Builds barycentric weights.
    barycentric_weights = np.random.uniform(size=(batch_size, 3))
    barycentric_weights = barycentric_weights / np.sum(
        barycentric_weights, axis=-1, keepdims=True)
    # Barycentric interpolation of vertex positions.
    convex_combination = np.einsum("ba, bac -> bc", barycentric_weights,
                                   triangles)
    # Build matrices.
    model_to_eye_matrix = look_at.right_handed(camera_origin, look_at_point,
                                               camera_up)
    perspective_matrix = perspective.right_handed(
        fov, (image_size[0:1] / image_size[1:2]), near_plane, far_plane)

    # Computes where those points project in screen coordinates.
    pixel_position, _ = glm.model_to_screen(convex_combination,
                                            model_to_eye_matrix,
                                            perspective_matrix, image_size,
                                            bottom_left)

    # Builds attributes.
    num_pixels = pixel_position.shape[0]
    attribute_size = np.random.randint(10)
    attributes = np.random.uniform(size=(num_pixels, 3, attribute_size))

    prediction = glm.perspective_correct_interpolation(triangles, attributes,
                                                       pixel_position[..., 0:2],
                                                       model_to_eye_matrix,
                                                       perspective_matrix,
                                                       image_size, bottom_left)

    groundtruth = np.einsum("ba, bac -> bc", barycentric_weights, attributes)
    self.assertAllClose(prediction, groundtruth)

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

    # Build matrices.
    model_to_eye_matrix_init = look_at.right_handed(camera_position_init,
                                                    look_at_init,
                                                    up_vector_init)
    perspective_matrix_init = perspective.right_handed(
        vertical_field_of_view_init,
        screen_dimensions_init[..., 0:1] / screen_dimensions_init[..., 1:2],
        near_init, far_init)

    self.assert_jacobian_is_correct_fn(glm.perspective_correct_interpolation, [
        vertices_init, attributes_init, pixel_position_init,
        model_to_eye_matrix_init, perspective_matrix_init,
        screen_dimensions_init, lower_left_corner_init
    ])

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
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

    # Build matrices.
    model_to_eye_matrix_init = look_at.right_handed(camera_position_init,
                                                    look_at_init,
                                                    up_vector_init)
    perspective_matrix_init = perspective.right_handed(
        vertical_field_of_view_init,
        screen_dimensions_init[..., 0:1] / screen_dimensions_init[..., 1:2],
        near_init, far_init)

    self.assert_jacobian_is_correct_fn(
        glm.perspective_correct_interpolation, [
            vertices_init, attributes_init, pixel_position_init,
            model_to_eye_matrix_init, perspective_matrix_init,
            screen_dimensions_init, lower_left_corner_init
        ],
        atol=1e-4)

  @parameterized.parameters(
      ((3, 3), (2,), (4, 4), (4, 4), (2,)),
      ((3, 3), (7, 2), (4, 4), (4, 4), (2,)),
      ((3, 3), (None, 2), (4, 4), (4, 4), (2,)),
      ((7, 3, 3), (2,), (4, 4), (4, 4), (2,)),
      ((None, 3, 3), (2,), (4, 4), (4, 4), (2,)),
  )
  def test_perspective_correct_barycentrics_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.perspective_correct_barycentrics,
                                        shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (3, 3), (2,), (4, 4),
       (4, 4), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, 4), (2,), (4, 4),
       (4, 4), (3,)),
      ("must have exactly 3 dimensions in axis -2", (4, 3), (2,), (4, 4),
       (4, 4), (3,)),
  )
  def test_perspective_correct_barycentrics_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.perspective_correct_barycentrics,
                                    error_msg, shapes)

  def test_perspective_correct_barycentrics_preset(self):
    """Tests that perspective_correct_barycentrics generates expected results."""
    camera_origin = np.array((0.0, 0.0, 0.0))
    camera_up = np.array((0.0, 1.0, 0.0))
    look_at_point = np.array((0.0, 0.0, 1.0))
    fov = np.array((90.0 * np.math.pi / 180.0,))
    bottom_left = np.array((0.0, 0.0))
    image_size = np.array((501.0, 501.0))
    near_plane = np.array((0.01,))
    far_plane = np.array((10.0,))
    batch_size = np.random.randint(1, 5)
    triangle_x_y = np.random.uniform(-10.0, 10.0, (batch_size, 3, 2))
    triangle_z = np.random.uniform(2.0, 10.0, (batch_size, 3, 1))
    triangles = np.concatenate((triangle_x_y, triangle_z), axis=-1)
    # Builds barycentric weights.
    barycentric_weights = np.random.uniform(size=(batch_size, 3))
    barycentric_weights = barycentric_weights / np.sum(
        barycentric_weights, axis=-1, keepdims=True)
    # Barycentric interpolation of vertex positions.
    convex_combination = np.einsum("ba, bac -> bc", barycentric_weights,
                                   triangles)
    # Build matrices.
    model_to_eye_matrix = look_at.right_handed(camera_origin, look_at_point,
                                               camera_up)
    perspective_matrix = perspective.right_handed(
        fov, (image_size[0:1] / image_size[1:2]), near_plane, far_plane)

    # Computes where those points project in screen coordinates.
    pixel_position, _ = glm.model_to_screen(convex_combination,
                                            model_to_eye_matrix,
                                            perspective_matrix, image_size,
                                            bottom_left)

    prediction = glm.perspective_correct_barycentrics(triangles,
                                                      pixel_position[..., 0:2],
                                                      model_to_eye_matrix,
                                                      perspective_matrix,
                                                      image_size, bottom_left)

    self.assertAllClose(prediction, barycentric_weights)

  def test_perspective_correct_barycentrics_jacobian_random(self):
    """Tests the Jacobian of perspective_correct_barycentrics."""
    tensor_size = np.random.randint(1, 3)
    tensor_shape = np.random.randint(1, 5, size=(tensor_size)).tolist()
    vertices_init = np.random.uniform(size=tensor_shape + [3, 3])
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

    # Build matrices.
    model_to_eye_matrix_init = look_at.right_handed(camera_position_init,
                                                    look_at_init,
                                                    up_vector_init)
    perspective_matrix_init = perspective.right_handed(
        vertical_field_of_view_init,
        screen_dimensions_init[..., 0:1] / screen_dimensions_init[..., 1:2],
        near_init, far_init)

    self.assert_jacobian_is_correct_fn(
        glm.perspective_correct_barycentrics, [
            vertices_init, pixel_position_init, model_to_eye_matrix_init,
            perspective_matrix_init, screen_dimensions_init,
            lower_left_corner_init
        ],
        atol=1e-4)

  @parameterized.parameters(
      ((3, 7), (3,)),
      ((2, 3, 7), (2, 3)),
      ((None, 3, 7), (None, 3)),
  )
  def test_interpolate_attributes_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(glm.interpolate_attributes, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -2", (2, 7), (3,)),
      ("must have exactly 3 dimensions in axis -1", (3, 7), (2,)),
      ("Not all batch dimensions are broadcast-compatible", (5, 3, 7), (4, 3)),
  )
  def test_interpolate_attributes_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(glm.interpolate_attributes, error_msg,
                                    shapes)

  def test_interpolate_attributes_random(self):
    """Checks the output of interpolate_attributes."""
    attributes = np.random.uniform(-1.0, 1.0, size=(3,))
    barycentric = np.random.uniform(0.0, 1.0, size=(3,))
    barycentric = barycentric / np.linalg.norm(
        barycentric, axis=-1, ord=1, keepdims=True)

    groundtruth = np.sum(attributes * barycentric, keepdims=True)
    attributes = np.reshape(attributes, (3, 1))
    prediction = glm.interpolate_attributes(attributes, barycentric)
    self.assertAllClose(groundtruth, prediction)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_interpolate_attributes_jacobian_random(self):
    """Tests the jacobian of interpolate_attributes."""
    batch_size = np.random.randint(1, 5)
    attributes = np.random.uniform(-1.0, 1.0, size=(batch_size, 3, 1))
    barycentric = np.random.uniform(
        0.0, 1.0, size=(
            batch_size,
            3,
        ))
    barycentric = barycentric / np.linalg.norm(
        barycentric, axis=-1, ord=1, keepdims=True)

    self.assert_jacobian_is_correct_fn(glm.interpolate_attributes,
                                       [attributes, barycentric])


if __name__ == "__main__":
  test_case.main()
