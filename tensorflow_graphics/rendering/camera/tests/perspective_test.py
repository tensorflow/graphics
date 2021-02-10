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
"""Tests for perspective camera functionalities."""

import math
import sys

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.util import test_case


class PerspectiveTest(test_case.TestCase):

  @parameterized.parameters(
      ("must have exactly 4 dimensions in axis -1", (4, 3)),
      ("must have exactly 4 dimensions in axis -2", (5, 4)),
      ("must have exactly 4 dimensions in axis -2", (None, 4)),
      ("must have exactly 4 dimensions in axis -1", (4, None)),
  )
  def test_parameters_from_right_handed_shape_exception_raised(
      self, error_msg, *shapes):
    """Checks the inputs of the from_right_handed_shape function."""
    self.assert_exception_is_raised(perspective.parameters_from_right_handed,
                                    error_msg, shapes)

  @parameterized.parameters(
      ((4, 4),),
      ((None, 4, 4),),
      ((None, None, 4, 4),),
  )
  def test_parameters_from_right_handed_shape_exception_not_raised(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(
        perspective.parameters_from_right_handed, shapes)

  def test_parameters_from_right_handed_random(self):
    """Tests that parameters_from_right_handed returns the expected values."""
    tensor_size = np.random.randint(2, 4)
    tensor_shape = np.random.randint(2, 5, size=(tensor_size)).tolist()
    vertical_field_of_view_gt = np.random.uniform(
        sys.float_info.epsilon, np.pi - sys.float_info.epsilon,
        tensor_shape + [1])
    aspect_ratio_gt = np.random.uniform(0.1, 10.0, tensor_shape + [1])
    near_gt = np.random.uniform(0.1, 100.0, tensor_shape + [1])
    far_gt = near_gt + np.random.uniform(0.1, 100.0, tensor_shape + [1])
    projection_matrix = perspective.right_handed(vertical_field_of_view_gt,
                                                 aspect_ratio_gt, near_gt,
                                                 far_gt)

    vertical_field_of_view_pred, aspect_ratio_pred, near_pred, far_pred = perspective.parameters_from_right_handed(
        projection_matrix)

    with self.subTest(name="vertical_field_of_view"):
      self.assertAllClose(vertical_field_of_view_gt,
                          vertical_field_of_view_pred)

    with self.subTest(name="aspect_ratio"):
      self.assertAllClose(aspect_ratio_gt, aspect_ratio_pred)

    with self.subTest(name="near_plane"):
      self.assertAllClose(near_gt, near_pred)

    with self.subTest(name="far_plane"):
      self.assertAllClose(far_gt, far_pred)

  def test_parameters_from_right_handed_jacobian_random(self):
    """Tests the Jacobian of parameters_from_right_handed."""
    tensor_size = np.random.randint(2, 4)
    tensor_shape = np.random.randint(2, 5, size=(tensor_size)).tolist()
    vertical_field_of_view = np.random.uniform(sys.float_info.epsilon,
                                               np.pi - sys.float_info.epsilon,
                                               tensor_shape + [1])
    aspect_ratio = np.random.uniform(0.1, 10.0, tensor_shape + [1])
    near = np.random.uniform(0.1, 100.0, tensor_shape + [1])
    far = near + np.random.uniform(0.1, 100.0, tensor_shape + [1])
    projection_matrix = perspective.right_handed(vertical_field_of_view,
                                                 aspect_ratio, near, far)

    with self.subTest(name="vertical_field_of_view"):
      self.assert_jacobian_is_finite_fn(
          lambda x: perspective.parameters_from_right_handed(x)[0],
          [projection_matrix])

    with self.subTest(name="aspect_ratio"):
      self.assert_jacobian_is_finite_fn(
          lambda x: perspective.parameters_from_right_handed(x)[1],
          [projection_matrix])

    with self.subTest(name="near_plane"):
      self.assert_jacobian_is_finite_fn(
          lambda x: perspective.parameters_from_right_handed(x)[2],
          [projection_matrix])

    with self.subTest(name="far_plane"):
      self.assert_jacobian_is_finite_fn(
          lambda x: perspective.parameters_from_right_handed(x)[3],
          [projection_matrix])

  def test_perspective_right_handed_preset(self):
    """Tests that perspective_right_handed generates expected results."""
    vertical_field_of_view = ((60.0 * math.pi / 180.0,),
                              (50.0 * math.pi / 180.0,))
    aspect_ratio = ((1.5,), (1.1,))
    near = ((1.0,), (1.2,))
    far = ((10.0,), (5.0,))

    pred = perspective.right_handed(vertical_field_of_view, aspect_ratio, near,
                                    far)
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
    self.assert_exception_is_not_raised(perspective.right_handed, shapes)

  @parameterized.parameters(
      ("Not all batch dimensions are identical", (1,), (3, 1), (3, 1), (3, 1)),
      ("Not all batch dimensions are identical", (3, 1), (None, 3, 1), (3, 1),
       (3, 1)),
  )
  def test_perspective_right_handed_shape_exception_raised(
      self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.right_handed, error_msg, shapes)

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
          perspective.right_handed(vertical_field_of_view, aspect_ratio, near,
                                   far))

  def test_perspective_right_handed_cross_jacobian_preset(self):
    """Tests the Jacobian of perspective_right_handed."""
    vertical_field_of_view_init = np.array((1.0,))
    aspect_ratio_init = np.array((1.0,))
    near_init = np.array((1.0,))
    far_init = np.array((10.0,))

    self.assert_jacobian_is_correct_fn(
        perspective.right_handed,
        [vertical_field_of_view_init, aspect_ratio_init, near_init, far_init])

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

    self.assert_jacobian_is_correct_fn(
        perspective.right_handed,
        [vertical_field_of_view_init, aspect_ratio_init, near_init, far_init])

  @parameterized.parameters(
      ((3, 3),),
      ((3, 3, 3),),
      ((None, 3, 3),),
  )
  def test_intrinsics_from_matrix_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.intrinsics_from_matrix,
                                        shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", (3,)),
      ("must have exactly 3 dimensions in axis -2", (None, 3)),
      ("must have exactly 3 dimensions in axis -1", (3, None)),
  )
  def test_intrinsics_from_matrix_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.intrinsics_from_matrix,
                                    error_msg, shapes)

  @parameterized.parameters(
      ((((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),), ((0., 0.), (0., 0.),
                                                       (0.0,))),
      ((((1., 0., 3.), (0., 2., 4.), (0., 0., 1.)),), ((1., 2.), (3., 4.),
                                                       (0.0,))),
  )
  def test_intrinsics_from_matrix_preset(self, test_inputs, test_outputs):
    """Tests that intrinsics_from_matrix gives the correct result."""
    self.assert_output_is_correct(perspective.intrinsics_from_matrix,
                                  test_inputs, test_outputs)

  def test_intrinsics_from_matrix_to_intrinsics_random(self):
    """Tests that converting intrinsics to a matrix and back is consistent."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    random_skew_coeff = np.random.normal(size=tensor_shape + [1])

    matrix = perspective.matrix_from_intrinsics(random_focal,
                                                random_principal_point,
                                                random_skew_coeff)
    focal, principal_point, skew_coeff = perspective.intrinsics_from_matrix(
        matrix)
    random_skew_coeff = np.reshape(random_skew_coeff, (1, 1))

    self.assertAllClose(random_focal, focal, rtol=1e-3)
    self.assertAllClose(random_principal_point, principal_point, rtol=1e-3)
    self.assertAllClose(random_skew_coeff, skew_coeff, rtol=1e-3)

  @parameterized.parameters(
      ((2,), (2,), (1,)),
      ((2, 2), (2, 2), (2, 1)),
      ((None, 2), (None, 2), (None, 1)),
  )
  def test_matrix_from_intrinsics_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.matrix_from_intrinsics,
                                        shapes)

  @parameterized.parameters(
      ((2,), (2,)),
      ((2, 2), (2, 2)),
      ((None, 2), (None, 2)),
  )
  def test_matrix_from_intrinsics_exception_not_raised_when_skew_not_passed(
      self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.matrix_from_intrinsics,
                                        shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (None,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (None,)),
      ("Not all batch dimensions are identical.", (3, 2), (2, 2)),
  )
  def test_matrix_from_intrinsics_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.matrix_from_intrinsics,
                                    error_msg, shapes)

  @parameterized.parameters(
      (((0.0, 0.0), (0.0, 0.0), (0.0,)), (((0.0, 0.0, 0.0), (0.0, 0.0, 0.0),
                                           (0.0, 0.0, 1.0)),)),
      (((1.0, 2.0), (3.0, 4.0), (0.0,)), (((1.0, 0.0, 3.0), (0.0, 2.0, 4.0),
                                           (0.0, 0.0, 1.0)),)))
  def test_matrix_from_intrinsics_preset(self, test_inputs, test_outputs):
    """Tests that matrix_from_intrinsics gives the correct result."""
    self.assert_output_is_correct(perspective.matrix_from_intrinsics,
                                  test_inputs, test_outputs)

  def test_matrix_from_intrinsics_to_matrix_random(self):
    """Tests that converting a matrix to intrinsics and back is consistent."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    fx = random_focal[..., 0]
    fy = random_focal[..., 1]
    cx = random_principal_point[..., 0]
    cy = random_principal_point[..., 1]
    zero = np.zeros_like(fx)
    one = np.ones_like(fx)
    random_matrix = np.stack((fx, zero, cx, zero, fy, cy, zero, zero, one),
                             axis=-1).reshape(tensor_shape + [3, 3])

    focal, principal_point, skew_coefficient = perspective.intrinsics_from_matrix(
        random_matrix)
    matrix = perspective.matrix_from_intrinsics(focal,
                                                principal_point,
                                                skew_coefficient)

    self.assertAllClose(random_matrix, matrix, rtol=1e-3)

  @parameterized.parameters(
      ((3,), (2,), (2,)),
      ((2, 3), (2, 2), (2, 2)),
      ((2, 3), (2,), (2,)),
      ((None, 3), (None, 2), (None, 2)),
  )
  def test_project_exception_not_exception_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.project, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (None,), (2,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (3,), (None,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (3,), (2,), (None,)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 3), (2, 2),
       (2, 2)),
  )
  def test_project_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.project, error_msg, shape)

  @parameterized.parameters(
      (((0., 0., 1.), (1., 1.), (0., 0.)), ((0., 0.),)),
      (((4., 2., 1.), (1., 1.), (-4., -2.)), ((0., 0.),)),
      (((4., 2., 10.), (1., 1.), (-.4, -.2)), ((0., 0.),)),
      (((4., 2., 10.), (2., 1.), (-.8, -.2)), ((0., 0.),)),
      (((4., 2., 10.), (2., 1.), (-.8, 0.)), ((0., .2),)),
  )
  def test_project_preset(self, test_inputs, test_outputs):
    """Tests that the project function gives the correct result."""
    self.assert_output_is_correct(perspective.project, test_inputs,
                                  test_outputs)

  def test_project_unproject_random(self):
    """Tests that projecting and unprojecting gives an identity mapping."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_point_3d = np.random.normal(size=tensor_shape + [3])
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    random_depth = np.expand_dims(random_point_3d[..., 2], axis=-1)

    point_2d = perspective.project(random_point_3d, random_focal,
                                   random_principal_point)
    point_3d = perspective.unproject(point_2d, random_depth, random_focal,
                                     random_principal_point)

    self.assertAllClose(random_point_3d, point_3d, rtol=1e-3)

  def test_project_ray_random(self):
    """Tests that that ray is pointing toward the correct location."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_point_3d = np.random.normal(size=tensor_shape + [3])
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    random_depth = np.expand_dims(random_point_3d[..., 2], axis=-1)

    point_2d = perspective.project(random_point_3d, random_focal,
                                   random_principal_point)
    ray_3d = perspective.ray(point_2d, random_focal, random_principal_point)
    ray_3d = random_depth * ray_3d

    self.assertAllClose(random_point_3d, ray_3d, rtol=1e-3)

  @parameterized.parameters(
      ((2,), (2,), (2,)),
      ((2, 2), (2, 2), (2, 2)),
      ((3, 2), (1, 2), (2,)),
      ((None, 2), (None, 2), (None, 2)),
  )
  def test_ray_exception_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.ray, shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (None,), (2,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (None,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (2,), (None,)),
      ("Not all batch dimensions are broadcast-compatible.", (3, 2), (1, 2),
       (2, 2)),
  )
  def test_ray_exception_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.ray, error_msg, shapes)

  @parameterized.parameters(
      (((0., 0.), (1., 1.), (0., 0.)), ((0., 0., 1.),)),
      (((0., 0.), (1., 1.), (-1., -2.)), ((1., 2., 1.),)),
      (((0., 0.), (10., 1.), (-1., -2.)), ((.1, 2., 1.),)),
      (((-2., -4.), (10., 1.), (-3., -6.)), ((.1, 2., 1.),)),
  )
  def test_ray_preset(self, test_inputs, test_outputs):
    """Tests that the ray function gives the correct result."""
    self.assert_output_is_correct(perspective.ray, test_inputs, test_outputs)

  def test_ray_project_random(self):
    """Tests that the end point of the ray projects at the good location."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_point_2d = np.random.normal(size=tensor_shape + [2])
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])

    ray_3d = perspective.ray(random_point_2d, random_focal,
                             random_principal_point)
    point_2d = perspective.project(ray_3d, random_focal, random_principal_point)

    self.assertAllClose(random_point_2d, point_2d, rtol=1e-3)

  @parameterized.parameters(
      ((2,), (1,), (2,), (2,)),
      ((2, 2), (2, 1), (2, 2), (2, 2)),
      ((None, 2), (None, 1), (None, 2), (None, 2)),
  )
  def test_unproject_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(perspective.unproject, shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (None,), (1,), (2,), (2,)),
      ("must have exactly 1 dimensions in axis -1", (2,), (None,), (2,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (1,), (None,), (2,)),
      ("must have exactly 2 dimensions in axis -1", (2,), (1,), (2,), (None,)),
      ("Not all batch dimensions are identical.", (1, 2), (2, 1), (2, 2),
       (2, 2)),
  )
  def test_unproject_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(perspective.unproject, error_msg, shapes)

  @parameterized.parameters(
      (((0., 0.), (1.,), (1., 1.), (0., 0.)), ((0., 0., 1.),)),
      (((0., 0.), (1.,), (1., 1.), (-4., -2.)), ((4., 2., 1.),)),
      (((0., 0.), (10.,), (1., 1.), (-.4, -.2)), ((4., 2., 10.),)),
      (((0., 0.), (10.,), (2., 1.), (-.8, -.2)), ((4., 2., 10.),)),
      (((0., .2), (10.,), (2., 1.), (-.8, 0.)), ((4., 2., 10.),)),
  )
  def test_unproject_preset(self, test_inputs, test_outputs):
    """Tests that the unproject function gives the correct result."""
    self.assert_output_is_correct(perspective.unproject, test_inputs,
                                  test_outputs)

  def test_unproject_project_random(self):
    """Tests that unprojecting and projecting gives and identity mapping."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_point_2d = np.random.normal(size=tensor_shape + [2])
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    random_depth = np.random.normal(size=tensor_shape + [1])

    point_3d = perspective.unproject(random_point_2d, random_depth,
                                     random_focal, random_principal_point)
    point_2d = perspective.project(point_3d, random_focal,
                                   random_principal_point)

    self.assertAllClose(random_point_2d, point_2d, rtol=1e-3)

  def test_unproject_ray_random(self):
    """Tests that that ray is pointing toward the correct location."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    random_point_2d = np.random.normal(size=tensor_shape + [2])
    random_focal = np.random.normal(size=tensor_shape + [2])
    random_principal_point = np.random.normal(size=tensor_shape + [2])
    random_depth = np.random.normal(size=tensor_shape + [1])

    point_3d = perspective.unproject(random_point_2d, random_depth,
                                     random_focal, random_principal_point)
    ray_3d = perspective.ray(random_point_2d, random_focal,
                             random_principal_point)
    ray_3d = random_depth * ray_3d

    self.assertAllClose(point_3d, ray_3d, rtol=1e-3)


if __name__ == "__main__":
  test_case.main()
