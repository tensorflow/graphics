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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.util import test_case


class PerspectiveTest(test_case.TestCase):

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
      ((((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),), ((0., 0.), (0., 0.))),
      ((((1., 0., 3.), (0., 2., 4.), (0., 0., 1.)),), ((1., 2.), (3., 4.))),
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

    matrix = perspective.matrix_from_intrinsics(random_focal,
                                                random_principal_point)
    focal, principal_point = perspective.intrinsics_from_matrix(matrix)

    self.assertAllClose(random_focal, focal, rtol=1e-3)
    self.assertAllClose(random_principal_point, principal_point, rtol=1e-3)

  @parameterized.parameters(
      ((2,), (2,)),
      ((2, 2), (2, 2)),
      ((None, 2), (None, 2)),
  )
  def test_matrix_from_intrinsics_exception_not_raised(self, *shapes):
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
      (((0., 0.), (0., 0.)), (((0., 0., 0.), (0., 0., 0.), (0., 0., 1.)),)),
      (((1., 2.), (3., 4.)), (((1., 0., 3.), (0., 2., 4.), (0., 0., 1.)),)),
  )
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

    focal, principal_point = perspective.intrinsics_from_matrix(random_matrix)
    matrix = perspective.matrix_from_intrinsics(focal, principal_point)

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
