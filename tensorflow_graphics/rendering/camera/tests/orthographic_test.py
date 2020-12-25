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
"""Tests for orthographic camera functionalities."""

from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.rendering.camera import orthographic
from tensorflow_graphics.util import test_case


class OrthographicTest(test_case.TestCase):

  @parameterized.parameters(
      ((3,),),
      ((None, 3),),
  )
  def test_project_exception_not_exception_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(orthographic.project, shapes)

  @parameterized.parameters(
      ("must have exactly 3 dimensions in axis -1", (1,)),
      ("must have exactly 3 dimensions in axis -1", (2,)),
      ("must have exactly 3 dimensions in axis -1", (4,)),
  )
  def test_project_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(orthographic.project, error_msg, shape)

  def test_project_jacobian_random(self):
    """Test the Jacobian of the project function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_3d_init = np.random.uniform(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(orthographic.project, [point_3d_init])

  def test_project_random(self):
    """Test the project function using random 3d points."""
    point_3d = np.random.uniform(size=(100, 3))

    pred = orthographic.project(point_3d)

    self.assertAllEqual(pred, point_3d[:, 0:2])

  @parameterized.parameters(
      ((2,),),
      ((None, 2),),
  )
  def test_ray_exception_not_exception_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(orthographic.ray, shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (1,)),
      ("must have exactly 2 dimensions in axis -1", (3,)),
  )
  def test_ray_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(orthographic.ray, error_msg, shape)

  def test_ray_jacobian_random(self):
    """Test the Jacobian of the ray function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_2d_init = np.random.uniform(size=tensor_shape + [2])

    self.assert_jacobian_is_correct_fn(orthographic.ray, [point_2d_init])

  def test_ray_random(self):
    """Test the ray function using random 2d points."""
    point_2d = np.random.uniform(size=(100, 2))

    pred = orthographic.ray(point_2d)
    gt = np.tile((0.0, 0.0, 1.0), (100, 1))

    self.assertAllEqual(pred, gt)

  @parameterized.parameters(
      ((2,), (1,)),
      ((None, 2), (None, 1)),
      ((2, 2), (2, 1)),
  )
  def test_unproject_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(orthographic.unproject, shapes)

  @parameterized.parameters(
      ("must have exactly 2 dimensions in axis -1", (1,), (1,)),
      ("must have exactly 2 dimensions in axis -1", (3,), (1,)),
      ("must have exactly 1 dimensions in axis -1", (2,), (2,)),
      ("Not all batch dimensions are identical.", (3, 2), (2, 1)),
  )
  def test_unproject_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(orthographic.unproject, error_msg, shape)

  def test_unproject_jacobian_random(self):
    """Test the Jacobian of the unproject function."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    point_2d_init = np.random.uniform(size=tensor_shape + [2])
    depth_init = np.random.uniform(size=tensor_shape + [1])

    self.assert_jacobian_is_correct_fn(orthographic.unproject,
                                       [point_2d_init, depth_init])

  def test_unproject_random(self):
    """Test the unproject function using random 2d points."""
    point_2d = np.random.uniform(size=(100, 2))
    depth = np.random.uniform(size=(100, 1))

    pred = orthographic.unproject(point_2d, depth)

    self.assertAllEqual(pred[:, 0:2], point_2d)
    self.assertAllEqual(pred[:, 2], np.squeeze(depth))


if __name__ == "__main__":
  test_case.main()
