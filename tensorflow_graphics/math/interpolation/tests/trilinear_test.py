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
"""Tests for trilinear interpolation."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.geometry.transformation import rotation_matrix_3d
from tensorflow_graphics.math.interpolation import trilinear
from tensorflow_graphics.util import test_case


def _sampling_points_from_grid(grid_size, dtype=tf.float64):
  """Returns a tensor of shape `[M, 3]`, with M the number of sampling points."""
  sampling_points = grid.generate((-1.0, -1.0, -1.0), (1.0, 1.0, 1.0),
                                  grid_size)
  sampling_points = tf.cast(sampling_points, dtype)
  return tf.reshape(sampling_points, [-1, 3])


def _transpose_last_two_dims(sampling_points):
  axes = [i for i in range(len(sampling_points.shape))]
  axes[-1], axes[-2] = axes[-2], axes[-1]
  sampling_points = tf.transpose(a=sampling_points, perm=axes)
  return sampling_points


def _sampling_points_in_volume(sampling_points, voxel_size):
  """Transforms the sampling points from [-1, 1] to [0, voxel_size]."""
  voxel_size = tf.convert_to_tensor(value=voxel_size)
  max_size = tf.cast(voxel_size - 1, sampling_points.dtype)
  return 0.5 * ((sampling_points + 1) * max_size)


ANGLE_90 = np.array((np.pi / 2.,))


def _get_random_voxel_grid(voxel_size):
  return np.random.uniform(size=voxel_size)


def _get_random_sampling_points(sampling_points_size, max_grid_dim):
  random_grid = np.random.randint(0, max_grid_dim, size=sampling_points_size)
  return random_grid.astype(np.float64)


def _generate_voxels_horizontal_plane(voxel_size):
  voxels = np.zeros(voxel_size)
  mid_x = int(np.floor(voxel_size[0] / 2))
  voxels[mid_x, :, :, :] = 1
  if voxel_size[0] % 2 == 0:
    voxels[mid_x - 1, :, :, :] = 1
  return voxels


def _generate_voxels_vertical_plane(voxel_size):
  voxels = np.zeros(voxel_size)
  mid_y = int(np.floor(voxel_size[1] / 2))
  voxels[:, mid_y, :, :] = 1
  if voxel_size[1] % 2 == 0:
    voxels[:, mid_y - 1, :, :] = 1
  return voxels


def _generate_voxel_cube(dims, plane_orientation=None):
  if plane_orientation == "horizontal":
    voxels_no_batch = _generate_voxels_horizontal_plane(dims[-4:])
  elif plane_orientation == "vertical":
    voxels_no_batch = _generate_voxels_vertical_plane(dims[-4:])
  else:
    voxels_no_batch = np.zeros(dims[-4:])

  voxels = np.zeros(dims)
  voxels[..., :, :, :, :] = voxels_no_batch
  return voxels


class TrilinearTest(test_case.TestCase):

  @parameterized.parameters(
      ("must have a rank greater than 3", ((5, 5, 5), (125, 3))),
      ("must have a rank greater than 1", ((2, 5, 5, 5, 1), (3,))),
      ("must have exactly 3 dimensions in axis -1", ((2, 5, 5, 5, 1),
                                                     (2, 125, 4))),
      ("Not all batch dimensions are broadcast-compatible.",
       ((2, 2, 5, 5, 5, 1), (2, 3, 125, 3))),
  )
  def test_interpolate_exception_raised(self, error_msg, shapes):
    """Tests whether exceptions are raised for incompatible shapes."""
    self.assert_exception_is_raised(
        trilinear.interpolate, error_msg, shapes=shapes)

  @parameterized.parameters(
      ((5, 5, 5, 3), (125, 3)),
      ((2, 5, 5, 5, 3), (2, 125, 3)),
      ((2, 2, 5, 5, 5, 3), (2, 2, 15, 3)),
  )
  def test_interpolate_exception_not_raised(self, *shapes):
    """Tests whether exceptions are not raised for compatible shapes."""
    self.assert_exception_is_not_raised(trilinear.interpolate, shapes)

  def test_interpolation_values_preset(self):
    voxels = np.zeros((2, 2, 2, 1))
    voxels[(0, 1, 1, 0), (0, 0, 1, 1), (0, 0, 0, 0), 0] = 1
    sampling_points = np.array(((0, 0, 0), (0.5, 0, 0), (1.0, 0, 0),
                                (0., 0, 0.25), (0., 0, 0.5), (0., 0, 0.75),
                                (0., 0, 1.0), (0., 0, 2.0), (-1.0, -0.5, 0),
                                (0, 2, 1.5)))
    correct_values = np.array(
        ((1.0, 1.0, 1.0, 0.75, 0.5, 0.25, 0.0, 0.0, 1.0, 0.0),)).T

    self.assert_output_is_correct(
        trilinear.interpolate, (voxels, sampling_points), (correct_values,),
        tile=False)

  def test_interpolation_preset(self):
    """Tests whether interpolation results are correct."""
    batch_dim_size = np.random.randint(0, 4)
    batch_dims = list(np.random.randint(1, 10, size=batch_dim_size))
    cube_single_dim = np.random.randint(3, 10)
    cube_dims = [cube_single_dim, cube_single_dim, cube_single_dim]
    num_channels = [np.random.randint(1, 10)]
    combined_dims = batch_dims + cube_dims + num_channels
    voxels_in = _generate_voxel_cube(combined_dims, "horizontal")
    euler_angles = np.zeros(batch_dims + [3])
    euler_angles[..., 2] = np.pi / 2.
    voxels_out = _generate_voxel_cube(combined_dims, "vertical")

    transformation_matrix = rotation_matrix_3d.from_euler(euler_angles)
    grid_size = (cube_single_dim, cube_single_dim, cube_single_dim)
    sampling_points = _sampling_points_from_grid(grid_size)
    sampling_points = tf.matmul(transformation_matrix,
                                tf.transpose(a=sampling_points))
    sampling_points = _transpose_last_two_dims(sampling_points)
    sampling_points = _sampling_points_in_volume(sampling_points,
                                                 voxels_in.shape[-4:-1])
    voxels_out = tf.reshape(voxels_out,
                            batch_dims + [cube_single_dim**3] + num_channels)

    self.assert_output_is_correct(
        trilinear.interpolate, (voxels_in, sampling_points), (voxels_out,),
        tile=False)

  @parameterized.parameters(
      (1, 4, 4, 4, 1),
      (2, 4, 4, 4, 3),
      (3, 4, 4, 4, 3),
  )
  def test_interpolate_jacobian_random(self, bsize, height, width, depth,
                                       channels):
    """Tests whether jacobian is correct."""
    grid_3d_np = np.random.uniform(size=(bsize, height, width, depth, channels))
    sampling_points_np = np.zeros((bsize, height * width * depth, 3))
    sampling_points_np[:, :, 0] = np.arange(0, height * width * depth)

    self.assert_jacobian_is_correct_fn(
        lambda grid_3d: trilinear.interpolate(grid_3d, sampling_points_np),
        [grid_3d_np])


if __name__ == "__main__":
  test_case.main()
