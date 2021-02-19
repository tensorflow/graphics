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
"""Tests for radiance ray rendering."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.rendering.volumetric import ray_radiance
from tensorflow_graphics.util import test_case


def generate_random_test_ray_render():
  """Generates random test for the voxels rendering functions."""
  batch_shape = np.random.randint(1, 3)
  n_rays = np.random.randint(1, 512)
  random_ray_values = np.random.uniform(size=[batch_shape] + [n_rays, 4])
  random_ray_dists = random_ray_values[..., -1]
  return random_ray_values, random_ray_dists


class VolumetricTest(test_case.TestCase):

  @parameterized.parameters(
      ((6, 4), (6,)),
      ((8, 16, 44, 4), (8, 16, 44)),
      ((12, 8, 16, 22, 4), (12, 8, 16, 22)),
      ((32, 32, 256, 4), (32, 32, 256)),
      ((32, 32, 256, 4), (1, 1, 256)),
      ((32, 32, 256, 4), (256,)),
  )
  def test_render_shape_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(ray_radiance.compute_radiance, shapes)

  @parameterized.parameters(
      ("must have a rank greater than 1", ((4,), (3,))),
      ("must have exactly 4 dimensions in axis -1", ((44, 5), (44, 1))),
      ("Not all batch dimensions are broadcast-compatible.",
       ((32, 32, 256, 4,), (32, 16, 256,))),
      ("must have the same number of dimensions",
       ((32, 32, 128, 4,), (32, 32, 555,))),
  )
  def test_render_shape_exception_raised(self, error_msg, shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(ray_radiance.compute_radiance,
                                    error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_render_jacobian_random(self):
    """Tests the Jacobian of render."""
    point_values, point_distance = generate_random_test_ray_render()
    self.assert_jacobian_is_correct_fn(
        lambda x: ray_radiance.compute_radiance(x, point_distance)[0],
        [point_values])
    self.assert_jacobian_is_correct_fn(
        lambda x: ray_radiance.compute_radiance(point_values, x)[0],
        [point_distance])

  def test_render_preset(self):
    """Checks that render returns the expected value."""

    image_rays = np.zeros((128, 128, 64, 4))
    image_rays[32:96, 32:96, 16:32, :] = 1
    distances = np.zeros((128, 128, 64)) + 1.5
    target_image = np.zeros((128, 128, 3))
    target_image[32:96, 32:96, :] = 1
    rendered_image, *_ = ray_radiance.compute_radiance(image_rays, distances)
    self.assertAllClose(rendered_image, target_image)


if __name__ == "__main__":
  test_case.main()
