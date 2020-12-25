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
"""Tests for emission absorption voxel rendering."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.rendering.voxels import emission_absorption
from tensorflow_graphics.rendering.voxels.tests import test_helpers
from tensorflow_graphics.util import test_case


class EmissionAbsorptionTest(test_case.TestCase):

  @parameterized.parameters(
      (0, (8, 16, 6, 1)),
      (1, (12, 8, 16, 6, 3)),
  )
  def test_render_shape_exception_not_raised(self, axis, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(emission_absorption.render,
                                        shape,
                                        axis=axis)

  @parameterized.parameters(
      ("must have a rank greater than 3", 2, (3,)),
      ("must have a rank greater than 3", 2, (16, 6, 3)),
      ("'axis' needs to be 0, 1 or 2", 5, (8, 16, 6, 1)),
  )
  def test_render_shape_exception_raised(self, error_msg, axis, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(emission_absorption.render,
                                    error_msg,
                                    shape,
                                    axis=axis)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_render_jacobian_random(self):
    """Tests the Jacobian of render."""
    voxels_init = test_helpers.generate_random_test_voxels_render()
    absorption_factor_init = np.float64(np.random.uniform(low=0.1, high=2.0))
    cell_size_init = np.float64(np.random.uniform(low=0.1, high=2.0))

    self.assert_jacobian_is_correct_fn(
        emission_absorption.render,
        [voxels_init, absorption_factor_init, cell_size_init],
        atol=1e-4)

  def test_render_preset(self):
    """Checks that render returns the expected value."""
    x_voxels_init, y_images_init = test_helpers.generate_preset_test_voxels_emission_absorption_render(
    )

    voxels = tf.convert_to_tensor(value=x_voxels_init)
    y_images = tf.convert_to_tensor(value=y_images_init)

    y = emission_absorption.render(voxels, absorption_factor=0.1, cell_size=0.1)

    self.assertAllClose(y_images, y)


if __name__ == "__main__":
  test_case.main()
