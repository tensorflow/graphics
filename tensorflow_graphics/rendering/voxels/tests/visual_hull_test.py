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
"""Tests for visual hull voxel rendering."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.rendering.voxels import visual_hull
from tensorflow_graphics.rendering.voxels.tests import test_helpers
from tensorflow_graphics.util import test_case


class VisualHullTest(test_case.TestCase):

  @parameterized.parameters(
      ((8, 16, 6, 1),),
      ((12, 8, 16, 6, 3),),
  )
  def test_render_shape_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(visual_hull.render, shape)

  @parameterized.parameters(
      ("must have a rank greater than 3", (3,)),
      ("must have a rank greater than 3", (16, 6, 3)),
  )
  def test_render_shape_exception_raised(self, error_msg, *shape):
    """Tests that the shape exception is raised."""
    self.assert_exception_is_raised(visual_hull.render, error_msg, shape)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_render_jacobian_random(self):
    """Tests the Jacobian of render."""
    voxels_init = test_helpers.generate_random_test_voxels_render()

    self.assert_jacobian_is_correct_fn(visual_hull.render, [voxels_init])

  def test_render_preset(self):
    """Checks that render returns the expected value."""
    x_voxels_init, y_images_init = test_helpers.generate_preset_test_voxels_visual_hull_render(
    )

    voxels = tf.convert_to_tensor(value=x_voxels_init)
    y_images = tf.convert_to_tensor(value=y_images_init)

    y = visual_hull.render(voxels)

    self.assertAllClose(y_images, y)


if __name__ == "__main__":
  test_case.main()
