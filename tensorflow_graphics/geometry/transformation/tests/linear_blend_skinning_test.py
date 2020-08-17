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
"""Tests for linear blend skinning."""

# pylint: disable=line-too-long

from absl.testing import flagsaver
from absl.testing import parameterized
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import linear_blend_skinning
from tensorflow_graphics.geometry.transformation.tests import test_helpers
from tensorflow_graphics.util import test_case


class LinearBlendSkinningTest(test_case.TestCase):

  # pyformat: disable
  @parameterized.parameters(
      ((3,), (7,), (7, 3, 3), (7, 3)),
      ((None, 3), (None, 9), (None, 9, 3, 3), (None, 9, 3)),
      ((7, 1, 3), (1, 4, 11), (5, 11, 3, 3), (1, 11, 3)),
      ((7, 4, 3), (4, 11), (11, 3, 3), (11, 3)),
      ((3,), (5, 4, 11), (11, 3, 3), (11, 3)),
  )
  # pyformat: enable
  def test_blend_exception_not_raised(self, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(linear_blend_skinning.blend, shapes)

  # pyformat: disable
  @parameterized.parameters(
      ("points must have exactly 3 dimensions in axis -1",
       (None,), (7,), (7, 3, 3), (7, 3)),
      ("bone_rotations must have a rank greater than 2", (3,), (7,), (3, 3), (3,)),
      ("bone_rotations must have exactly 3 dimensions in axis -1",
       (3,), (7,), (7, 3, None), (7, 3)),
      ("bone_rotations must have exactly 3 dimensions in axis -2",
       (3,), (7,), (7, None, 3), (7, 3)),
      ("bone_translations must have a rank greater than 1", (3,), (7,), (7, 3, 3), (3,)),
      ("bone_translations must have exactly 3 dimensions in axis -1",
       (3,), (7,), (7, 3, 3), (7, None)),
      (r"Tensors \[\'skinning_weights\', \'bone_rotations\'\] must have the same number of dimensions in axes",
       (3,), (9,), (7, 3, 3), (9, 3)),
      (r"Tensors \[\'skinning_weights\', \'bone_translations\'\] must have the same number of dimensions in axes",
       (3,), (9,), (9, 3, 3), (7, 3)),
      ("Not all batch dimensions are broadcast-compatible",
       (2, 3, 3), (3, 1, 7), (7, 3, 3), (7, 3)),
      ("Not all batch dimensions are broadcast-compatible",
       (2, 3, 3), (2, 1, 7), (3, 7, 3, 3), (2, 7, 3)),
  )
  # pyformat: enable
  def test_blend_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(linear_blend_skinning.blend, error_msg,
                                    shapes)

  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_blend_jacobian_random(self):
    """Test the Jacobian of the blend function."""
    (x_points_init, x_weights_init, x_rotations_init,
     x_translations_init) = test_helpers.generate_random_test_lbs_blend()

    self.assert_jacobian_is_correct_fn(
        linear_blend_skinning.blend,
        [x_points_init, x_weights_init, x_rotations_init, x_translations_init])

  def test_blend_preset(self):
    """Checks that blend returns the expected value."""
    (x_points_init, x_weights_init, x_rotations_init, x_translations_init,
     y_blended_points_init) = test_helpers.generate_preset_test_lbs_blend()

    x_points = tf.convert_to_tensor(value=x_points_init)
    x_weights = tf.convert_to_tensor(value=x_weights_init)
    x_rotations = tf.convert_to_tensor(value=x_rotations_init)
    x_translations = tf.convert_to_tensor(value=x_translations_init)
    y_blended_points = tf.convert_to_tensor(value=y_blended_points_init)

    y = linear_blend_skinning.blend(x_points, x_weights, x_rotations,
                                    x_translations)

    self.assertAllClose(y_blended_points, y)


if __name__ == "__main__":
  test_case.main()
