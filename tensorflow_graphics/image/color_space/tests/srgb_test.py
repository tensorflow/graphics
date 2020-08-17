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
"""Tests for srgb."""

from absl.testing import flagsaver
from absl.testing import parameterized
import numpy as np

from tensorflow_graphics.image.color_space import linear_rgb
from tensorflow_graphics.image.color_space import srgb
from tensorflow_graphics.util import test_case


class SrgbTest(test_case.TestCase):

  def test_cycle_linear_rgb_srgb_linear_rgb_for_random_input(self):
    """Tests loop from linear RGB to sRGB and back for random inputs."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    linear_input = np.random.uniform(size=tensor_shape + [3])

    srgb_output = srgb.from_linear_rgb(linear_input)
    linear_reverse = linear_rgb.from_srgb(srgb_output)

    self.assertAllClose(linear_input, linear_reverse)

  @parameterized.parameters(
      (((0., 0.5, 1.), (0.00312, 0.0031308, 0.00314)),
       ((0., 0.735357, 1.), (0.04031, 0.04045, 0.040567))),)
  def test_from_linear_rgb_preset(self, test_inputs, test_outputs):
    """Tests conversion from linear to sRGB color space for preset inputs."""
    self.assert_output_is_correct(srgb.from_linear_rgb, (test_inputs,),
                                  (test_outputs,))

  def test_from_linear_rgb_jacobian_random(self):
    """Tests the Jacobian of the from_linear_rgb function for random inputs."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    linear_random_init = np.random.uniform(size=tensor_shape + [3])

    self.assert_jacobian_is_correct_fn(srgb.from_linear_rgb,
                                       [linear_random_init])

  @parameterized.parameters((np.array((0., 0.001, 0.002)),), (np.array(
      (0.004, 0.005, 1.)),), (np.array((0.00312, 0.004, 0.00314)),))
  @flagsaver.flagsaver(tfg_add_asserts_to_graph=False)
  def test_from_linear_rgb_jacobian_preset(self, inputs_init):
    """Tests the Jacobian of the from_linear_rgb function for preset inputs."""
    self.assert_jacobian_is_correct_fn(srgb.from_linear_rgb, [inputs_init])

  @parameterized.parameters(
      ((3,),),
      ((None, None, None, 3),),
  )
  def test_from_linear_rgb_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(srgb.from_linear_rgb, shape)

  @parameterized.parameters(
      ("must have a rank greater than 0", ()),
      ("must have exactly 3 dimensions in axis -1", (2, 3, 4)),
  )
  def test_from_linear_rgb_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(srgb.from_linear_rgb, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
