#Copyright 2019 Google LLC
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

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow.compat.v1 as tf

from tensorflow_graphics.image.color_space import linear_rgb
from tensorflow_graphics.image.color_space import srgb
from tensorflow_graphics.util import test_case


class LinearRGBTest(test_case.TestCase):

  def test_cycle_srgb_linear_rgb_srgb_for_random_input(self):
    """Tests loop from sRGB to linear RGB and back for random inputs."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    srgb_input = np.random.uniform(size=tensor_shape + [3])
    linear_output = linear_rgb.from_srgb(srgb_input)
    srgb_recovered = srgb.from_linear_rgb(linear_output)

    self.assertAllClose(srgb_input, srgb_recovered)

  @parameterized.parameters(
      (((0., 0.5, 1.), (0.0404, 0.04045, 0.0405)),
       ((0., 0.214041, 1.), (0.003127, 0.003131, 0.003135))),)
  def test_from_srgb_preset(self, test_inputs, test_outputs):
    """Tests conversion from sRGB to linear RGB space for preset inputs."""
    self.assert_output_is_correct(linear_rgb.from_srgb, (test_inputs,),
                                  (test_outputs,))

  def test_from_srgb_jacobian_random(self):
    """Tests the Jacobian of the from_srgb function for random inputs."""
    tensor_size = np.random.randint(3)
    tensor_shape = np.random.randint(1, 10, size=(tensor_size)).tolist()
    srgb_random_init = np.random.uniform(size=tensor_shape + [3])
    # Wrap this in identity because some assert_* ops look at the constant
    # tensor value and mark it as unfeedable.
    srgb_random = tf.identity(tf.convert_to_tensor(value=srgb_random_init))
    linear_random = linear_rgb.from_srgb(srgb_random)
    self.assert_jacobian_is_correct(srgb_random, srgb_random_init,
                                    linear_random)

  @parameterized.parameters((np.array((0., 0.01, 0.02)),), (np.array(
      (0.05, 0.06, 1.)),), (np.array((0.01, 0.04, 0.06)),))
  def test_from_srgb_jacobian_preset(self, inputs_init):
    """Tests the Jacobian of the from_srgb function for preset inputs."""
    # Wrap this in identity because some assert_* ops look at the constant
    # tensor value and mark it as unfeedable.
    inputs_tensor = tf.identity(tf.convert_to_tensor(value=inputs_init))
    outputs = linear_rgb.from_srgb(inputs_tensor)
    self.assert_jacobian_is_correct(inputs_tensor, inputs_init, outputs)

  @parameterized.parameters(
      ((3,),),
      ((None, None, None, 3),),
  )
  def test_from_srgb_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(linear_rgb.from_srgb, shape)

  @parameterized.parameters(
      ("must have a rank greater than 0", ()),
      ("must have exactly 3 dimensions in axis -1", (2, 3, 4)),
  )
  def test_from_srgb_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    self.assert_exception_is_raised(linear_rgb.from_srgb, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
