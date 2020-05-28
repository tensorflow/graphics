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
"""Tests for grid."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.util import test_case


class GridTest(test_case.TestCase):

  @parameterized.parameters(
      (((1,), (1,), (1,)), (tf.float32, tf.float32, tf.int32)),
      (((1, 1), (1, 1), (1,)), (tf.float32, tf.float32, tf.int32)),
  )
  def test_generate_exception_not_raised(self, shapes, dtypes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_not_raised(grid.generate, shapes, dtypes)

  @parameterized.parameters(
      ("starts must have a rank greater than 0", (), (None,), (None,)),
      ("stops must have a rank greater than 0", (None,), (), (None,)),
      ("nums must have a rank of 1", (None,), (None,), ()),
      ("Not all batch dimensions are identical.", (1,), (0,), (1,)),
      ("Not all batch dimensions are identical.", (0,), (1,), (1,)),
      ("must have the same number of dimensions", (1,), (1,), (0,)),
  )
  def test_generate_exception_raised(self, error_msg, *shapes):
    """Tests that the shape exceptions are not raised."""
    self.assert_exception_is_raised(grid.generate, error_msg, shapes)

  @parameterized.parameters(
      (((-1.,), (1.,), (3,)), (((-1.,), (0.,), (1.,)),)),
      ((((-1.,), (-1.,)), ((1.,), (1.,)), (1,)), ((((-1.,),), ((-1.,),)),)),
  )
  def test_generate_preset(self, test_inputs, test_outputs):
    """Test the uniform grid generation using fix test cases."""
    self.assert_output_is_correct(
        grid.generate, test_inputs, test_outputs, tile=False)

  def test_generate_random(self):
    """Test the uniform grid generation."""
    starts = np.array((0., 0.), dtype=np.float32)
    stops = np.random.randint(1, 10, size=(2))
    nums = stops + 1
    stops = stops.astype(np.float32)

    g = grid.generate(starts, stops, nums)
    shape = nums.tolist() + [2]
    xv, yv = np.meshgrid(range(shape[0]), range(shape[1]), indexing="ij")
    gt = np.stack((xv, yv), axis=-1).astype(np.float32)

    self.assertAllClose(g, gt)


if __name__ == "__main__":
  test_case.main()
