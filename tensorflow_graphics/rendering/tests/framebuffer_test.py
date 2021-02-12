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
# Lint as: python3
"""Tests for tensorflow_graphics.rendering.framebuffer."""

import tensorflow as tf
from tensorflow_graphics.rendering import framebuffer as fb
from tensorflow_graphics.util import test_case


class FramebufferTest(test_case.TestCase):

  def test_initialize_rasterized_attribute_and_derivatives_with_wrong_rank(
      self):
    with self.assertRaisesRegex(
        ValueError, "Expected value and derivatives to be of the same rank"):
      fb.RasterizedAttribute(
          tf.ones([4, 4, 1]), tf.ones([4, 3]), tf.ones([3, 4, 4, 5, 5]))

  def test_initialize_rasterized_attribute_with_wrong_rank(self):
    with self.assertRaisesRegex(ValueError,
                                "Expected input value to be rank 4"):
      fb.RasterizedAttribute(tf.ones([4, 4, 1]))

  def test_initialize_rasterized_attribute_with_wrong_shapes(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Expected all input shapes to be the same"):
      fb.RasterizedAttribute(
          tf.ones([2, 4, 4, 1]), tf.ones([2, 4, 3, 1]), tf.ones([2, 4, 3, 5]))

  def test_initialize_framebuffer_with_wrong_rank(self):
    with self.assertRaisesRegex(ValueError,
                                "Expected all inputs to have the same rank"):
      fb.Framebuffer(
          fb.RasterizedAttribute(tf.ones([1, 4, 4, 1])), tf.ones([4, 3]),
          tf.ones([3, 4, 4, 5, 5]), tf.ones([3, 4, 4, 5, 5]))

  def test_initialize_framebuffer_with_wrong_shapes(self):
    with self.assertRaisesRegex(tf.errors.InvalidArgumentError,
                                "Expected all input shapes to be the same"):
      fb.Framebuffer(
          fb.RasterizedAttribute(tf.ones([2, 4, 4, 3])), tf.ones([2, 4, 4, 1]),
          tf.ones([2, 4, 4, 3]), tf.ones([2, 4, 4, 1]),
          {"an_attr": fb.RasterizedAttribute(tf.ones([2, 4, 3, 4]))})


if __name__ == "__main__":
  tf.test.main()
