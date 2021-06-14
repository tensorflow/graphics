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
"""Tests for texture_map."""

import functools

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_graphics.rendering.texture import texture_map
from tensorflow_graphics.util import test_case


class TextMapTest(test_case.TestCase):

  @parameterized.parameters(
      (False, (1, 2, 2, 2), (3, 2, 1)),
      (False, (3, 3, 2, 2, 2), (4, 2, 1)),
      (True, (3, 3, 2, 2, 2), (2, 2, 1)),
  )
  def test_map_texture_exception_not_raised(self, tiling, *shapes):
    map_texture_fn = functools.partial(texture_map.map_texture, tiling=tiling)
    self.assert_exception_is_not_raised(map_texture_fn, shapes)

  @parameterized.parameters(
      ([[0, 1], [2, 3], [4, 5]], [[0.25, 0.25], [0.75, 0.5]], [[
          0, 1.0 / 6.0
      ], [0.5, 0.5]], [[[2, -2], [4, -4]], [[3, -3], [2.5, -2.5]]], False),
      ([[0, 1], [2, 3]], [[0, 1], [0.75, -1.5]], [[0.25, 0.75], [1.25, 1.75]],
       [[[2.5, -2.5], [0.5, -0.5]], [[3, -3], [0.5, -0.5]]], True),
  )
  def test_map_texture_interpolates_correctly(self, texture_image, u_map, v_map,
                                              interpolated_gt, tiling):
    texture = tf.convert_to_tensor(value=texture_image, dtype=tf.float32)
    texture = tf.stack((
        texture,
        -texture,
    ), axis=-1)
    u_map = tf.convert_to_tensor(value=u_map, dtype=tf.float32)
    v_map = tf.convert_to_tensor(value=v_map, dtype=tf.float32)
    uv_map = tf.stack((
        u_map,
        v_map,
    ), axis=-1)
    uv_map = tf.expand_dims(uv_map, axis=0)
    interpolated = texture_map.map_texture(uv_map, texture, tiling)
    interpolated = tf.squeeze(interpolated)
    interpolated_gt = tf.convert_to_tensor(
        value=interpolated_gt, dtype=tf.float32)
    self.assertAllClose(interpolated, interpolated_gt)


if __name__ == "__main__":
  tf.test.main()
