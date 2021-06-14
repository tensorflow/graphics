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
"""Tests for mipmap."""

import functools

from absl.testing import parameterized
import tensorflow as tf
from tensorflow_graphics.rendering.texture import mipmap
from tensorflow_graphics.util import test_case


class MipmapTest(test_case.TestCase):

  @parameterized.parameters(
      (False, 2, (1, 2, 2, 2), (4, 4, 3)),
      (False, 3, (3, 3, 2, 2, 2), (16, 16, 1)),
      (True, 2, (3, 3, 2, 2, 2), (8, 8, 1)),
  )
  def test_map_texture_exception_not_raised(self, tiling, num_mipmap_levels,
                                            *shapes):
    map_texture_fn = functools.partial(
        mipmap.map_texture,
        tiling=tiling,
        num_mipmap_levels=num_mipmap_levels,
    )
    self.assert_exception_is_not_raised(func=map_texture_fn, shapes=shapes)

  @parameterized.parameters(
      (
          [[[0.25, 0.25], [0.25, 0.75]], [[0.75, 0.25], [0.75, 0.75]]],
          [[[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
           [[3.5, 5.5], [11.5, 13.5]]],
          False,
          [[[[11.5, 115], [3.5, 35]], [[13.5, 135], [5.5, 55]]]],
      ),)
  def test_map_texture_interpolates_correctly_mipmap_images(
      self, uv_map, mipmap_images_list, tiling, interpolated_gt):
    uv_map = tf.convert_to_tensor(value=uv_map)
    uv_map = tf.expand_dims(uv_map, 0)

    mipmap_images = []
    for mipmap_image in mipmap_images_list:
      mipmap_image = tf.convert_to_tensor(value=mipmap_image)
      mipmap_image = tf.stack((mipmap_image, mipmap_image * 10), -1)
      mipmap_images.append(mipmap_image)

    interpolated = mipmap.map_texture(
        uv_map=uv_map,
        texture_image=None,
        mipmap_images=mipmap_images,
        num_mipmap_levels=len(mipmap_images),
        tiling=tiling)

    interpolated_gt = tf.convert_to_tensor(
        value=interpolated_gt, dtype=tf.float32)
    self.assertAllClose(interpolated, interpolated_gt)

  @parameterized.parameters(
      (
          [[[0.25, 0.25], [0.25, 0.75]], [[0.75, 0.25], [0.75, 0.75]]],
          [[1.0, 2, 3, 4], [5, 6, 7, 8], [9, 10, 11, 12], [13, 14, 15, 16]],
          2,
          False,
          [[[[11.5, 115], [3.5, 35]], [[13.5, 135], [5.5, 55]]]],
      ),)
  def test_map_texture_interpolates_correctly_texture_image(
      self, uv_map, texture_image, num_mipmap_levels, tiling, interpolated_gt):
    uv_map = tf.convert_to_tensor(value=uv_map)
    uv_map = tf.expand_dims(uv_map, 0)

    texture_image = tf.convert_to_tensor(value=texture_image)
    texture_image = tf.stack((texture_image, texture_image * 10), -1)

    interpolated = mipmap.map_texture(
        uv_map=uv_map,
        texture_image=texture_image,
        mipmap_images=None,
        num_mipmap_levels=num_mipmap_levels,
        tiling=tiling)

    interpolated_gt = tf.convert_to_tensor(
        value=interpolated_gt, dtype=tf.float32)
    self.assertAllClose(interpolated, interpolated_gt)


if __name__ == '__main__':
  tf.test.main()
