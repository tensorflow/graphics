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
"""Tests for image pyramids."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf

from tensorflow_graphics.image import pyramid
from tensorflow_graphics.util import test_case

_NUM_LEVELS = 3


class PyramidTest(test_case.TestCase):

  @parameterized.parameters(
      ((1, 1, 1, 1),),
      ((None, None, None, None),),
  )
  def test_downsample_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    downsample = lambda image: pyramid.downsample(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_not_raised(downsample, shape)

  @parameterized.parameters(
      ("must have a rank of 4", ()),)
  def test_downsample_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    downsample = lambda image: pyramid.downsample(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_raised(downsample, error_msg, shape)

  def test_downsample_jacobian_random(self):
    """Tests the Jacobian for random inputs."""
    downsample = lambda image: pyramid.downsample(image, num_levels=_NUM_LEVELS)
    tensor_shape = np.random.randint(1, 5, size=4).tolist()
    image_random_init = np.random.uniform(size=tensor_shape)

    for level in range(_NUM_LEVELS):
      # We skip testing level = 0, which returns the image as is. In graph mode,
      # the gradient calculation fails when there are no nodes in the graph.
      if level == 0 and not tf.executing_eagerly():
        continue
      self.assert_jacobian_is_correct_fn(
          lambda x, level=level: downsample(x)[level], [image_random_init])

  @parameterized.parameters(
      (((0.,),), ((0.,),)),
      (((0., 0.), (0., 0.)), ((0.,),)),
      (((1., 1.), (1., 1.)), ((100. / 256.,),)),
  )
  def test_downsample_preset(self, image_high, image_low):
    """Tests that the downsample works as expected."""
    downsample = lambda image: pyramid.downsample(image, num_levels=1)
    image_high = tf.expand_dims(tf.expand_dims(image_high, axis=-1), axis=0)
    image_low = tf.expand_dims(tf.expand_dims(image_low, axis=-1), axis=0)

    pyr = downsample(image_high)

    with self.subTest(name="image_high"):
      self.assertAllClose(image_high, pyr[0])

    with self.subTest(name="image_low"):
      self.assertAllClose(image_low, pyr[1])

  @parameterized.parameters(
      ((1, 1, 1, 1),),
      ((None, None, None, None),),
  )
  def test_upsample_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    upsample = lambda image: pyramid.upsample(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_not_raised(upsample, shape)

  @parameterized.parameters(
      ("must have a rank of 4", ()),)
  def test_upsample_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    upsample = lambda image: pyramid.upsample(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_raised(upsample, error_msg, shape)

  def test_upsample_jacobian_random(self):
    """Tests the Jacobian for random inputs."""
    upsample = lambda image: pyramid.upsample(image, num_levels=_NUM_LEVELS)
    tensor_shape = np.random.randint(1, 5, size=4).tolist()
    image_random_init = np.random.uniform(size=tensor_shape)

    for level in range(_NUM_LEVELS):
      # We skip testing level = 0, which returns the image as is. In graph mode,
      # the gradient calculation fails when there are no nodes in the graph.
      if level == 0 and not tf.executing_eagerly():
        continue
      self.assert_jacobian_is_correct_fn(
          lambda x, level=level: upsample(x)[level], [image_random_init])

  @parameterized.parameters(
      (((0.,),), ((0., 0.), (0., 0.))),
      (((1.,),), ((64. / 256., 96. / 256.), (96. / 256., 144. / 256.))),
  )
  def test_upsample_preset(self, image_low, image_high):
    """Tests that the upsample works as expected."""
    upsample = lambda image: pyramid.upsample(image, num_levels=1)
    image_low = tf.expand_dims(tf.expand_dims(image_low, axis=-1), axis=0)
    image_high = tf.expand_dims(tf.expand_dims(image_high, axis=-1), axis=0)

    pyr = upsample(image_low)

    with self.subTest(name="image_low"):
      self.assertAllClose(image_low, pyr[0])
    with self.subTest(name="image_high"):
      self.assertAllClose(image_high, pyr[1])

  @parameterized.parameters(
      ((1, 1, 1, 1), (1, 1, 1, 1)),
      ((None, None, None, None), (None, None, None, None)),
  )
  def test_merge_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    merge = lambda l0, l1: pyramid.merge((l0, l1))

    self.assert_exception_is_not_raised(merge, shape)

  @parameterized.parameters(
      ("level 0 must have a rank of 4.", (), (1, 1, 1, 1)),
      ("level 1 must have a rank of 4.", (1, 1, 1, 1), ()),
  )
  def test_merge_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    merge = lambda l0, l1: pyramid.merge((l0, l1))

    self.assert_exception_is_raised(merge, error_msg, shape)

  @parameterized.parameters(
      (((0., 0.), (0., 0.)), ((0., 0.), (0., 0.)), ((0.,),)),
      (((1., 1.), (1., 1.)), ((1., 1.), (1., 1.)), ((0.,),)),
  )
  def test_merge_preset(self, image, image_high, image_low):
    """Tests that the merge function merges as expected."""
    image = tf.expand_dims(tf.expand_dims(image, axis=-1), axis=0)
    image_high = tf.expand_dims(tf.expand_dims(image_high, axis=-1), axis=0)
    image_low = tf.expand_dims(tf.expand_dims(image_low, axis=-1), axis=0)

    self.assertAllClose(image, pyramid.merge((image_high, image_low)))

  @parameterized.parameters(
      ((1, 1, 1, 1),),
      ((None, None, None, None),),
  )
  def test_split_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""
    split = lambda image: pyramid.split(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_not_raised(split, shape)

  @parameterized.parameters(
      ("must have a rank of 4.", ()),
      ("must have a rank of 4.", (1,)),
      ("must have a rank of 4.", (1, 1)),
      ("must have a rank of 4.", (1, 1, 1)),
  )
  def test_split_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""
    split = lambda image: pyramid.split(image, num_levels=_NUM_LEVELS)

    self.assert_exception_is_raised(split, error_msg, shape)

  def test_split_jacobian_random(self):
    """Tests the Jacobian for random inputs."""
    split = lambda image: pyramid.split(image, num_levels=_NUM_LEVELS)
    tensor_shape = np.random.randint(1, 5, size=4).tolist()
    image_random_init = np.random.uniform(size=tensor_shape)

    for level in range(_NUM_LEVELS):  # pylint: disable=unused-variable
      self.assert_jacobian_is_correct_fn(
          lambda x, level=level: split(x)[level], [image_random_init])

  @parameterized.parameters(
      (0,),
      (1,),
      (2,),
  )
  def test_split_merge_random(self, num_levels):
    """Tests that splitting and merging back can reproduce the input."""
    tensor_shape = np.random.randint(1, 5, size=4).tolist()
    image_random = np.random.uniform(size=tensor_shape)

    split = pyramid.split(image_random, num_levels=num_levels)
    merge = pyramid.merge(split)

    self.assertAllClose(image_random, merge)

  @parameterized.parameters(
      (((0., 0.), (0., 0.)), ((0., 0.), (0., 0.)), ((0.,),)),
      (((1., 0.), (0., 0.)), ((252. / 256., -6. / 256.),
                              (-6. / 256., -9. / 256.)), ((16. / 256.,),)),
  )
  def test_split_preset(self, image, image_high, image_low):
    """Tests that the split function splits the image as expected."""
    split = lambda image: pyramid.split(image, num_levels=1)
    image = tf.expand_dims(tf.expand_dims(image, axis=-1), axis=0)
    image_high = tf.expand_dims(tf.expand_dims(image_high, axis=-1), axis=0)
    image_low = tf.expand_dims(tf.expand_dims(image_low, axis=-1), axis=0)

    pyramid_split = split(image)

    with self.subTest(name="image_high"):
      self.assertAllClose(image_high, pyramid_split[0])

    with self.subTest(name="image_low"):
      self.assertAllClose(image_low, pyramid_split[1])


if __name__ == "__main__":
  test_case.main()
