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
"""Tests for image transformation functionalities."""

from absl.testing import parameterized
import numpy as np
import tensorflow as tf
from tensorflow_addons import image as tfa_image

from tensorflow_graphics.image import transformer
from tensorflow_graphics.util import test_case


class TransformerTest(test_case.TestCase, parameterized.TestCase):

  @parameterized.parameters(
      ((None, 1, 2, None), (None, 3, 3)),
      ((1, 2, 3, 4), (1, 3, 3)),
  )
  def test_perspective_transform_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""

    self.assert_exception_is_not_raised(transformer.perspective_transform,
                                        shape)

  @parameterized.parameters(
      ("must have a rank of 4.", (1, 1, 1), (1, 3, 3)),
      ("must have a rank of 3.", (1, 1, 1, 1), (3, 3)),
      ("Not all batch dimensions are identical.", (1, 1, 1, 1), (2, 3, 3)),
  )
  def test_perspective_transform_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""

    self.assert_exception_is_raised(transformer.perspective_transform,
                                    error_msg, shape)

  @parameterized.parameters(
      (tf.float32, "NEAREST"),
      (tf.float64, "NEAREST"),
      (tf.float32, "BILINEAR"),
      (tf.float64, "BILINEAR"),
  )
  def test_perspective_transform_half_integer_centers_preset(
      self, dtype, interpolation):
    """Tests that we can reproduce the results of tf.image.resize."""
    image = tf.constant(
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)),
        dtype=dtype)
    scale = 3
    transformation = tf.constant(
        ((1.0 / scale, 0.0, 0.0), (0.0, 1.0 / scale, 0.0), (0.0, 0.0, 1.0)),
        dtype=dtype)

    image_shape = tf.shape(input=image)
    image_resized_shape = image_shape * scale
    image = image[tf.newaxis, ..., tf.newaxis]
    transformation = transformation[tf.newaxis, ...]
    image_resized = tf.image.resize(
        image,
        size=image_resized_shape,
        method=tf.image.ResizeMethod.NEAREST_NEIGHBOR
        if interpolation == "NEAREST" else tf.image.ResizeMethod.BILINEAR)
    image_transformed = transformer.perspective_transform(
        image,
        transformation,
        resampling_type=transformer.ResamplingType.NEAREST
        if interpolation == "NEAREST" else transformer.ResamplingType.BILINEAR,
        border_type=transformer.BorderType.DUPLICATE,
        output_shape=image_resized_shape)

    self.assertAllClose(image_resized, image_transformed)

  @parameterized.parameters(
      (tf.float32, "NEAREST"),
      (tf.float64, "NEAREST"),
      (tf.float32, "BILINEAR"),
      (tf.float64, "BILINEAR"),
  )
  def test_perspective_transform_integer_centers_preset(self, dtype,
                                                        interpolation):
    """Tests that we can reproduce the results of tfa_image.transform."""
    image = tf.constant(
        ((1.0, 2.0, 3.0), (4.0, 5.0, 6.0), (7.0, 8.0, 9.0), (10.0, 11.0, 12.0)),
        dtype=dtype)
    scale = 3
    transformation = tf.constant(
        ((1.0 / scale, 0.0, 0.0), (0.0, 1.0 / scale, 0.0), (0.0, 0.0, 1.0)),
        dtype=dtype)

    image_shape = tf.shape(input=image)
    image_resized_shape = image_shape * scale
    image = image[tf.newaxis, ..., tf.newaxis]
    transformation = transformation[tf.newaxis, ...]
    image_resized = tfa_image.transform(
        tf.cast(image, tf.float32),
        tf.cast(
            tfa_image.transform_ops.matrices_to_flat_transforms(transformation),
            tf.float32),
        interpolation=interpolation,
        output_shape=image_resized_shape)
    image_transformed = transformer.perspective_transform(
        image,
        transformation,
        resampling_type=transformer.ResamplingType.NEAREST
        if interpolation == "NEAREST" else transformer.ResamplingType.BILINEAR,
        pixel_type=transformer.PixelType.INTEGER,
        output_shape=image_resized_shape)

    self.assertAllClose(image_resized, image_transformed)

  def test_perspective_transform_jacobian_random(self):
    """Tests the Jacobian of the transform function."""
    tensor_shape = np.random.randint(2, 4, size=4)
    image_init = np.random.uniform(0.0, 1.0, size=tensor_shape.tolist())
    transformation_init = np.random.uniform(
        0.0, 1.0, size=(tensor_shape[0], 3, 3))

    self.assert_jacobian_is_correct_fn(
        lambda x: transformer.perspective_transform(x, transformation_init),
        [image_init])
    self.assert_jacobian_is_correct_fn(
        lambda x: transformer.perspective_transform(image_init, x),
        [transformation_init])

  @parameterized.parameters(
      ((None, 1, 2, None), (None, 2)),
      ((1, 3, 2, 4), (1, 2)),
  )
  def test_sample_exception_not_raised(self, *shape):
    """Tests that the shape exceptions are not raised."""

    self.assert_exception_is_not_raised(transformer.sample, shape)

  @parameterized.parameters(
      ("must have a rank of 4.", (1, 1, 1), (1, 2)),
      ("must have a rank greater than 1", (1, 1, 1, 1), (2,)),
      ("Not all batch dimensions are identical.", (1, 1, 1, 1), (2, 2)),
  )
  def test_sample_exception_raised(self, error_msg, *shape):
    """Tests that the shape exceptions are properly raised."""

    self.assert_exception_is_raised(transformer.sample, error_msg, shape)


if __name__ == "__main__":
  test_case.main()
