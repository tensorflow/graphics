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
"""Util functions for rasterization tests."""

import os

import numpy as np
import tensorflow as tf

from tensorflow_graphics.geometry.transformation import look_at
from tensorflow_graphics.rendering.camera import perspective
from tensorflow_graphics.util import shape


def make_perspective_matrix(image_width=None, image_height=None):
  """Generates perspective matrix for a given image size.

  Args:
    image_width: int representing image width.
    image_height: int representing image height.

  Returns:
    Perspective matrix, tensor of shape [4, 4].

  Note: Golden tests require image size to be fixed and equal to the size of
  golden image examples. The rest of the camera parameters are set such that
  resulting image will be equal to the baseline image.
  """

  field_of_view = (40 * np.math.pi / 180,)
  near_plane = (0.01,)
  far_plane = (10.0,)
  return perspective.right_handed(field_of_view,
                                  (float(image_width) / float(image_height),),
                                  near_plane, far_plane)


def make_look_at_matrix(
    camera_origin=(0.0, 0.0, 0.0), look_at_point=(0.0, 0.0, 0.0)):
  """Shortcut util function to creat model-to-eye matrix for tests."""
  camera_up = (0.0, 1.0, 0.0)
  return look_at.right_handed(camera_origin, look_at_point, camera_up)


def compare_images(test_case,
                   baseline_image,
                   image,
                   max_outlier_fraction=0.005,
                   pixel_error_threshold=0.04):
  """Compares two image arrays.

  The comparison is soft: the images are considered identical if fewer than
  max_outlier_fraction of the pixels differ by more than pixel_error_threshold
  of the full color value.

  Differences in JPEG encoding can produce pixels with pretty large variation,
  so by default we use 0.04 (4%) for pixel_error_threshold and 0.005 (0.5%) for
  max_outlier_fraction.

  Args:
    test_case: test_case.TestCase instance this util function is used in.
    baseline_image: tensor of shape [batch, height, width, channels] containing
      the baseline image.
    image: tensor of shape [batch, height, width, channels] containing the
      result image.
    max_outlier_fraction: fraction of pixels that may vary by more than the
      error threshold. 0.005 means 0.5% of pixels. Number of outliers are
      computed and compared per image.
    pixel_error_threshold: pixel values are considered to differ if their
      difference exceeds this amount. Range is 0.0 - 1.0.

  Returns:
    Tuple of a boolean and string error message. Boolean indicates
    whether images are close to each other or not. Error message contains
    details of two images mismatch.
  """
  tf.assert_equal(baseline_image.shape, image.shape)
  if baseline_image.dtype != image.dtype:
    return False, ("Image types %s and %s do not match" %
                   (baseline_image.dtype, image.dtype))
  shape.check_static(
      tensor=baseline_image, tensor_name="baseline_image", has_rank=4)
  # Flatten height, width and channels dimensions since we're interested in
  # error per image.
  image_height, image_width = image.shape[1:3]
  baseline_image = tf.reshape(baseline_image, [baseline_image.shape[0]] + [-1])
  image = tf.reshape(image, [image.shape[0]] + [-1])
  abs_diff = tf.abs(baseline_image - image)
  outliers = tf.math.greater(abs_diff, pixel_error_threshold)
  num_outliers = tf.math.reduce_sum(tf.cast(outliers, tf.int32))
  perc_outliers = num_outliers / (image_height * image_width)
  error_msg = "{:.2%} pixels are not equal to baseline image pixels.".format(
      test_case.evaluate(perc_outliers) * 100.0)
  return test_case.evaluate(perc_outliers < max_outlier_fraction), error_msg


def load_baseline_image(filename, image_shape=None):
  """Loads baseline image and makes sure it is of the right shape.

  Args:
    filename: file name of the image to load.
    image_shape: expected shape of the image.

  Returns:
    tf.Tensor with baseline image
  """
  image_path = tf.compat.v1.resource_loader.get_path_to_datafile(
      os.path.join("test_data", filename))
  file = tf.io.read_file(image_path)
  baseline_image = tf.cast(tf.image.decode_image(file), tf.float32) / 255.0
  baseline_image = tf.expand_dims(baseline_image, axis=0)
  if image_shape is not None:
    # Graph-mode requires image shape to be known in advance.
    baseline_image = tf.ensure_shape(baseline_image, image_shape)
  return baseline_image
