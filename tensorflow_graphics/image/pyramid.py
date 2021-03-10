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
"""This module implements image pyramid functionalities.

More details about image pyramids can be found on [this page.]
(https://en.wikipedia.org/wiki/Pyramid_(image_processing))
"""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import numpy as np
from six.moves import range
import tensorflow as tf

from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


def _downsample(image, kernel):
  """Downsamples the image using a convolution with stride 2.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.

  Returns:
    A tensor of shape `[B, H_d, W_d, C]`, where `H_d` and `W_d` are the height
    and width of the downsampled image.

  """
  return tf.nn.conv2d(
      input=image, filters=kernel, strides=[1, 2, 2, 1], padding="SAME")


def _binomial_kernel(num_channels, dtype=tf.float32):
  """Creates a 5x5 binomial kernel.

  Args:
    num_channels: The number of channels of the image to filter.
    dtype: The type of an element in the kernel.

  Returns:
    A tensor of shape `[5, 5, num_channels, num_channels]`.
  """
  kernel = np.array((1., 4., 6., 4., 1.), dtype=dtype.as_numpy_dtype)
  kernel = np.outer(kernel, kernel)
  kernel /= np.sum(kernel)
  kernel = kernel[:, :, np.newaxis, np.newaxis]
  return tf.constant(kernel, dtype=dtype) * tf.eye(num_channels, dtype=dtype)


def _build_pyramid(image, sampler, num_levels):
  """Creates the different levels of the pyramid.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    sampler: A function to execute for each level (_upsample or _downsample).
    num_levels: The number of levels.

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the image for the level i.
  """
  kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
  levels = [image]
  for _ in range(num_levels):
    image = sampler(image, kernel)
    levels.append(image)
  return levels


def _split(image, kernel):
  """Splits the image into high and low frequencies.

  This is achieved by smoothing the input image and substracting the smoothed
  version from the input.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.

  Returns:
    A tuple of two tensors of shape `[B, H, W, C]` and `[B, H_d, W_d, C]`, where
    the first one contains the high frequencies of the image and the second one
    the low frequencies. `H_d` and `W_d` are the height and width of the
    downsampled low frequency image.

  """
  low = _downsample(image, kernel)
  high = image - _upsample(low, kernel, tf.shape(input=image))
  return high, low


def _upsample(image, kernel, output_shape=None):
  """Upsamples the image using a transposed convolution with stride 2.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    kernel: A tensor of shape `[H_k, W_k, C, C]`, where `H_k` and `W_k` are the
      height and width of the kernel.
    output_shape: The output shape.

  Returns:
    A tensor of shape `[B, H_u, W_u, C]`, where `H_u` and `W_u` are the height
    and width of the upsampled image.
  """
  if output_shape is None:
    output_shape = tf.shape(input=image)
    output_shape = (output_shape[0], output_shape[1] * 2, output_shape[2] * 2,
                    output_shape[3])
  return tf.nn.conv2d_transpose(
      image,
      kernel * 4.0,
      output_shape=output_shape,
      strides=[1, 2, 2, 1],
      padding="SAME")


def downsample(image, num_levels, name="pyramid_downsample"):
  """Generates the different levels of the pyramid (downsampling).

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_downsample".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the downsampled image for the
    level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _downsample, num_levels)


def merge(levels, name="pyramid_merge"):
  """Merges the different levels of the pyramid back to an image.

  Args:
    levels: A list containing tensors of shape `[B, H_i, W_i, C]`, where `B` is
      the batch size, H_i and W_i are the height and width of the image for the
      level i, and `C` the number of channels of the image.
    name: A name for this op that defaults to "pyramid_merge".

  Returns:
      A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.

  Raises:
    ValueError: If the shape of the elements of `levels` is not supported.
  """
  with tf.name_scope(name):
    levels = [tf.convert_to_tensor(value=level) for level in levels]

    for index, level in enumerate(levels):
      shape.check_static(
          tensor=level, tensor_name="level {}".format(index), has_rank=4)

    image = levels[-1]
    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    for level in reversed(levels[:-1]):
      image = _upsample(image, kernel, tf.shape(input=level)) + level
    return image


def split(image, num_levels, name="pyramid_split"):
  """Generates the different levels of the pyramid.

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_split".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the image for the level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    kernel = _binomial_kernel(tf.shape(input=image)[3], dtype=image.dtype)
    low = image
    levels = []
    for _ in range(num_levels):
      high, low = _split(low, kernel)
      levels.append(high)
    levels.append(low)
    return levels


def upsample(image, num_levels, name="pyramid_upsample"):
  """Generates the different levels of the pyramid (upsampling).

  Args:
    image: A tensor of shape `[B, H, W, C]`, where `B` is the batch size, `H`
      the height of the image, `W` the width of the image, and `C` the number of
      channels of the image.
    num_levels: The number of levels to generate.
    name: A name for this op that defaults to "pyramid_upsample".

  Returns:
    A list containing `num_levels` tensors of shape `[B, H_i, W_i, C]`, where
    `H_i` and `W_i` are the height and width of the upsampled image for the
    level i.

  Raises:
    ValueError: If the shape of `image` is not supported.
  """
  with tf.name_scope(name):
    image = tf.convert_to_tensor(value=image)

    shape.check_static(tensor=image, tensor_name="image", has_rank=4)

    return _build_pyramid(image, _upsample, num_levels)

# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
