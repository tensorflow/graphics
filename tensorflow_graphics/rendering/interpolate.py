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
"""Interpolation utilities for attributes."""

from typing import Iterable, Optional, Union

import tensorflow as tf

from tensorflow_graphics.rendering import framebuffer as fb


def interpolate_vertex_attribute(
    attribute: tf.Tensor,
    framebuffer: fb.Framebuffer,
    background_value: Optional[Union[tf.Tensor, Iterable[float]]] = None
) -> fb.RasterizedAttribute:
  """Interpolate a single vertex attribute across the input framebuffer.

  Args:
    attribute: 2-D or 3-D vertex attribute Tensor with shape [batch,
      num_vertices, num_channels] or [num_vertices, num_channels].
    framebuffer: Framebuffer to interpolate across. Expected to contain
      barycentrics, vertex_ids, and foreground_mask.
    background_value: 1-D Tensor (or convertible value) with shape
      [num_channels] containing the value to use for background pixels. If None,
      defaults to zero.

  Returns:
    A RasterizedAttribute containing the per-pixel interpolated values.
  """
  vertex_ids = tf.reshape(framebuffer.vertex_ids,
                          [framebuffer.batch_size, framebuffer.pixel_count, 3])

  num_channels = tf.compat.dimension_value(tf.shape(attribute)[-1])

  # Creates indices with each pixel's clip-space triangle's extrema (the pixel's
  # 'corner points') ids to look up the attributes for each pixel's triangle.
  # Handles batched or unbatched attributes. In either case, corner_attribute
  # will be shaped [batch_size, pixel_count, 3, num_channels] (with
  # batch_size = 1 if input is unbatched).
  if len(attribute.shape) == 3:
    corner_attribute = tf.gather(attribute, vertex_ids, batch_dims=1)
  else:
    vertex_ids = tf.reshape(vertex_ids, (-1, 3))
    corner_attribute = tf.gather(attribute, vertex_ids, batch_dims=0)
    corner_attribute = tf.reshape(
        corner_attribute,
        (framebuffer.batch_size, framebuffer.pixel_count, 3, num_channels))

  # Computes the pixel attributes by interpolating the known attributes at the
  # corner points of the triangle interpolated with the barycentric
  # coordinates.
  reshaped_barycentrics = tf.reshape(
      framebuffer.barycentrics.value,
      (framebuffer.batch_size, framebuffer.pixel_count, 3, 1))
  weighted_vertex_attribute = tf.multiply(corner_attribute,
                                          reshaped_barycentrics)
  summed_attribute = tf.reduce_sum(weighted_vertex_attribute, axis=2)
  out_shape = (framebuffer.batch_size, framebuffer.height, framebuffer.width,
               num_channels)
  attribute_image = tf.reshape(summed_attribute, out_shape)
  if background_value is None:
    background_value = tf.zeros((num_channels), dtype=attribute.dtype)
  else:
    background_value = tf.convert_to_tensor(
        background_value, dtype=attribute.dtype)
  attribute_image = (
      framebuffer.foreground_mask * attribute_image +
      framebuffer.background_mask * background_value)
  return fb.RasterizedAttribute(value=attribute_image, d_dx=None, d_dy=None)
