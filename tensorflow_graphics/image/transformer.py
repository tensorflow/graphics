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
"""This module implements image transformation functionalities."""

import enum
from typing import Optional

import tensorflow as tf
from tensorflow_addons import image as tfa_image

from tensorflow_graphics.geometry.representation import grid
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


class ResamplingType(enum.Enum):
  NEAREST = 0
  BILINEAR = 1


class BorderType(enum.Enum):
  ZERO = 0
  DUPLICATE = 1


class PixelType(enum.Enum):
  INTEGER = 0
  HALF_INTEGER = 1


def sample(image: type_alias.TensorLike,
           warp: type_alias.TensorLike,
           resampling_type: ResamplingType = ResamplingType.BILINEAR,
           border_type: BorderType = BorderType.ZERO,
           pixel_type: PixelType = PixelType.HALF_INTEGER,
           name: Optional[str] = None) -> tf.Tensor:
  """Samples an image at user defined coordinates.

  Note:
    The warp maps target to source. In the following, A1 to An are optional
    batch dimensions.

  Args:
    image: A tensor of shape `[B, H_i, W_i, C]`, where `B` is the batch size,
      `H_i` the height of the image, `W_i` the width of the image, and `C` the
      number of channels of the image.
    warp: A tensor of shape `[B, A_1, ..., A_n, 2]` containing the x and y
      coordinates at which sampling will be performed. The last dimension must
      be 2, representing the (x, y) coordinate where x is the index for width
      and y is the index for height.
   resampling_type: Resampling mode. Supported values are
     `ResamplingType.NEAREST` and `ResamplingType.BILINEAR`.
    border_type: Border mode. Supported values are `BorderType.ZERO` and
      `BorderType.DUPLICATE`.
    pixel_type: Pixel mode. Supported values are `PixelType.INTEGER` and
      `PixelType.HALF_INTEGER`.
    name: A name for this op. Defaults to "sample".

  Returns:
    Tensor of sampled values from `image`. The output tensor shape
    is `[B, A_1, ..., A_n, C]`.

  Raises:
    ValueError: If `image` has rank != 4. If `warp` has rank < 2 or its last
    dimension is not 2. If `image` and `warp` batch dimension does not match.
  """
  with tf.name_scope(name or "sample"):
    image = tf.convert_to_tensor(image, name="image")
    warp = tf.convert_to_tensor(warp, name="warp")

    shape.check_static(image, tensor_name="image", has_rank=4)
    shape.check_static(
        warp,
        tensor_name="warp",
        has_rank_greater_than=1,
        has_dim_equals=(-1, 2))
    shape.compare_batch_dimensions(
        tensors=(image, warp), last_axes=0, broadcast_compatible=False)

    if pixel_type == PixelType.HALF_INTEGER:
      warp -= 0.5

    if resampling_type == ResamplingType.NEAREST:
      warp = tf.math.round(warp)

    if border_type == BorderType.DUPLICATE:
      image_size = tf.cast(tf.shape(image)[1:3], dtype=warp.dtype)
      height, width = tf.unstack(image_size, axis=-1)
      warp_x, warp_y = tf.unstack(warp, axis=-1)
      warp_x = tf.clip_by_value(warp_x, 0.0, width - 1.0)
      warp_y = tf.clip_by_value(warp_y, 0.0, height - 1.0)
      warp = tf.stack((warp_x, warp_y), axis=-1)

    return tfa_image.resampler(image, warp)


def perspective_transform(
    image: type_alias.TensorLike,
    transform_matrix: type_alias.TensorLike,
    output_shape: Optional[type_alias.TensorLike] = None,
    resampling_type: ResamplingType = ResamplingType.BILINEAR,
    border_type: BorderType = BorderType.ZERO,
    pixel_type: PixelType = PixelType.HALF_INTEGER,
    name: Optional[str] = None,
) -> tf.Tensor:
  """Applies a projective transformation to an image.

  The projective transformation is represented by a 3 x 3 matrix
  [[a0, a1, a2], [b0, b1, b2], [c0, c1, c2]], mapping a point `[x, y]` to a
  transformed point
  `[x', y'] = [(a0 x + a1 y + a2) / k, (b0 x + b1 y + b2) / k]`, where
  `k = c0 x + c1 y + c2`.

  Note:
      The transformation matrix maps target to source by transforming output
      points to input points.

  Args:
    image: A tensor of shape `[B, H_i, W_i, C]`, where `B` is the batch size,
      `H_i` the height of the image, `W_i` the width of the image, and `C` the
      number of channels of the image.
    transform_matrix: A tensor of shape `[B, 3, 3]` containing projective
      transform matrices. The transformation maps target to source by
      transforming output points to input points.
    output_shape: The heigh `H_o` and width `W_o` output dimensions after the
      transform. If None, output is the same size as input image.
    resampling_type: Resampling mode. Supported values are
      `ResamplingType.NEAREST` and `ResamplingType.BILINEAR`.
    border_type: Border mode. Supported values are `BorderType.ZERO` and
      `BorderType.DUPLICATE`.
    pixel_type: Pixel mode. Supported values are `PixelType.INTEGER` and
      `PixelType.HALF_INTEGER`.
    name: A name for this op. Defaults to "perspective_transform".

  Returns:
    A tensor of shape `[B, H_o, W_o, C]` containing transformed images.

  Raises:
    ValueError: If `image` has rank != 4. If `transform_matrix` has rank < 3 or
    its last two dimensions are not 3. If `image` and `transform_matrix` batch
    dimension does not match.
  """
  with tf.name_scope(name or "perspective_transform"):
    image = tf.convert_to_tensor(image, name="image")
    transform_matrix = tf.convert_to_tensor(
        transform_matrix, name="transform_matrix")
    output_shape = tf.shape(
        image)[-3:-1] if output_shape is None else tf.convert_to_tensor(
            output_shape, name="output_shape")

    shape.check_static(image, tensor_name="image", has_rank=4)
    shape.check_static(
        transform_matrix,
        tensor_name="transform_matrix",
        has_rank=3,
        has_dim_equals=((-1, 3), (-2, 3)))
    shape.compare_batch_dimensions(
        tensors=(image, transform_matrix),
        last_axes=0,
        broadcast_compatible=False)

    dtype = image.dtype
    zero = tf.cast(0.0, dtype)
    height, width = tf.unstack(output_shape, axis=-1)
    warp = grid.generate(
        starts=(zero, zero),
        stops=(tf.cast(width, dtype) - 1.0, tf.cast(height, dtype) - 1.0),
        nums=(width, height))
    warp = tf.transpose(warp, perm=[1, 0, 2])

    if pixel_type == PixelType.HALF_INTEGER:
      warp += 0.5

    padding = [[0, 0] for _ in range(warp.shape.ndims)]
    padding[-1][-1] = 1
    warp = tf.pad(
        tensor=warp, paddings=padding, mode="CONSTANT", constant_values=1.0)

    warp = warp[..., tf.newaxis]
    transform_matrix = transform_matrix[:, tf.newaxis, tf.newaxis, ...]
    warp = tf.linalg.matmul(transform_matrix, warp)
    warp = warp[..., 0:2, 0] / warp[..., 2, :]

    return sample(image, warp, resampling_type, border_type, pixel_type)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
