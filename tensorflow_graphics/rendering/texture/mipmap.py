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
"""This module implements mip-mapping.

Mip-mapping is texture mapping with a multi resolution texture. The original
texture is downsampled at multiple resolutions. These downsampled images are
blended at each pixel to reduce aliasing artifacts. You may find more
information on mipmapping on https://en.wikipedia.org/wiki/Mipmap.

In practice, you may use mip-mapping the same way as you use standard texture
mapping. You will see reduced aliasing artifacts when there are edges or other
high frequency details.

Texture mapping is the process of fetching values (e.g. colors) from an image or
tensor based on the (u, v) coordinates at each pixel (please see
https://en.wikipedia.org/wiki/Texture_mapping for more information on
texturing). You can find how the uv-coordinates map to textures exactly in the
documentation of the ops.
"""
from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

from typing import Optional, Sequence, Text

from six.moves import range
import tensorflow as tf
from tensorflow_graphics.rendering.texture import texture_map
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias as tfg_type


def map_texture(uv_map: tfg_type.TensorLike,
                texture_image: Optional[tfg_type.TensorLike] = None,
                mipmap_images: Optional[Sequence[tfg_type.TensorLike]] = None,
                num_mipmap_levels: Optional[int] = 5,
                tiling: bool = False,
                name: Text = 'mipmap_map_texture') -> tf.Tensor:
  """Maps the texture texture_image using uv_map with mip-mapping.

  The convention we use is that the origin in the uv-space is at (0, 0), u
  corresponds to the x-axis, v corresponds to the y-axis, and the color for each
  pixel is associated with the center of the corresponding pixel. E.g. if we
  have a texture [[1, 2], [3, 4]], then the uv-coordinates that correspond to
  the values 1, 2, 3, and 4 are (0.25, 0.75), (0.75, 0.75), (0.25, 0.25),
  (0.75, 0.25), respectively. You can see that the v-axis starts from the bottom
  of the texture image as would be in cartesian coordinates and that by
  multiplying the uv-coordinates with the length of the texture image, 2, you
  can recover the pixel centers in this case, e.g. (0.25, 0.25) * 2 = (0.5, 0.5)
  corresponds to the bottom-left pixel color that is 3.

  If the aspect ratio of the texture is not 1, the texture is compressed to fit
  into a square.

  Note that all shapes are assumed to be static.

  Args:
    uv_map: A tensor of shape `[A1, ..., An, H, W, 2]` containing the uv
      coordinates with range [0, 1], height H and width W.
    texture_image: An optional tensor of shape `[H', W', C]` containing the
      texture to be mapped with height H', width W', and number of channels C of
      the texture image.
    mipmap_images: Optional list containing the original texture image at
      multiple resolutions starting from the highest resolution. If not
      provided, these are computed from texture_image and hence, texture_image
      needs to be provided in that case. If both texture_image and mipmap_images
      are provided, mipmap_images are used and texture_image is ignored.
    num_mipmap_levels: An optional integer specifying the number of mipmap
      levels. Each level is computed by downsampling by a factor of two. If
      mipmap_images is provided, num_mipmap_levels is comptued as its length.
    tiling: If enabled, the texture is tiled so that any uv value outside the
      range [0, 1] will be mapped to the tiled texture. E.g. if uv-coordinate is
      (0, 1.5), it is mapped to (0, 0.5). When tiling, the aspect ratio of the
      texture image should be 1.
    name: A name for this op that defaults to "mipmap_map_texture".

  Returns:
    A tensor of shape `[A1, ..., An, H, W, C]` containing the interpolated
    values.

  Raises:
    ValueError: If texture_image is too small for the mipmap images to be
      constructed.
  """
  with tf.name_scope(name):

    if mipmap_images is None and texture_image is None:
      raise ValueError('Either texture_image or mipmap_images should be '
                       'provided.')
    # Shape checks
    shape.check_static(
        tensor=uv_map,
        tensor_name='uv_map',
        has_rank_greater_than=3,
        has_dim_equals=(-1, 2))

    if mipmap_images is not None:
      num_mipmap_levels = len(mipmap_images)
      for idx, mipmap_image in enumerate(mipmap_images):
        shape.check_static(
            tensor=mipmap_image, tensor_name=f'mipmap_image{idx}', has_rank=3)

    if texture_image is not None:
      shape.check_static(
          tensor=texture_image, tensor_name='texture_image', has_rank=3)

    # Initializations
    uv_map = tf.convert_to_tensor(value=uv_map, dtype=tf.float32)

    if mipmap_images is not None:
      for mipmap_image in mipmap_images:
        mipmap_image = tf.convert_to_tensor(value=mipmap_image)

      texture_shape = mipmap_images[0].get_shape().as_list()
      texture_height, texture_width = texture_shape[-3:-1]
    elif texture_image is not None:
      texture_image = tf.convert_to_tensor(
          value=texture_image, dtype=tf.float32)
      texture_shape = texture_image.get_shape().as_list()
      texture_height, texture_width = texture_shape[-3:-1]

      if (texture_height / 2**num_mipmap_levels < 1 or
          texture_width / 2**num_mipmap_levels < 1):
        raise ValueError('The size of texture_image '
                         f'({texture_height}, {texture_width}) '
                         'is too small for the provided number of mipmap '
                         f'levels ({num_mipmap_levels}).')

      mipmap_images = [texture_image]
      for idx in range(num_mipmap_levels - 1):
        previous_size = mipmap_images[idx].shape.as_list()
        current_height = tf.floor(previous_size[0] / 2)
        current_width = tf.floor(previous_size[1] / 2)
        mipmap_images.append(
            tf.image.resize(mipmap_images[idx],
                            [current_height, current_width]))

    # Computing the per-pixel mipmapping level and level indices
    uv_shape = uv_map.get_shape().as_list()
    uv_batch_dimensions = uv_shape[:-3]
    uv_height, uv_width = uv_shape[-3:-1]
    uv_map = tf.reshape(uv_map, (-1, uv_height, uv_width, 2))

    ddx, ddy = tf.image.image_gradients(uv_map)
    max_derivative = tf.math.maximum(
        tf.reduce_max(input_tensor=tf.math.abs(ddx), axis=-1),
        tf.reduce_max(input_tensor=tf.math.abs(ddy), axis=-1))
    max_derivative = max_derivative * [texture_height, texture_width]
    max_derivative = tf.math.maximum(max_derivative, 1.0)

    mipmap_level = tf.experimental.numpy.log2(max_derivative)
    mipmap_indices = tf.stack(
        (tf.math.floor(mipmap_level), tf.math.ceil(mipmap_level)), axis=-1)
    mipmap_level = mipmap_level - mipmap_indices[..., 0]
    mipmap_indices = tf.clip_by_value(mipmap_indices, 0, num_mipmap_levels - 1)
    mipmap_indices = tf.cast(mipmap_indices, dtype=tf.int32)

    # Map texture for each level and stack the results
    mapped_texture_stack = []
    for mipmap_image in mipmap_images:
      mapped_texture_stack.append(
          texture_map.map_texture(
              uv_map=uv_map, texture_image=mipmap_image, tiling=tiling))
    mapped_texture_stack = tf.stack(mapped_texture_stack, axis=-2)

    # Gather the lower and higher mipmapped textures
    mapped_texture_lower = tf.gather(
        mapped_texture_stack, mipmap_indices[..., 0], batch_dims=3, axis=3)
    mapped_texture_higher = tf.gather(
        mapped_texture_stack, mipmap_indices[..., 1], batch_dims=3, axis=3)

    # Interpolate with the mipmap_level
    # Note: If the original mipmap index is above
    # num_mipmap_levels - 1, after flooring, ceiling, and clipping to the range
    # 0 to num_mipmap_levels - 1, mipmap_indices[..., 0] and
    # mipmap_indices[..., 1] will be the same and hence mapped_texture_lower and
    # mapped_texture_higher will be the same, resulting in the correct
    # non-interpolated value coming from level num_mipmap_levels - 1.
    mipmap_level = tf.expand_dims(mipmap_level, axis=-1)
    mapped_texture = mapped_texture_lower * (
        1.0 - mipmap_level) + mapped_texture_higher * mipmap_level

    return tf.reshape(
        mapped_texture,
        uv_batch_dimensions + [uv_height, uv_width, texture_shape[-1]])


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
