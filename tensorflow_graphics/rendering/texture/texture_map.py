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
"""This module implements texture mapping.

Texture mapping is the process of fetching values (e.g. colors) from an image or
tensor based on the (u, v) coordinates at each pixel (please see
https://en.wikipedia.org/wiki/Texture_mapping for more information on
texturing). You can find how the uv-coordinates map to textures exactly in the
documentation of the ops.
"""
from typing import Text

import tensorflow as tf
from tensorflow_graphics.image import transformer
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias as tfg_type


def map_texture(uv_map: tfg_type.TensorLike,
                texture_image: tfg_type.TensorLike,
                tiling: bool = False,
                interpolation_method: Text = 'bilinear',
                name: Text = 'map_texture') -> tf.Tensor:
  """Maps the texture texture_image using uv_map.

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

  When a uv-coordinate corresponds to a point on the texture image that does
  not coincide with any of the pixel centers, bilinear interpolation is applied
  to compute the color value.

  If the aspect ratio of the texture is not 1, the texture is compressed to fit
  into a square.

  Args:
    uv_map: A tensor of shape `[A1, ..., An, H, W, 2]` containing the uv
      coordinates with range [0, 1], height H and width W.
    texture_image: A tensor of shape `[H', W', C]` containing the texture to be
      mapped with height H', width W', and number of channels C of the texture
      image.
    tiling: If enabled, the texture is tiled so that any uv value outside the
      range [0, 1] will be mapped to the tiled texture. E.g. if uv-coordinate is
      (0, 1.5), it is mapped to (0, 0.5). When tiling, the aspect ratio of the
      texture image should be 1.
    interpolation_method: A string specifying which interplolation method to
      use. It can be 'bilinear' or 'nearest' for bilinear or nearest neighbor
      interpolation, respectively.
    name: A name for this op that defaults to "map_texture".

  Returns:
    A tensor of shape `[A1, ..., An, H, W, C]` containing the interpolated
    values.
  """
  with tf.name_scope(name):

    uv_map = tf.convert_to_tensor(value=uv_map, dtype=tf.float32)
    texture_image = tf.convert_to_tensor(value=texture_image, dtype=tf.float32)

    shape.check_static(
        tensor=uv_map,
        tensor_name='uv_map',
        has_rank_greater_than=3,
        has_dim_equals=(-1, 2))

    shape.check_static(
        tensor=texture_image, tensor_name='texture_image', has_rank=3)

    if interpolation_method == 'bilinear':
      resampling_type = transformer.ResamplingType.BILINEAR
    elif interpolation_method == 'nearest':
      resampling_type = transformer.ResamplingType.NEAREST
    else:
      raise ValueError('The interpolation_method is not recognized. It should '
                       'either be bilinear or nearest.')

    texture_image_shape = tf.shape(input=texture_image)
    texture_height = tf.cast(texture_image_shape[-3], tf.float32)
    texture_width = tf.cast(texture_image_shape[-2], tf.float32)
    texture_num_channels = texture_image_shape[-1]

    query_points = tf.reshape(uv_map, [1, -1, 2])
    if tiling:
      query_points_u = query_points[..., 0] * texture_width
      query_points_v = query_points[..., 1] * texture_height
      query_points_u = tf.math.floormod(query_points_u, texture_width)
      query_points_v = tf.math.floormod(query_points_v, texture_height)
      query_points_v = texture_height - query_points_v

      # Pad texture with the first/last row/column for interpolating with
      # periodic boundary conditions.
      padded_texture = tf.concat(
          (tf.expand_dims(texture_image[:, -1, :], axis=1), texture_image,
           tf.expand_dims(texture_image[:, 0, :], axis=1)),
          axis=1)
      padded_texture = tf.concat(
          (tf.expand_dims(padded_texture[-1, :, :], axis=0), padded_texture,
           tf.expand_dims(padded_texture[0, :, :], axis=0)),
          axis=0)
      texture_image = padded_texture

      query_points_u = query_points_u + 1
      query_points_v = query_points_v + 1
    else:
      query_points_u = query_points[..., 0] * texture_width
      query_points_v = (1 - query_points[..., 1]) * texture_height

    query_points = tf.stack((query_points_u, query_points_v), axis=-1)

    interpolated = transformer.sample(
        image=tf.expand_dims(texture_image, axis=0),
        warp=query_points,
        resampling_type=resampling_type)

    # pylint: disable=bad-whitespace
    interpolated_shape = tf.concat((tf.shape(input=uv_map)[:-1],
                                    tf.convert_to_tensor(value=[
                                        texture_num_channels,
                                    ])),
                                   axis=0)
    # pylint: enable=bad-whitespace

    return tf.reshape(interpolated, interpolated_shape)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
