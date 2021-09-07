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
"""Differentiable point splatting functions for rasterize-then-splat."""

import enum
import math
from typing import Callable, Dict, Tuple

import tensorflow as tf

from tensorflow_graphics.rendering import interpolate
from tensorflow_graphics.rendering import rasterization_backend
from tensorflow_graphics.rendering import utils
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def splat_at_pixel_centers(
    xyz_rgba: Tuple[tf.Tensor, tf.Tensor]
) -> Tuple[tf.Tensor, tf.Tensor, tf.Tensor]:
  """Splat a buffer of XYZ, RGBA samples onto a pixel grid of the same size.

  This is a specialized splatting function that takes a multi-layer buffer of
  screen-space XYZ positions and RGBA colors and splats each sample into
  a buffer of the same size, using a 3x3 Gaussian kernel of variance 0.25.
  The accumulated layers are then composited back-to-front.

  The specialized part is that the 3x3 kernel is always centered on the
  pixel-coordinates of the sample in the input buffer, *not* the XY position
  stored at that sample, but the weights are defined by using the XY position.
  Computing weights w.r.t. the XY positions, rather than the pixel-centers,
  allows gradients to flow from the output RGBA back to the XY positions. When
  used in rasterize-then-splat, XY positions will always coincide with the pixel
  centers, so the forward computation is the same as if the XY positions defined
  the position of the splat.

  When splatting, the Z of the splat is compared with the Z of the layers under
  the splat sample. The sample is accumulated into the layer with the Z closest
  to the Z of the splat itself.

  Args:
    xyz_rgba: a tuple of a float32 tensor of rasterized XYZ positions with shape
      [num_layers, height, width, 3] and a tensor of RGBA colors [num_layers,
      height, width, 4]. Passed as a tuple to support tf.vectorized_map.

  Returns:
    A tensor of shape [height, width, 4] with RGBA values, as well as
    [num_layers, height, width, 4] tensor of accumulated and normalized colors
    for visualization and debugging.
  """
  extra_accumulation_epsilon = 0.05
  xyz_layers, rgba_layers = xyz_rgba

  xyz_layers = tf.convert_to_tensor(xyz_layers)
  shape.check_static(tensor=xyz_layers, tensor_name='xyz_layers', has_rank=4)
  rgba_layers = tf.convert_to_tensor(rgba_layers)
  shape.check_static(tensor=rgba_layers, tensor_name='rgba_layers', has_rank=4)

  gaussian_variance = 0.5**2
  gaussian_exp_scale = -1.0 / (2 * gaussian_variance)

  # The normalization coefficient for the Gaussian must be computed with care so
  # that a full accumulation of neighboring splats adds up to 1.0 + epsilon. We
  # need to trigger normalization when the splats accumulate to a full surface
  # in order to avoid a spurious "spread-splats-to-darken-color" derivative, but
  # we do not want to normalize otherwise (e.g., at the boundary with the
  # background), so we use a small epsilon here.
  weight_sum = 0
  for u in (-1, 0, 1):
    for v in (-1, 0, 1):
      weight_sum += math.exp(gaussian_exp_scale * (u**2 + v**2))
  gaussian_coef = (1.0 + extra_accumulation_epsilon) / weight_sum

  # Accumulation buffers need a 1 pixel border because of 3x3 splats.
  padding = ((0, 0), (1, 1), (1, 1), (0, 0))
  # 3 accumulation layers (fg, surface, bg) of the same size as the image.
  accumulation_shape = [3, rgba_layers.shape[1], rgba_layers.shape[2]]
  accumulate_rgba = tf.pad(
      tf.zeros(accumulation_shape + [4], dtype=rgba_layers.dtype), padding)
  accumulate_weights = tf.pad(
      tf.zeros(accumulation_shape + [1], dtype=rgba_layers.dtype), padding)
  padded_center_z = tf.pad(xyz_layers[..., 2:3], padding, constant_values=1.0)
  surface_idx_uv_map = {}
  for u in (-1, 0, 1):
    for v in (-1, 0, 1):
      padding = [[max(v + 1, 0), abs(min(v - 1, 0))],
                 [max(u + 1, 0), abs(min(u - 1, 0))], [0, 0]]
      # Find the layer index of the first surface shared by the center of the
      # splat and the splat filter tap (i.e., sample position).
      # The first surface must appear as the top layer either at center or at
      # tap. The best matching Z between the top center layer and the tap
      # layers is compared against the best match between the center layers
      # and the top tap layer, and the pair of layers with smallest
      # difference in Z is the estimated surface.
      tap_z_layers = tf.pad(
          xyz_layers[..., 2:3], [[0, 0]] + padding, constant_values=1.0)
      dist_center_to_tap_layers = tf.abs(tap_z_layers - padded_center_z[0, ...])
      best_center_surface_idx = tf.argmin(dist_center_to_tap_layers, axis=0)
      best_center_surface_z = tf.reduce_min(dist_center_to_tap_layers, axis=0)
      dist_tap_to_center_layers = tf.abs(padded_center_z - tap_z_layers[0, ...])
      best_tap_surface_idx = tf.argmin(dist_tap_to_center_layers, axis=0)
      best_tap_surface_z = tf.reduce_min(dist_tap_to_center_layers, axis=0)
      # surface_idx is 0 if the first surface is the top layer for both center
      # and tap, a negative number (of layers) if the surface is occluded at
      # center, and a positive number if occluded at tap.
      surface_idx = tf.where(best_tap_surface_z < best_center_surface_z,
                             -best_tap_surface_idx, best_center_surface_idx)
      surface_idx_uv_map[(u, v)] = surface_idx

  num_layers = rgba_layers.shape[0]
  for l in range(num_layers):
    rgba = rgba_layers[l, ...]
    alpha = rgba_layers[l, :, :, 3:4]
    xyz = xyz_layers[l, ...]

    # Computes the offset from the splat to the pixel underneath the splat. Note
    # that in the forward pass, splat_to_center_pixel will always be zero to
    # within numerical precision, but it is necessary to define the filter tap
    # weights as a function of the splat position so derivatives will flow to
    # the splat. As the splat moves right, the pixel moves left relative to it,
    # so the splat position xy is negated here.
    splat_to_center_pixel = tf.floor(xyz[..., :2]) + (0.5, 0.5) - xyz[..., :2]

    for u in (-1, 0, 1):
      for v in (-1, 0, 1):
        splat_to_pixel = splat_to_center_pixel + (u, v)
        dist_sqr = tf.math.reduce_sum(splat_to_pixel**2, axis=-1, keepdims=True)
        tap_weights = alpha * gaussian_coef * tf.exp(
            gaussian_exp_scale * dist_sqr)

        tap_rgba = tap_weights * rgba

        padding = [[max(v + 1, 0), abs(min(v - 1, 0))],
                   [max(u + 1, 0), abs(min(u - 1, 0))], [0, 0]]
        tap_rgba = tf.pad(tap_rgba, padding)
        tap_weights = tf.pad(tap_weights, padding)
        surface_idx = surface_idx_uv_map[(u, v)]

        # If the current layer is in front of the surface, accumulate into fg.
        # If at the surface, accumulate into surf. If behind, accumulate into
        # bg. We use a masked accumulation here rather than a scatter, though
        # scatter could also work if there are a lot of layers.
        fg_mask = tf.cast(surface_idx > l, tf.float32)
        surf_mask = tf.cast(surface_idx == l, tf.float32)
        bg_mask = tf.cast(surface_idx < l, tf.float32)
        layer_mask = tf.stack((fg_mask, surf_mask, bg_mask), axis=0)

        masked_tap_rgba = tf.tile(
            tf.expand_dims(tap_rgba, axis=0), (3, 1, 1, 1)) * layer_mask
        masked_tap_weights = tf.tile(
            tf.expand_dims(tap_weights, axis=0), (3, 1, 1, 1)) * layer_mask

        accumulate_rgba += masked_tap_rgba
        accumulate_weights += masked_tap_weights

  # Normalize the accumulated colors by the accumulated weights. Normalization
  # only happens if the accumulate weights are > 1.0.
  accumulate_rgba = accumulate_rgba[:, 1:-1, 1:-1, :]
  accumulate_weights = accumulate_weights[:, 1:-1, 1:-1, :]
  normalization_scales = 1.0 / (tf.maximum(accumulate_weights - 1.0, 0.0) + 1.0)
  normalized_rgba = accumulate_rgba * normalization_scales

  # Composite the foreground, surface, and background layers back-to-front.
  output_rgba = normalized_rgba[-1, ...]
  for i in (2, 3):
    alpha = normalized_rgba[-i, :, :, 3:4]
    output_rgba = normalized_rgba[-i, ...] + (1.0 - alpha) * output_rgba

  return output_rgba, accumulate_rgba, normalized_rgba


def rasterize_then_splat(
    vertices: type_alias.TensorLike,
    triangles: type_alias.TensorLike,
    attributes: Dict[str, type_alias.TensorLike],
    view_projection_matrix: type_alias.TensorLike,
    image_size: Tuple[int, int],
    shading_function: Callable[[Dict[str, tf.Tensor]], tf.Tensor],
    num_layers=1,
    return_extra_buffers=False,
    backend: enum.Enum = rasterization_backend.RasterizationBackends.CPU,
    name='rasterize_then_splat'):
  """Rasterization with differentiable occlusion using rasterize-then-splat.

  Rasterizes the input triangles to produce surface point samples, applies
  a user-specified shading function, then splats the shaded point
  samples onto the pixel grid.

  The attributes are arbitrary per-vertex quantities (colors, normals, texture
  coordinates, etc.). The rasterization step interpolates these attributes
  across triangles to produce a dictionary of per-pixel interpolated attributes
  buffers with shapes `[H, W, K]` where `K` is the number of channels of the
  input attribute. This dictionary is passed to the user-provided
  `shading_function`, which performs shading and outputs a `[H, W, 4]`
  buffer of RGBA colors. The result of the shader is replaced with (0,0,0,0) for
  background pixels.

  In the common case that the attributes are RGBA vertex colors, the shading
  function may just pass the rasterized attributes through (i.e.,
  `shading_function = lambda x: x['color']` where `color` is an RGBA attribute).

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, 3]` containing batches of `V`
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[T, 3]` containing `T` triangles, each
      associated with 3 vertices from `vertices`.
    attributes: A dictionary of tensors, each of shape `[A1, ..., An, V, K]`
      containing batches of `V` vertices, each associated with `K`-dimensional
      attributes. `K` may vary by attribute.
    view_projection_matrix: A tensor of shape `[A1, ..., An, 4, 4]` containing
      batches of matrices used to transform vertices from model to clip
      coordinates.
    image_size: A tuple (height, width) containing the dimensions in pixels of
      the rasterized image.
    shading_function: a function that takes a dictionary of `[H, W, K]`
      rasterized attribute tensors and returns a `[H, W, 4]` RGBA tensor.
    num_layers: int specifying number of depth layers to composite.
    return_extra_buffers: if True, the function will return raw accumulation
      buffers for visualization.
    backend: A rasterization_backend.RasterizationBackends enum containing the
      backend method to use for rasterization.
    name: A name for this op. Defaults to "rasterize_then_splat".

  Returns:
    a `[A1, ..., An, H, W, 4]` tensor of RGBA values.
  """
  with tf.name_scope(name):
    vertices = tf.convert_to_tensor(vertices)
    triangles = tf.convert_to_tensor(triangles)
    view_projection_matrix = tf.convert_to_tensor(view_projection_matrix)
    shape.check_static(
        tensor=vertices,
        tensor_name='vertices',
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=triangles,
        tensor_name='triangles',
        has_rank=2,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=view_projection_matrix,
        tensor_name='view_projection_matrix',
        has_dim_equals=(((-2, 4), (-1, 4))))

    input_batch_shape = vertices.shape[:-2]
    view_projection_matrix = utils.merge_batch_dims(
        view_projection_matrix, last_axis=-2)
    vertices = utils.merge_batch_dims(vertices, last_axis=-2)
    image_size_backend = (int(image_size[1]), int(image_size[0]))

    # We don't need derivatives of barycentric coordinates for RtS, so use
    # rasterization_backend directly.
    # Back face culling is necessary when rendering multiple layers so that
    # back faces aren't counted as occluding layers.
    rasterized = rasterization_backend.rasterize(
        vertices,
        triangles,
        view_projection_matrix,
        image_size_backend,
        enable_cull_face=True,
        num_layers=num_layers,
        backend=backend)

    # TODO(fcole): check if any of these keys already exist in attributes
    shader_dict = {
        'mask': rasterized.foreground_mask,
        'triangle_indices': rasterized.triangle_id,
        'barycentrics': rasterized.barycentrics.value
    }
    for key, attribute in attributes.items():
      attribute = tf.convert_to_tensor(value=attribute)
      attribute = utils.merge_batch_dims(attribute, last_axis=-2)
      interpolated = interpolate.interpolate_vertex_attribute(
          attribute, rasterized)
      shader_dict[key] = interpolated.value

    # Nested vectorized map over batch and layer dimensions.
    shaded_buffer = tf.vectorized_map(
        lambda l: tf.vectorized_map(shading_function, l), shader_dict)
    # Zero out shader result outside of foreground mask.
    shaded_buffer = shaded_buffer * rasterized.foreground_mask

    clip_space_vertices = utils.transform_homogeneous(view_projection_matrix,
                                                      vertices)
    clip_space_buffer = interpolate.interpolate_vertex_attribute(
        clip_space_vertices, rasterized, (0, 0, 1, 1)).value

    ndc_xyz = clip_space_buffer[..., :3] / clip_space_buffer[..., 3:4]
    image_height, image_width = image_size
    viewport_xyz = (ndc_xyz + 1.0) * tf.constant([image_width, image_height, 1],
                                                 dtype=tf.float32,
                                                 shape=[1, 1, 1, 1, 3]) * 0.5
    output, accum, norm_accum = tf.vectorized_map(splat_at_pixel_centers,
                                                  (viewport_xyz, shaded_buffer))
    if return_extra_buffers:
      return output, accum, norm_accum

    output = utils.restore_batch_dims(output, input_batch_shape)
    return output
