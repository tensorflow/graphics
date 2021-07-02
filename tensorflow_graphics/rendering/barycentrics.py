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
"""Set of functions to compute differentiable barycentric coordinates."""

from typing import Tuple

import tensorflow as tf

from tensorflow_graphics.rendering import framebuffer as fb
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias


def differentiable_barycentrics(
    framebuffer: fb.Framebuffer, clip_space_vertices: type_alias.TensorLike,
    triangles: type_alias.TensorLike) -> fb.Framebuffer:
  """Computes differentiable barycentric coordinates from a Framebuffer.

  The barycentric coordinates will be differentiable w.r.t. the input vertices.
  Later, we may support derivatives w.r.t. pixel position for mip-mapping.

  Args:
    framebuffer: a multi-layer Framebuffer containing triangle ids and a
      foreground mask with shape [batch, num_layers, height, width, 1]
    clip_space_vertices: a 2-D float32 tensor with shape [vertex_count, 4] or a
      3-D tensor with shape [batch, vertex_count, 4] containing homogenous
      vertex positions (xyzw).
    triangles: a 2-D int32 tensor with shape [triangle_count, 3] or a 3-D tensor
      with shape [batch, triangle_count, 3] containing per-triangle vertex
      indices in counter-clockwise order.

  Returns:
    a copy of `framebuffer`, but the differentiable barycentric coordinates will
    replace any barycentric coordinates already in the `framebuffer`.
  """
  rank = lambda t: len(t.shape)

  clip_space_vertices = tf.convert_to_tensor(clip_space_vertices)
  shape.check_static(
      tensor=clip_space_vertices,
      tensor_name="clip_space_vertices",
      has_rank_greater_than=1,
      has_rank_less_than=4)
  if rank(clip_space_vertices) == 2:
    clip_space_vertices = tf.expand_dims(clip_space_vertices, axis=0)

  triangles = tf.convert_to_tensor(triangles)
  shape.check_static(
      tensor=triangles,
      tensor_name="triangles",
      has_rank_greater_than=1,
      has_rank_less_than=4)
  if rank(triangles) == 2:
    triangles = tf.expand_dims(triangles, axis=0)
  else:
    shape.compare_batch_dimensions(
        tensors=(clip_space_vertices, triangles),
        last_axes=(-3, -3),
        broadcast_compatible=False)

  shape.compare_batch_dimensions(
      tensors=(clip_space_vertices, framebuffer.triangle_id),
      last_axes=(-3, -4),
      broadcast_compatible=False)

  # Compute image pixel coordinates.
  px, py = normalized_pixel_coordinates(framebuffer.width, framebuffer.height)

  def compute_barycentrics_fn(
      slices: Tuple[type_alias.TensorLike, type_alias.TensorLike,
                    type_alias.TensorLike]
  ) -> tf.Tensor:
    clip_vertices_slice, triangle_slice, triangle_id_slice = slices
    triangle_id_slice = triangle_id_slice[..., 0]
    if rank(triangle_id_slice) == 2:  # There is no layer dimension.
      triangle_id_slice = tf.expand_dims(triangle_id_slice, axis=0)
    # Compute per-triangle inverse matrices.
    triangle_matrices = compute_triangle_matrices(clip_vertices_slice,
                                                  triangle_slice)

    # Compute per-pixel barycentric coordinates.
    barycentric_coords = compute_barycentric_coordinates(
        triangle_id_slice, triangle_matrices, px, py)
    barycentric_coords = tf.transpose(barycentric_coords, perm=[1, 2, 3, 0])
    return barycentric_coords

  per_image_barycentrics = tf.vectorized_map(
      compute_barycentrics_fn,
      (clip_space_vertices, triangles, framebuffer.triangle_id))

  barycentric_coords = tf.stack(per_image_barycentrics, axis=0)
  # After stacking barycentrics will have layers dimension no matter what.
  # In order to make sure we return differentiable barycentrics of the same
  # shape - reshape the tensor using original shape.
  barycentric_coords = tf.reshape(
      barycentric_coords, shape=tf.shape(framebuffer.barycentrics.value))
  # Mask out barycentrics for background pixels.
  barycentric_coords = barycentric_coords * framebuffer.foreground_mask

  return fb.Framebuffer(
      triangle_id=framebuffer.triangle_id,
      vertex_ids=framebuffer.vertex_ids,
      foreground_mask=framebuffer.foreground_mask,
      attributes=framebuffer.attributes,
      barycentrics=fb.RasterizedAttribute(barycentric_coords, None, None))


def normalized_pixel_coordinates(
    image_width: int, image_height: int) -> Tuple[tf.Tensor, tf.Tensor]:
  """Computes the normalized pixel coordinates for the specified image size.

  The x-coordinates will range from -1 to 1 left to right.
  The y-coordinates will range from -1 to 1 top to bottom.
  The extrema +-1 will fall onto the exterior pixel boundaries, while the
  coordinates will be evaluated at pixel centers. So, image of width 4 will have
  normalized pixel x-coordinates at [-0.75 -0.25 0.25 0.75], while image of
  width 3 will have them at [-0.667 0 0.667].

  Args:
    image_width: int specifying desired output image width in pixels.
    image_height: int specifying desired output image height in pixels.

  Returns:
    Two float32 tensors with shape [image_height, image_width] containing x- and
    y- coordinates, respecively, for each image pixel.
  """
  width = tf.cast(image_width, tf.float32)
  height = tf.cast(image_height, tf.float32)
  x_range = (2 * tf.range(width) + 1) / width - 1
  y_range = (2 * tf.range(height) + 1) / height - 1
  x_coords, y_coords = tf.meshgrid(x_range, y_range)
  return x_coords, y_coords


def compute_triangle_matrices(clip_space_vertices: type_alias.TensorLike,
                              triangles: type_alias.TensorLike) -> tf.Tensor:
  """Computes per-triangle matrices used in barycentric coordinate calculation.

  The result corresponds to the inverse matrix from equation (4) in the paper
  "Triangle Scan Conversion using 2D Homogeneous Coordinates". Our matrix
  inverses are not divided by the determinant, only multiplied by its sign. The
  division happens in compute_barycentric_coordinates.

  Args:
    clip_space_vertices: float32 tensor with shape [vertex_count, 4] containing
      vertex positions in clip space (x, y, z, w).
    triangles: 2-D int32 tensor with shape [triangle_count, 3]. Each triplet
      contains a triangle's vertex indices in counter-clockwise order.

  Returns:
    3-D float32 tensor with shape [3, 3, triangle_count] containing per-triangle
    matrices.
  """
  # First make a vertex tensor of size [triangle_count, 3, 3], where the last
  # dimension contains x, y, w coordinates of the corresponding vertex in each
  # triangle
  xyw = tf.stack([
      clip_space_vertices[:, 0], clip_space_vertices[:, 1],
      clip_space_vertices[:, 3]
  ],
                 axis=1)
  xyw = tf.gather(xyw, triangles)
  xyw = tf.transpose(xyw, perm=[0, 2, 1])
  # Compute the sub-determinants.
  d11 = xyw[:, 1, 1] * xyw[:, 2, 2] - xyw[:, 1, 2] * xyw[:, 2, 1]
  d21 = xyw[:, 1, 2] * xyw[:, 2, 0] - xyw[:, 1, 0] * xyw[:, 2, 2]
  d31 = xyw[:, 1, 0] * xyw[:, 2, 1] - xyw[:, 1, 1] * xyw[:, 2, 0]
  d12 = xyw[:, 2, 1] * xyw[:, 0, 2] - xyw[:, 2, 2] * xyw[:, 0, 1]
  d22 = xyw[:, 2, 2] * xyw[:, 0, 0] - xyw[:, 2, 0] * xyw[:, 0, 2]
  d32 = xyw[:, 2, 0] * xyw[:, 0, 1] - xyw[:, 2, 1] * xyw[:, 0, 0]
  d13 = xyw[:, 0, 1] * xyw[:, 1, 2] - xyw[:, 0, 2] * xyw[:, 1, 1]
  d23 = xyw[:, 0, 2] * xyw[:, 1, 0] - xyw[:, 0, 0] * xyw[:, 1, 2]
  d33 = xyw[:, 0, 0] * xyw[:, 1, 1] - xyw[:, 0, 1] * xyw[:, 1, 0]
  matrices = tf.stack([[d11, d12, d13], [d21, d22, d23], [d31, d32, d33]])
  # Multiply by the sign of the determinant, avoiding divide by zero.
  determinant = xyw[:, 0, 0] * d11 + xyw[:, 1, 0] * d12 + xyw[:, 2, 0] * d13
  sign = tf.sign(determinant) + tf.cast(determinant == 0, tf.float32)
  matrices = sign * matrices
  return matrices


def compute_barycentric_coordinates(triangle_ids: type_alias.TensorLike,
                                    triangle_matrices: type_alias.TensorLike,
                                    px: type_alias.TensorLike,
                                    py: type_alias.TensorLike) -> tf.Tensor:
  """Computes per-pixel barycentric coordinates.

  Args:
    triangle_ids: 2-D int tensor with shape [image_height, image_width]
      containing per-pixel triangle ids, as computed by rasterize_triangles.
    triangle_matrices: 3-D float32 tensor with shape [3, 3, triangle_count]
      containing per-triangle matrices computed by compute_triangle_matrices.
    px: 2-D float32 tensor with shape [image_height, image_width] containing
      per-pixel x-coordinates, as computed by normalized_pixel_coordinates.
    py: 2-D float32 tensor with shape [image_height, image_width] containing
      per-pixel y-coordinates, as computed by normalized_pixel_coordinates.

  Returns:
    3-D float32 tensor with shape [height, width, 3] containing the barycentric
    coordinates of the point at each pixel within the triangle specified by
    triangle_ids.
  """
  # Gather per-pixel triangle matrices into m.
  pixel_triangle_matrices = tf.gather(triangle_matrices, triangle_ids, axis=-1)
  # Compute barycentric coordinates by evaluating edge equations.
  barycentric_coords = (
      pixel_triangle_matrices[:, 0] * px + pixel_triangle_matrices[:, 1] * py +
      pixel_triangle_matrices[:, 2])
  # Normalize so the barycentric coordinates sum to 1. Guard against division
  # by zero in the case that the barycentrics sum to zero, which can happen for
  # background pixels when the 0th triangle in the list is degenerate, due to
  # the way we use triangle id 0 for both background and the first triangle.
  barycentric_coords = tf.math.divide_no_nan(
      barycentric_coords, tf.reduce_sum(barycentric_coords, axis=0))
  return barycentric_coords
