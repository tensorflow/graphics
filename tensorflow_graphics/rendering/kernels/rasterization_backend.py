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
"""CPU rasterization backend."""

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import enum
from typing import Tuple

from six.moves import range
import tensorflow as tf

from tensorflow_graphics.rendering import framebuffer as fb
from tensorflow_graphics.rendering import utils
from tensorflow_graphics.util import shape
from tensorflow_graphics.util import type_alias

# pylint: disable=g-import-not-at-top
try:
  from tensorflow_graphics.rendering.kernels import gen_rasterizer_op as render_ops
except ImportError:
  import os
  dir_path = os.path.dirname(os.path.abspath(__file__))
  render_ops = tf.load_op_library(os.path.join(dir_path, "rasterizer_op.so"))
# pylint: enable=g-import-not-at-top


class FaceCullingMode(enum.IntEnum):
  # The values of this enum must match those in rasterize_triangles_impl.h.
  NONE = 0
  BACK = 1
  FRONT = 2


def rasterize(
    vertices: type_alias.TensorLike,
    triangles: type_alias.TensorLike,
    view_projection_matrices: type_alias.TensorLike,
    image_size: Tuple[int, int],
    enable_cull_face: bool,
    num_layers: int,
    name: str = "rasterization_backend_cpu_rasterize") -> fb.Framebuffer:
  """Rasterizes the scene.

    This rasterizer estimates which triangle is associated with each pixel using
    the C++ software rasterizer.

  Args:
    vertices: A tensor of shape `[batch, num_vertices, 3]` containing batches of
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[num_triangles, 3]` containing triangles, each
      associated with 3 vertices from `scene_vertices`.
    view_projection_matrices: A tensor of shape `[batch, 4, 4]` containing
      batches of view projection matrices.
    image_size: An tuple of integers (width, height) containing the dimensions
      in pixels of the rasterized image.
    enable_cull_face: A boolean, which will enable BACK face culling when True
      and no face culling when False.
    num_layers: Number of depth layers to render.
    name: A name for this op. Defaults to "rasterization_backend_cpu_rasterize".

  Returns:
    A Framebuffer containing the rasterized values: barycentrics, triangle_id,
    foreground_mask, vertex_ids. Returned Tensors have shape
    [batch, num_layers, height, width, channels]
    Note: triangle_id contains the triangle id value for each pixel in the
    output image. For pixels within the mesh, this is the integer value in the
    range [0, num_vertices] from triangles. For vertices outside the mesh this
    is 0; 0 can either indicate belonging to triangle 0, or being outside the
    mesh. This ensures all returned triangle ids will validly index into the
    vertex array, enabling the use of tf.gather with indices from this tensor.
    The barycentric coordinates can be used to determine pixel validity instead.
    See framebuffer.py for a description of the Framebuffer fields.
  """
  with tf.name_scope(name):
    vertices = tf.convert_to_tensor(value=vertices)
    triangles = tf.convert_to_tensor(value=triangles)
    view_projection_matrices = tf.convert_to_tensor(
        value=view_projection_matrices)

    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_rank=3,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_rank=2,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=view_projection_matrices,
        tensor_name="view_projection_matrices",
        has_rank=3,
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.compare_batch_dimensions(
        tensors=(vertices, view_projection_matrices),
        tensor_names=("vertices", "view_projection_matrices"),
        last_axes=(-3, -3),
        broadcast_compatible=True)

    if not num_layers > 0:
      raise ValueError("num_layers must be > 0.")

    vertices = utils.transform_homogeneous(view_projection_matrices, vertices)
    batch_size = tf.compat.dimension_value(vertices.shape[0])

    per_image_barycentrics = []
    per_image_triangle_ids = []
    per_image_masks = []
    image_width, image_height = image_size
    face_culling_mode = FaceCullingMode.BACK if enable_cull_face else FaceCullingMode.NONE
    for batch_index in range(batch_size):
      clip_vertices_slice = vertices[batch_index, ...]

      barycentrics, triangle_ids, z_buffer = (
          render_ops.rasterize_triangles(clip_vertices_slice, triangles,
                                         image_width, image_height, num_layers,
                                         face_culling_mode))
      # Shape [num_layers, image_height, image_width, 3]
      barycentrics = tf.stop_gradient(barycentrics)

      barycentrics = tf.ensure_shape(barycentrics,
                                     [num_layers, image_height, image_width, 3])
      triangle_ids = tf.ensure_shape(triangle_ids,
                                     [num_layers, image_height, image_width])
      z_buffer = tf.ensure_shape(z_buffer,
                                 [num_layers, image_height, image_width])

      mask = tf.cast(tf.not_equal(z_buffer, 1.0), tf.float32)

      per_image_barycentrics.append(barycentrics)
      per_image_triangle_ids.append(triangle_ids)
      per_image_masks.append(mask)

    # Shape: [batch_size, num_layers, image_height, image_width, 1]
    triangle_id = tf.expand_dims(
        tf.stack(per_image_triangle_ids, axis=0), axis=-1)

    # Shape: [batch_size, num_layers, image_height, image_width, 3]
    vertex_ids = tf.gather(triangles, triangle_id[..., 0])

    # Shape: [batch_size, num_layers, image_height, image_width, 1]
    mask = tf.expand_dims(tf.stack(per_image_masks, axis=0), axis=-1)

    # Shape: [batch_size, num_layers, image_height, image_width, 3]
    barycentrics = tf.stack(per_image_barycentrics, axis=0)

    return fb.Framebuffer(
        foreground_mask=mask,
        triangle_id=triangle_id,
        vertex_ids=vertex_ids,
        barycentrics=fb.RasterizedAttribute(
            value=barycentrics, d_dx=None, d_dy=None))
