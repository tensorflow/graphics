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
"""Rasterization backends selector for TF Graphics."""

import enum

from tensorflow_graphics.rendering.opengl import rasterization_backend as gl_backend
from tensorflow_graphics.util import export_api


class RasterizationBackends(enum.Enum):
  OPENGL = 0


_BACKENDS = {
    RasterizationBackends.OPENGL: gl_backend,
}


def rasterize(vertices,
              triangles,
              view_projection_matrices,
              image_size,
              backend=RasterizationBackends.OPENGL):
  """Rasterizes the scene.

    This rasterizer estimates which triangle is associated with each pixel using
    OpenGL.

  Args:
    vertices: A tensor of shape `[batch, num_vertices, 3]` containing batches of
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[num_triangles, 3]` containing triangles, each
      associated with 3 vertices from `scene_vertices`
    view_projection_matrices: A tensor of shape `[batch, 4, 4]` containing
      batches of view projection matrices
    image_size: An tuple of integers (width, height) containing the dimensions
      in pixels of the rasterized image.
    backend: An enum containing the backend method to use for rasterization.
      Supported options are defined in the RasterizationBackends enum.

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
  return _BACKENDS[backend].rasterize(vertices, triangles,
                                      view_projection_matrices, image_size)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
