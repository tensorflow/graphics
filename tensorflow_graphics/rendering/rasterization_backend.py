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
import importlib
from typing import Tuple

from tensorflow_graphics.rendering import framebuffer as fb
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import type_alias


class RasterizationBackends(enum.Enum):
  OPENGL = 0
  CPU = 1


def rasterize(
    vertices: type_alias.TensorLike,
    triangles: type_alias.TensorLike,
    view_projection_matrices: type_alias.TensorLike,
    image_size: Tuple[int, int],
    enable_cull_face: bool = True,
    num_layers: int = 1,
    backend: enum.Enum = RasterizationBackends.OPENGL) -> fb.Framebuffer:
  """Rasterizes the scene.

    This rasterizer estimates which triangle is associated with each pixel.

  Args:
    vertices: A tensor of shape `[batch, num_vertices, 3]` containing batches of
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[num_triangles, 3]` containing triangles, each
      associated with 3 vertices from `vertices`.
    view_projection_matrices: A tensor of shape `[batch, 4, 4]` containing
      batches of view projection matrices.
    image_size: A tuple of integers (width, height) containing the dimensions in
      pixels of the rasterized image.
    enable_cull_face: A boolean, which will enable BACK face culling when True
      and no face culling when False.
    num_layers: Number of depth layers to render. Output tensors shape depends
      on whether num_layers=1 or not. Supported by CPU rasterizer only and does
      nothing for OpenGL backend.
    backend: An enum containing the backend method to use for rasterization.
      Supported options are defined in the RasterizationBackends enum.

  Raises:
    KeyError: if backend is not part of supported rasterization backends.

  Returns:
    A Framebuffer containing the rasterized values: barycentrics, triangle_id,
    foreground_mask, vertex_ids. Returned Tensors have shape
    [batch, num_layers, height, width, channels].
    Note: triangle_id contains the triangle id value for each pixel in the
    output image. For pixels within the mesh, this is the integer value in the
    range [0, num_vertices] from triangles. For vertices outside the mesh this
    is 0; 0 can either indicate belonging to triangle 0, or being outside the
    mesh. This ensures all returned triangle ids will validly index into the
    vertex array, enabling the use of tf.gather with indices from this tensor.
    The barycentric coordinates can be used to determine pixel validity instead.
    See framebuffer.py for a description of the Framebuffer fields.
  """
  if backend == RasterizationBackends.CPU:
    backend_module = importlib.import_module(
        "tensorflow_graphics.rendering.kernels.rasterization_backend")
  elif backend == RasterizationBackends.OPENGL:
    backend_module = importlib.import_module(
        "tensorflow_graphics.rendering.opengl.rasterization_backend")
  else:
    raise KeyError("Backend is not supported: %s." % backend)

  return backend_module.rasterize(vertices, triangles, view_projection_matrices,
                                  image_size, enable_cull_face, num_layers)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
