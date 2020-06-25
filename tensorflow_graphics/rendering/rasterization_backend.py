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

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible for inputs `vertices` and `view_projection_matrices`.

  Args:
    vertices: A tensor of shape `[A1, ..., An, V, 3]` containing batches of `V`
      vertices, each defined by a 3D point.
    triangles: A tensor of shape `[T, 3]` containing `T` triangles, each
      associated with 3 vertices from `scene_vertices`
    view_projection_matrices: A tensor of shape `[A1, ..., An, 4, 4]` containing
      batches of view projection matrices
    image_size: An tuple of integers (width, height) containing the dimensions
      in pixels of the rasterized image.
    backend: An enum containing the backend method to use for rasterization.
      Supported options are defined in the RasterizationBackends enum.

  Returns:
    A tuple of 3 elements. The first one of shape `[A1, ..., An, H, W, 1]`
    representing the triangle index associated with each pixel. If no triangle
    is associated to a pixel, the index is set to -1.
    The second element in the tuple is of shape `[A1, ..., An, H, W, 3]` and
    correspond to barycentric coordinates per pixel. The last element in the
    tuple is of shape `[A1, ..., An, H, W]` and stores a value of `0` of the
    pixel is assciated with the background, and `1` with the foreground.
  """
  return _BACKENDS[backend].rasterize(vertices, triangles,
                                      view_projection_matrices, image_size)


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
