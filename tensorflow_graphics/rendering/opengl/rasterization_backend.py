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
"""OpenGL rasterization backend for TF Graphics."""

import tensorflow as tf

from tensorflow_graphics.rendering import framebuffer as fb
from tensorflow_graphics.rendering.opengl import math
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape

# pylint: disable=g-import-not-at-top
try:
  from tensorflow_graphics.rendering.opengl import gen_rasterizer_op as render_ops
except ImportError:
  import os
  dir_path = os.path.dirname(os.path.abspath(__file__))
  render_ops = tf.load_op_library(os.path.join(dir_path, "rasterizer_op.so"))
# pylint: enable=g-import-not-at-top


def _dim_value(dim):
  return 1 if dim is None else tf.compat.v1.dimension_value(dim)


# Empty vertex shader; all the work happens in the geometry shader.
vertex_shader = """
#version 430
void main() { }
"""

# Geometry shader that projects the vertices of visible triangles onto the image
# plane.
geometry_shader = """
#version 430

uniform mat4 view_projection_matrix;

layout(points) in;
layout(triangle_strip, max_vertices=3) out;

out layout(location = 0) vec2 barycentric_coordinates;
out layout(location = 1) float triangle_index;

layout(binding=0) buffer triangular_mesh { float mesh_buffer[]; };


vec3 get_vertex_position(int vertex_index) {
  // Triangles are packed as 3 consecuitve vertices, each with 3 coordinates.
  int offset = gl_PrimitiveIDIn * 9 + vertex_index * 3;
  return vec3(mesh_buffer[offset], mesh_buffer[offset + 1],
    mesh_buffer[offset + 2]);
}


void main() {
  vec3 positions[3] = {get_vertex_position(0), get_vertex_position(1),
                        get_vertex_position(2)};
  vec4 projected_vertices[3] = {
                            view_projection_matrix * vec4(positions[0], 1.0),
                            view_projection_matrix * vec4(positions[1], 1.0),
                            view_projection_matrix * vec4(positions[2], 1.0)};

  for (int i = 0; i < 3; ++i) {
    // gl_Position is a pre-defined size 4 output variable.
    gl_Position = projected_vertices[i];
    barycentric_coordinates = vec2(i==0 ? 1.0 : 0.0, i==1 ? 1.0 : 0.0);
    triangle_index = gl_PrimitiveIDIn;

    EmitVertex();
  }
  EndPrimitive();
}
"""

# Fragment shader that packs barycentric coordinates, and triangle index.
fragment_shader = """
#version 430

in layout(location = 0) vec2 barycentric_coordinates;
in layout(location = 1) float triangle_index;

out vec4 output_color;

void main() {
  output_color = vec4(round(triangle_index), barycentric_coordinates, 1.0);
}
"""


def _tile_to_image_size(tensor, image_shape):
  """Inserts `image_shape` dimensions after `tensor` batch dimension."""
  non_batch_dims = len(tensor.shape) - 1
  for _ in image_shape:
    tensor = tf.expand_dims(tensor, axis=1)
  tensor = tf.tile(tensor, [1] + image_shape + [1] * non_batch_dims)
  return tensor


def _perspective_correct_barycentrics(vertices_per_pixel, model_to_eye_matrix,
                                      perspective_matrix, height, width):
  """Creates the pixels grid and computes barycentrics."""
  # Construct the pixel grid with half-integer pixel centers.
  width = float(width)
  height = float(height)
  px = tf.linspace(0.5, width - 0.5, num=int(width))
  py = tf.linspace(0.5, height - 0.5, num=int(height))
  xv, yv = tf.meshgrid(px, py)
  pixel_position = tf.stack((xv, yv), axis=-1)

  # Since `model_to_eye_matrix` is defined per batch, while vertices in
  # `vertices_per_pixel` are defined per batch per pixel, we have to make them
  # broadcast-compatible. In other words we want to tile matrices per vertex
  # per pixel.
  image_shape = vertices_per_pixel.shape[1:-1]
  model_to_eye_matrix = _tile_to_image_size(model_to_eye_matrix, image_shape)
  perspective_matrix = _tile_to_image_size(perspective_matrix, image_shape)

  return math.perspective_correct_barycentrics(vertices_per_pixel,
                                               pixel_position,
                                               model_to_eye_matrix,
                                               perspective_matrix,
                                               (width, height))


def rasterize(vertices,
              triangles,
              model_to_eye_matrix,
              perspective_matrix,
              image_size,
              name=None):
  """Rasterizes the scene.

    This rasterizer estimates which triangle is associated with each pixel using
    OpenGL.

  Note:
    In the following, A1 to An are optional batch dimensions which must be
    broadcast compatible for inputs `vertices` and `view_projection_matrices`.

  Args:
    vertices: A tensor of shape `[batch_size, num_vertices, 3]` containing
      batches vertices, each defined by a 3D point.
    triangles: A tensor of shape `[num_triangles, 3]` each associated with 3
      vertices from `scene_vertices`
    model_to_eye_matrix: A tensor of shape `[batch_size, 4, 4]` containing
      batches of matrices used to transform vertices from model to eye
      coordinates.
    perspective_matrix: A tensor of shape `[batch_size, 4, 4]` containing
      batches of matrices used to project vertices from eye to clip coordinates.
    image_size: An tuple of integers (width, height) containing the dimensions
      in pixels of the rasterized image.
    name: A name for this op. Defaults to 'rasterization_backend_rasterize'.

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
  with tf.compat.v1.name_scope(
      name, "rasterization_backend_rasterize",
      (vertices, triangles, model_to_eye_matrix, perspective_matrix)):
    vertices = tf.convert_to_tensor(value=vertices)
    triangles = tf.convert_to_tensor(value=triangles)
    model_to_eye_matrix = tf.convert_to_tensor(value=model_to_eye_matrix)
    perspective_matrix = tf.convert_to_tensor(value=perspective_matrix)

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
        tensor=perspective_matrix,
        tensor_name="perspective_matrix",
        has_rank=3,
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.check_static(
        tensor=model_to_eye_matrix,
        tensor_name="model_to_eye_matrix",
        has_rank=3,
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.compare_batch_dimensions(
        tensors=(vertices, perspective_matrix, model_to_eye_matrix),
        tensor_names=("vertices", "perspective_matrix", "model_to_eye_matrix"),
        last_axes=(-3, -3, -3),
        broadcast_compatible=True)

    view_projection_matrix = tf.linalg.matmul(perspective_matrix,
                                              model_to_eye_matrix)

    geometry = tf.gather(vertices, triangles, axis=-2)

    # Extract batch size in order to make sure it is preserved after `gather`
    # operation.
    batch_size = _dim_value(vertices.shape[0])

    rasterized = render_ops.rasterize(
        num_points=geometry.shape[-3],
        alpha_clear=0.0,
        enable_cull_face=True,
        variable_names=("view_projection_matrix", "triangular_mesh"),
        variable_kinds=("mat", "buffer"),
        variable_values=(view_projection_matrix,
                         tf.reshape(geometry, shape=[batch_size, -1])),
        output_resolution=image_size,
        vertex_shader=vertex_shader,
        geometry_shader=geometry_shader,
        fragment_shader=fragment_shader)

    triangle_index = tf.cast(rasterized[..., 0], tf.int32)
    # Slicing of the tensor will result in all batch dimensions being
    # `None` for tensorflow graph mode, therefore we have to fix it in order to
    # have explicit shape.
    width, height = image_size
    triangle_index = tf.reshape(triangle_index, [batch_size, height, width, 1])
    vertex_ids = tf.gather(triangles, triangle_index[..., 0], batch_dims=0)

    # Compute perspective-corrected barycentric coordinates.
    vertices_per_pixel = tf.gather(vertices, vertex_ids, batch_dims=1)
    barycentrics = _perspective_correct_barycentrics(vertices_per_pixel,
                                                     model_to_eye_matrix,
                                                     perspective_matrix,
                                                     height, width)
    mask = tf.cast(rasterized[..., 3], tf.int32)
    mask = tf.reshape(mask, [batch_size, height, width, 1])

    return fb.Framebuffer(
        foreground_mask=mask,
        triangle_id=triangle_index,
        vertex_ids=vertex_ids,
        barycentrics=fb.RasterizedAttribute(
            value=barycentrics, d_dx=None, d_dy=None))


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
