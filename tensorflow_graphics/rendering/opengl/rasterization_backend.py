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

from tensorflow_graphics.rendering.opengl import gen_rasterizer_op as render_ops
from tensorflow_graphics.util import export_api
from tensorflow_graphics.util import shape


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

in int gl_PrimitiveIDIn;
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
  output_color = vec4(round(triangle_index + 1.0), barycentric_coordinates, 1.0);
}
"""


def rasterize(vertices,
              triangles,
              view_projection_matrices,
              image_size,
              name=None):
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
    name: A name for this op. Defaults to 'rasterization_backend_rasterize'.

  Returns:
    A tuple of 3 elements. The first one of shape `[A1, ..., An, H, W, 1]`
    representing the triangle index associated with each pixel. If no triangle
    is associated to a pixel, the index is set to -1.
    The second element in the tuple is of shape `[A1, ..., An, H, W, 3]` and
    correspond to barycentric coordinates per pixel. The last element in the
    tuple is of shape `[A1, ..., An, H, W]` and stores a value of `0` of the
    pixel is assciated with the background, and `1` with the foreground.
  """
  with tf.compat.v1.name_scope(name, "rasterization_backend_rasterize",
                               (vertices, triangles, view_projection_matrices)):
    vertices = tf.convert_to_tensor(value=vertices)
    triangles = tf.convert_to_tensor(value=triangles)
    view_projection_matrices = tf.convert_to_tensor(
        value=view_projection_matrices)

    shape.check_static(
        tensor=vertices,
        tensor_name="vertices",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=triangles,
        tensor_name="triangles",
        has_rank=2,
        has_dim_equals=((-1, 3)))
    shape.check_static(
        tensor=view_projection_matrices,
        tensor_name="view_projection_matrices",
        has_rank_greater_than=1,
        has_dim_equals=((-1, 4), (-2, 4)))
    shape.compare_batch_dimensions(
        tensors=(vertices, view_projection_matrices),
        tensor_names=("vertices", "view_projection_matrices"),
        last_axes=(-3, -3),
        broadcast_compatible=True)

    common_batch_shape = shape.get_broadcasted_shape(
        vertices.shape[:-2], view_projection_matrices.shape[:-2])
    common_batch_shape = [_dim_value(dim) for dim in common_batch_shape]
    vertices = tf.broadcast_to(vertices,
                               common_batch_shape + vertices.shape[-2:])
    view_projection_matrices = tf.broadcast_to(view_projection_matrices,
                                               common_batch_shape + [4, 4])

    geometry = tf.gather(vertices, triangles, axis=-2)

    rasterized = render_ops.rasterize(
        num_points=geometry.shape[-3],
        alpha_clear=0.0,
        enable_cull_face=True,
        variable_names=("view_projection_matrix", "triangular_mesh"),
        variable_kinds=("mat", "buffer"),
        variable_values=(view_projection_matrices,
                         tf.reshape(geometry, shape=common_batch_shape + [-1])),
        output_resolution=image_size,
        vertex_shader=vertex_shader,
        geometry_shader=geometry_shader,
        fragment_shader=fragment_shader)

    triangle_index = tf.cast(rasterized[..., 0], tf.int32) - 1
    barycentric_coordinates = rasterized[..., 1:3]
    barycentric_coordinates = tf.concat(
        (barycentric_coordinates, 1.0 - barycentric_coordinates[..., 0:1] -
         barycentric_coordinates[..., 1:2]),
        axis=-1)
    mask = tf.cast(rasterized[..., 3], tf.int32)

    return triangle_index, barycentric_coordinates, mask


# API contains all public functions and classes.
__all__ = export_api.get_functions_and_classes()
